# poetry run deepspeed experiments/pretraining/train.py   --deepspeed experiments/pretraining/deepspeed_config.json

import torch
import deepspeed
from torch.utils.data import DataLoader
from hitsbe.models import hitsBERT
from hitsbe import Hitsbe, HitsbeConfig
from hitsbe import Vocabulary
from transformers import BertConfig
from dataset import get_dataloader, MultipleChoiceTimeSeriesDataset
from utils import train_one_epoch
from transformers import get_cosine_schedule_with_warmup
from torch.optim import AdamW
import time
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--deepspeed", type=str, help="Ruta al archivo de configuración DeepSpeed")
    return parser.parse_args()

def main():
    # Load config arguments
    args = parse_args()
    
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    # Load model and configuration
    filename = "experiments/vocabulary/uniformwords/50kv2.vocab"
    vocabulary = Vocabulary(filename)

    ts_len = 1024
    dim_model = 768
    dim_segment = 8
    max_haar_depth = 8

    hitsbe_config = HitsbeConfig(
        vocabulary_path=filename, 
        ts_len=ts_len,
        dim_model=dim_model, 
        dim_segment=dim_segment, 
        max_haar_depth=max_haar_depth
    )

    hitsbe = Hitsbe(hitsbe_config=hitsbe_config)

    config = BertConfig(
        vocab_size=len(hitsbe.vocabulary),               
        hidden_size=hitsbe.dim_model,              
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=4 * hitsbe.dim_model,
        max_position_embeddings=hitsbe.dim_seq + 1,
        pad_token_id=0,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1
    )

    base_model = hitsBERT.HitsBERT(bert_config=config, hitsbe_config=hitsbe_config)
    model = hitsBERT.HitsBERTPretraining(model=base_model)
    #model.half()
    
    # Initialize DeepSpeed
    model_engine, _, _, _ = deepspeed.initialize(
        model=model,
        config=args.deepspeed,
        args=args
    )

    micro_batch_size = 32  # train_batch_size / num_gpus / gradient_accumulate_steps
    train_loader = get_dataloader(split="",batch_size=micro_batch_size)

    n_batches = len(train_loader)
    grad_steps = model_engine.gradient_accumulation_steps()

    print(f"Num batches por época: {len(train_loader)}")
    print(f"Micro-batch size: {model_engine.train_micro_batch_size_per_gpu()}")
    print(f"Grad Accumulation Steps: {model_engine.gradient_accumulation_steps()}")
    print(f"Global batch size: {model_engine.train_batch_size()}")
    
    start = time.time()
    # Training loop
    for epoch in range(args.epochs):
        train_one_epoch(model_engine, train_loader, epoch)
   
    end = time.time()
    print("Tiempo de ejecucion: " + str(end - start))

    model_engine.module.model.save_pretrained("experiments/pretraining/savemodel")

if __name__ == "__main__":
    main()
