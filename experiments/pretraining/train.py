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


import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--deepspeed", type=str, help="Ruta al archivo de configuración DeepSpeed")
    return parser.parse_args()

def main():
    # Load config arguments
    args = parse_args()

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

    base_model = hitsBERT.HitsBERT(bert_config=config, hitsbe=hitsbe)
    base_model = base_model.half()
    model = hitsBERT.HitsBERTPretraining(model=base_model)

    # Prepare optimizer and scheduler
    micro_batch_size = 200  # debe coincidir con train_batch_size / num_gpus
    
    train_loader = get_dataloader(split="ETTh1_",batch_size=micro_batch_size)
    steps_per_epoch = len(train_loader)
    total_training_steps = args.epochs * steps_per_epoch
    warmup_steps = int(0.06 * total_training_steps)

    # https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html#torch.optim.AdamW
    optimizer = AdamW(
        model.parameters(),
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=1e-2
    )

    # https://www.kaggle.com/code/isbhargav/guide-to-pytorch-learning-rate-scheduling#8.OneCycleLR---linear
    # https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html#torch.optim.lr_scheduler.OneCycleLR
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,             
                                                    max_lr=2e-3, 
                                                    epochs=2,
                                                    steps_per_epoch=1050//150, # Cambiar
                                                    pct_start=0.1, # % steps of total steps for warmup 
                                                    div_factor = 1e3, # initial_lr = max_lr/div_factor
                                                    final_div_factor = 25, # minimum lr initial_lr/final_div_factor
                                                    anneal_strategy='linear')

    # Initialize DeepSpeed
    model_engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        optimizer=optimizer,
        config=args.deepspeed,  # <- aquí usas el JSON que tú pasas
        args=args
    )



    # Training loop
    for epoch in range(args.epochs):
        train_one_epoch(model_engine, train_loader, epoch, scheduler)
        model_engine.save_checkpoint("checkpoints/", tag=f"epoch{epoch}")

if __name__ == "__main__":
    main()
