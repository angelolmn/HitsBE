from hitsbe.models import hitsBERT
from hitsbe import Hitsbe

from aeon.datasets import load_classification

import numpy as np
import torch

from torch.utils.data import DataLoader, TensorDataset


#establecer semilla de torch

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


hb = Hitsbe("experiments/vocabulary/uniformwords/50kv2.vocab")

from transformers import BertConfig
config = BertConfig(
    vocab_size=len(hb.vocabulary),               
    hidden_size=hb.dim_model,              
    num_hidden_layers=6,
    num_attention_heads=4,
    max_position_embeddings=hb.dim_seq + 1,  # 128 segments + CLS
    pad_token_id=0
)

model_base = hitsBERT.HitsBERT(bert_config=config, hitsbe=hb)
model_pretraining = hitsBERT.HitsBERTPretraining(model_base)

if not torch.cuda.is_available():
    sys.exit("ERROR: No GPU detected.")

device = torch.device("cuda")

# Initialize the model and move it to the device
model_pretraining.to(device)

X_train, _ = load_classification("Mallat", split="train")
X_train_shaped = np.squeeze(X_train, axis=1)
X_train_shaped = torch.tensor(X_train_shaped)

dim_seq = model_pretraining.model.hitsbe.dim_seq
# Create masks
wordfreq = [1,2,4,8]

MLM_catalog = []

for f in wordfreq:
    rows = int(dim_seq/f)
    mask = torch.zeros(rows, dim_seq)
    index = torch.arange(rows)

    for j in range(f):
        mask[index, f*index+j] = 1

    MLM_catalog.append(mask)


# Pretraining. Hay que optimizar los bucles --------------------------

def similar_option(hitsbe, mask, x):
    x_mod = x.clone()
    word_len = hitsbe.segment_len

    nonzero_mask_index = torch.nonzero(mask, as_tuple = True)[0]

    # Select method
    method = torch.randint(0, 4, (1,)).item()
    match method:
        # 1.- Same monotony
        case 0:
            for i in nonzero_mask_index:
                # Compute index monotony
                mono_index = hitsbe.vocabulary._compute_index(x[i*word_len: (i+1)*word_len])

                # Get another word with same monotony
                num_words = len(hitsbe.vocabulary.words[mono_index])
                word_index = torch.randint(0, num_words, (1,)).item()

                x_mod[i*word_len: (i+1)*word_len] = torch.tensor(hitsbe.vocabulary.words[mono_index][word_index])
            
        # 2.- Inverse monotony
        case 1:
            for i in nonzero_mask_index:
                # Compute index monotony
                mono_index = hitsbe.vocabulary._compute_index(x[i*word_len: (i+1)*word_len])

                # Compute complement 1 monotony
                bit_mask = 0b1111111
                comp_mono_index = ~mono_index & bit_mask

                # Get another word with inverse monotony
                num_words = len(hitsbe.vocabulary.words[comp_mono_index])
                word_index = torch.randint(0, num_words, (1,)).item()
                
                x_mod[i*word_len: (i+1)*word_len] = torch.tensor(hitsbe.vocabulary.words[comp_mono_index][word_index])
                        
        # 3.- Same freq words
        case 2:
            num_ones = torch.sum(mask)
            shift = torch.randint(1, int(mask.size(0)/num_ones), (1,)).item()

            for i in nonzero_mask_index:
                idx = (i + shift) % mask.size(0)
                x_mod[i*word_len: (i+1)*word_len] = x[idx*word_len: (idx + 1)*word_len]
            
        # 4.- Linear transformation (for Haar embeddings)
        case 3:
            x_min = torch.min(x)
            x_max = torch.max(x)

            scale = x_max - x_min

            shift = torch.empty(1).uniform_(-0.1 * scale, 0.1 * scale).item()
            scalar = torch.empty(1).uniform_(0.5, 2).item()
            
            for i in nonzero_mask_index:
                x_mod[i*word_len: (i+1)*word_len] = x[i*word_len: (i+1)*word_len]*scalar + shift
            
    return x_mod

# mask is a torch tensor
def random_option(hitsbe, mask, x):
    x_mod = x.clone()
    
    nonzero_mask_index = torch.nonzero(mask, as_tuple = True)[0]

    num_monotony_classes = len(hitsbe.vocabulary.words)
    word_len = hitsbe.segment_len

    # For scale

    for i in nonzero_mask_index:
        # generar aleatoriamente un indice de monotonia
        mono_index = torch.randint(0, num_monotony_classes, (1,)).item()

        # generar aleatoriamente un indice dentro de la monotonia
        num_words = len(hitsbe.vocabulary.words[mono_index])
        word_index = torch.randint(0, num_words, (1,)).item()

        # Se asigna la nueva palabra
        x_mod[i*word_len: (i+1)*word_len] = torch.tensor(hitsbe.vocabulary.words[mono_index][word_index])

        # Se transforma
        segment = x[i*word_len: (i+1)*word_len]
        x_min = torch.min(segment)
        x_max = torch.max(segment)
        scale = x_max - x_min

        x_mod[i*word_len: (i+1)*word_len] = x_mod[i*word_len: (i+1)*word_len]*scale + x_min

    return x_mod


model_pretraining.train()

optimizer = torch.optim.AdamW(model_pretraining.parameters(), lr=1e-3)

accuracy= []

# Para cada serie dentro del batch. Lo haria simplemente 1 a 1. 
for i,x in enumerate(X_train_shaped):
    type(x)
    options = []
    solutions_index = []
    MLM = []

    optimizer.zero_grad()

    # Create test for the series
    for j,freq in enumerate(wordfreq):
        # Quizas esta desbalanceado?
        # Create options
        for _ in range(int(8/freq)):
            # Se selcciona una mascara 
            mask = MLM_catalog[j][torch.randint(0, int(dim_seq/freq), (1,)).item()]
            MLM.append(mask)

            # Se crea opcion similar
            sim_option = similar_option(model_pretraining.model.hitsbe, mask, x)

            # Se crea opcion completamente aleatoria
            rand_option = random_option(model_pretraining.model.hitsbe, mask, x) 

            stacked_options = torch.stack([x, sim_option, rand_option])

            # Shuffle the options
            perm = torch.randperm(3) 
            shuffled_options = stacked_options[perm]

            # The original solution index was at 0
            solutions_index.append(perm[0])

            options.extend(shuffled_options)

    options = torch.stack(options)  # (num_iter * 3, dim_seq)
    options = options.view(-1, options.shape[-1])

    embeddings, _ = model_pretraining.model.hitsbe.get_embedding(options)

    options_group_embed = embeddings.view(-1, 3, dim_seq, model_pretraining.model.hitsbe.dim_model)
    
    # create options
    MLM = torch.stack(MLM)
    solutions_index = torch.tensor(solutions_index, device=device)

    # batch: time serie (1, ts_len)
    # MLM: binary tensor with 1 at [MASK] token (num_iter, dim_seq, 1)
    # options: (num_iter, num_options, seq_length, dim_model)
    outputs_logits = model_pretraining( x.unsqueeze(0), MLM, options_group_embed)

    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(outputs_logits, solutions_index)

    predicted_index = torch.argmax(outputs_logits, dim=1)
    correct = (predicted_index == solutions_index).sum().item()
    total = solutions_index.size(0)
    accuracy.append(correct / total)
    print("Accuracy: " + str(accuracy[-1]))

    # Compute gradients
    loss.backward()
    # Update weights
    optimizer.step()
    print("Weights updated")

print(accuracy)