"""
"ETTh1", "ETTh2", "ETTm1", "ETTm2", "ECL_1", "Exchange", "ILI", "Weather"
"""

# --------------------------------------------------------------
import torch
import pandas as pd
from datasetsforecast.long_horizon import LongHorizon
from hitsbe import Vocabulary
vocab = Vocabulary("experiments/vocabulary/uniformwords/50kv2.vocab")
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence


def similar_option(vocabulary: Vocabulary, mask, x):
    x_mod = x.clone()
    word_len = vocabulary.domain_len
    nonzero_mask_index = torch.nonzero(mask, as_tuple = True)[0]
    # Select method
    method = torch.randint(0, 4, (1,)).item()
    match method:
        # 1.- Same monotony
        case 0:
            for i in nonzero_mask_index:
                # Compute index monotony
                mono_index = vocabulary._compute_index(x[i*word_len: (i+1)*word_len])
                # Get another word with same monotony
                num_words = len(vocabulary.words[mono_index])
                word_index = torch.randint(0, num_words, (1,)).item()
                x_mod[i*word_len: (i+1)*word_len] = torch.tensor(vocabulary.words[mono_index][word_index])
        # 2.- Inverse monotony
        case 1:
            for i in nonzero_mask_index:
                # Compute index monotony
                mono_index = vocabulary._compute_index(x[i*word_len: (i+1)*word_len])
                # Compute complement 1 monotony
                bit_mask = 0b1111111
                comp_mono_index = ~mono_index & bit_mask
                # Get another word with inverse monotony
                num_words = len(vocabulary.words[comp_mono_index])
                word_index = torch.randint(0, num_words, (1,)).item()
                x_mod[i*word_len: (i+1)*word_len] = torch.tensor(vocabulary.words[comp_mono_index][word_index])
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

def random_option(vocabulary, mask, x):
    x_mod = x.clone()
    nonzero_mask_index = torch.nonzero(mask, as_tuple = True)[0]
    num_monotony_classes = len(vocabulary.words)
    word_len = vocabulary.domain_len
    for i in nonzero_mask_index:
        # generar aleatoriamente un indice de monotonia
        mono_index = torch.randint(0, num_monotony_classes, (1,)).item()
        # generar aleatoriamente un indice dentro de la monotonia
        num_words = len(vocabulary.words[mono_index])
        word_index = torch.randint(0, num_words, (1,)).item()
        # Se asigna la nueva palabra
        x_mod[i*word_len: (i+1)*word_len] = torch.tensor(vocabulary.words[mono_index][word_index])
        # Se transforma
        segment = x[i*word_len: (i+1)*word_len]
        x_min = torch.min(segment)
        x_max = torch.max(segment)
        scale = x_max - x_min
        x_mod[i*word_len: (i+1)*word_len] = x_mod[i*word_len: (i+1)*word_len]*scale + x_min
    return x_mod

def center_pad(seq, target_len, pad_val):
    seq_len = len(seq)
    total_pad = target_len - seq_len
    pad_left = total_pad // 2
    pad_right = total_pad - pad_left
    return [pad_val] * pad_left + seq + [pad_val] * pad_right

dim_seq=128
ts_size = 1024

wordfreq = [1,2,4,8]
MLM_catalog = []

for f in wordfreq:
    rows = int(dim_seq/f)
    mask = torch.zeros(rows, dim_seq)
    index = torch.arange(rows)
    for j in range(f):
        mask[index, f*index+j] = 1
    MLM_catalog.append(mask)

monash_configs = [
    "weather", "tourism_monthly",
    "bitcoin", "vehicle_trips", "nn5_daily",
    "kaggle_web_traffic",
    "traffic_weekly", "covid_deaths", "sunspot", "saugeenday", "us_births"
]


# Para cada dataset
#long_horizon_data_dir = "experiments/datasets/datasets/long_horizon_data"
#train_df, val_df, test_df = LongHorizon.load(directory=long_horizon_data_dir, group="Weather")
# Para cada conjunto
#dataset = torch.tensor(val_df["ex_4"].values, dtype = torch.float32)
max = 8192
dataset_all = load_dataset("monash_tsf", "saugeenday")

dataset = dataset_all["train"]["target"]
dataset += dataset_all["validation"]["target"]
dataset += dataset_all["test"]["target"]

MLM = []
inputs = []
solutions = []

dataset_reshaped = []


"""
padded = [
    torch.tensor(center_pad(data, ts_size, 0), dtype=torch.float32)
    if len(data) <= ts_size else torch.tensor(data[:ts_size], dtype=torch.float32)
    for data in dataset
]

dataset_reshaped = torch.stack(padded)
"""

for data in dataset:
    data = torch.tensor(data)
    len_data = data.size(0)
    num_data = len_data//ts_size
    dataset_reshaped.append(data[: num_data*ts_size].reshape(num_data, ts_size))

dataset_reshaped = torch.cat(dataset_reshaped, dim=0)

torch.isnan(dataset_reshaped).any()
dataset_reshaped = torch.nan_to_num(dataset_reshaped, nan=0.0)

dataset_reshaped = dataset_reshaped[:max]

for x in dataset_reshaped:
    for j,freq in enumerate(wordfreq):
            # Quizas en vez de 8 de 1, 4 de 2, 2 de 4, 1 de 8 hacer
            # 4 de 1, 4 de 2, 2 de 4, 2 de 8, 2 de 16
            for _ in range(int(8/freq)):
                # Se selcciona una mascara
                idx = int(torch.randint(0, int(dim_seq/freq), (1,)).item())
                mask = MLM_catalog[j][idx]
                MLM.append(mask)
                # Se crea opcion similar
                sim_option = similar_option(vocab, mask, x)
                # Se crea opcion completamente aleatoria
                rand_option = random_option(vocab, mask, x) 
                stacked_options = torch.stack([x, sim_option, rand_option])
                # Shuffle the options
                perm = torch.randperm(3) 
                shuffled_options = stacked_options[perm]
                # The original solution index was at 0
                solutions.append(perm[0])
                inputs.append(shuffled_options)


path_dataset = "experiments/pretraining/data/"
inputs = torch.stack(inputs)
MLM = torch.stack(MLM)
solutions = torch.stack(solutions)

# ----------------------------------------------
index_shuffled = torch.randperm(inputs.size(0))

inputs = inputs[index_shuffled]
MLM = MLM[index_shuffled]
solutions = solutions[index_shuffled]
# ----------------------------------------------

torch.save(inputs, path_dataset + "saugeenday_inputs.pt")
torch.save(MLM, path_dataset + "saugeenday_masks.pt")
torch.save(solutions, path_dataset + "saugeenday_solutions.pt")

# tensor_cargado = torch.load(path_dataset + "input.pt")
# inputs, MLM, solutions a .pt