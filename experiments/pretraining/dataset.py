from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch
import numpy as np


class MultipleChoiceTimeSeriesDataset(Dataset):
    def __init__(self, data_path="inputs.pt", mask_path="masks.pt", solutions_path="solutions.pt"):
        self.inputs = torch.load(data_path)      # (N, noptions, ts_len)
        self.masks = torch.load(mask_path)       # (N, 128)
        self.solutions = torch.load(solutions_path)    # (N,)

    def __len__(self):
        return len(self.inputs)

    """
    __getitem__ es llamada en el propio 'for batch in dataLoader:'
    batch_size veces
    """
    def __getitem__(self, idx):
        return {
            "input": self.inputs[idx],       # (1, noptions, ts_len)
            "mask": self.masks[idx],         # (1, dim_seq)
            "solution": self.solutions[idx]  # (1,) int in [0, 1, 2]
        }
    

"""
El objetivo es devolver el dataLoader deseado
"""
def get_dataloader(split="", batch_size=16, shuffle=True, num_workers=2):
    base_path = f"experiments/pretraining/data/{split}"
    dataset = MultipleChoiceTimeSeriesDataset(
        f"{base_path}inputs.pt",
        f"{base_path}masks.pt",
        f"{base_path}solutions.pt"
    )

    # reparte el dataset entre GPUs
    # sampler = DistributedSampler(dataset)  


    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        #sampler = sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
