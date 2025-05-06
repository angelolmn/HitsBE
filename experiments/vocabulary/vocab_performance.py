import numpy as np
import torch

import time

import matplotlib.pyplot as plt
import os

from aeon.datasets import load_classification

from hitsbe import Hitsbe as HB1
from hitsbe import hitsbe2 as HB2



model_embedding1 = HB1("experiments/vocabulary/spline50kv2.vocab")

model_embedding2 = HB2.Hitsbe("experiments/vocabulary/spline50kv2.vocab")

X_train, _ = load_classification("Mallat", split="train")

X_train_shaped = np.squeeze(X_train, axis=1)

X_train_tensor = torch.tensor(X_train_shaped, dtype=torch.float)

start = time.time()

for X in X_train_tensor:
    model_embedding1.get_sequence(X)

end = time.time()

print(f"Time hitsbe1: {end - start:.6f} seconds with 55 iterations")

start = time.time()

for X in X_train_tensor:
    model_embedding2.get_sequence(X)

end = time.time()

print(f"Time hitsbe2: {end - start:.6f} seconds with 55 iterations")
