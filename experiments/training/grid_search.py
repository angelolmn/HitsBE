from aeon.datasets import load_classification
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from hitsbe.models.hitsBERT import HitsBERT, HitsBERTClassifier

import sys
from itertools import product

# Verifica GPU
if not torch.cuda.is_available():
    sys.exit("ERROR: No GPU detected.")

device = torch.device("cuda")

# Dataset
name = "WormsTwoClass"

X_train, y_train = load_classification(name, split="train")
X_test, y_test = load_classification(name, split="test")

num_classes = len(np.unique(y_train))

# Pretrained model path
path = "experiments/pretraining/savemodel"

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

batch_size_grid = [16, 32, 64]
lr_grid = [0.001, 0.005, 0.01]
weight_decay = [1e-2, 5e-3, 1e-3]

epochs = 8
# Guardar resultados
results = []

# Grid Search manual
for batch_size, lr, weight_decay in product(batch_size_grid, lr_grid, weight_decay):
    print(f"\n=== Entrenando {epochs} épocas, batch_size={batch_size}, lr={lr} ===, weight decay={weight_decay}")

    # Recarga el modelo base cada vez
    model_base = HitsBERT.from_pretrained(path)
    classifier = HitsBERTClassifier(model=model_base, num_classes=num_classes)

    # Entrenamiento
    classifier.fit(
        X_train=X_train,
        y_train=y_train_encoded,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        device=device
    )
