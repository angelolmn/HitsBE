from aeon.datasets import load_classification
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from itertools import product

from hitsbe.models.hitsBERT import HitsBERT, HitsBERTClassifier

import sys

if not torch.cuda.is_available():
    sys.exit("ERROR: No GPU detected.")

device = torch.device("cuda")

name = "WormsTwoClass"

X_train, y_train = load_classification(name, split="train")
X_test, y_test = load_classification(name, split="test")

num_classes = len(np.unique(y_train))

path = "experiments/pretraining/savemodel"

model_base = HitsBERT.from_pretrained(path)
#model_base = HitsBERT.from_pretrained(path, filename = "hitsbert_model_base.bin")

classifier = HitsBERTClassifier(model=model_base, num_classes=num_classes)

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Grid search
classifier.fit(X_train=X_train, y_train=y_train_encoded, epochs=50, batch_size=64, lr=0.01, weight_decay = 0.001, device=device)

y_pred = classifier.predict(X_test, batch_size=32, device=device)

acc = accuracy_score(y_test_encoded, y_pred)
print(f"Precisión en test: {acc:.4f}")

