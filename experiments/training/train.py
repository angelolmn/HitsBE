from aeon.datasets import load_classification
import pandas as pd
import numpy as np
import torch

from hitsbe.models.hitsBERT import HitsBERT, HitsBERTClassifier

import sys


if not torch.cuda.is_available():
    sys.exit("ERROR: No GPU detected.")

device = torch.device("cuda")

name = "Adiac"

X_train, y_train = load_classification(name, split="train")
X_test, y_test = load_classification(name, split="test")

num_classes = len(np.unique(y_train))

path = "experiments/pretraining/savemodel"
model_base = HitsBERT.from_pretrained(path)

classifier = HitsBERTClassifier(model=model_base, num_classes=num_classes)

# Grid search
classifier.fit(X_train=X_train, y_train=y_train, epochs=2, batch_size=32, lr=0.01, device=device)