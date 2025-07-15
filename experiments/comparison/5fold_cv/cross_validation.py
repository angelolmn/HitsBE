import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from collections import Counter
import json

names = ["Earthquakes", "ScreenType", "ShapeletSim",
         "Strawberry", "UWaveGestureLibraryAll", "Wine", "InsectWingbeatSound", "Fish",
         "RefrigerationDevices", "ShapesAll", "OliveOil", "Computers", "Mallat",
         "Car", "Phoneme"]


nfolds = 5

folds_dict = {}

for name in names:
    dataset = pd.read_csv(f"experiments/data/raw/{name}/train.csv")
    y = dataset["label"].values
    
    skf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=42)
    splits = []

    class_counts = Counter(y)
    min_class = min(class_counts.values())

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        splits.append({
                "train_idx": train_idx.tolist(),
                "val_idx": val_idx.tolist()
            })

    folds_dict[name] = splits

with open(f"experiments/comparison/5fold_cv/folds.json", "w") as f:
    json.dump(folds_dict, f)
