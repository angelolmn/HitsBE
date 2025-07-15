import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

import json
import numpy as np

"""
names = ["Car"]
"""

names = ["Earthquakes", "ScreenType", "ShapeletSim",
         "Strawberry", "UWaveGestureLibraryAll", "Wine", "InsectWingbeatSound", "Fish",
         "RefrigerationDevices", "ShapesAll", "Computers", "Car"]

clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,               
    max_features="sqrt",          
    min_samples_leaf=1,
    n_jobs=-1,
    random_state=42
)



with open("experiments/comparison/5fold_cv/folds.json") as f:
    folds_dict = json.load(f)

for name in names:
    print("Dataset " + name)
    df = pd.read_csv(f"experiments/data/tsfeatured/{name}/train.csv")

    X = df.drop(columns=["label"]).values
    y = df["label"].values

    fold = folds_dict[name]
    nfolds = len(fold)

    fold_accuracies = []
    fold_f1 = []                       

    for idx in range(nfolds):
        split = fold[idx]

        train_idx = split["train_idx"]
        val_idx = split["val_idx"]

        X_train = X[train_idx]
        y_train = y[train_idx]

        X_val = X[val_idx]
        y_val = y[val_idx]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        f1  = f1_score(y_val, y_pred, average='macro')

        fold_accuracies.append(acc)
        fold_f1.append(f1)


    print(f"Average accuracy: {np.mean(fold_accuracies):.4f}")
    print(f"Average macro-F1: {np.mean(fold_f1):.4f}")

