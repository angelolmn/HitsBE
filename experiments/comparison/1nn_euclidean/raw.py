import pandas as pd
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from sklearn.metrics import accuracy_score
import json
import numpy as np

names = ["Earthquakes", "ScreenType", "ShapeletSim",
         "Strawberry", "UWaveGestureLibraryAll", "Wine", "InsectWingbeatSound", "Fish",
         "RefrigerationDevices", "ShapesAll", "Computers", "Car"]


clf = KNeighborsTimeSeriesClassifier(
        n_neighbors=1,
        metric="dtw",
        metric_params=dict(sakoe_chiba_radius=0.1) 
)

with open("experiments/comparison/5fold_cv/folds.json") as f:
    folds_dict = json.load(f)

for name in names:
        df = pd.read_csv(f"experiments/data/raw/{name}/train.csv")

        X = df.drop(columns=["label"]).values 
        y = df["label"].values

        fold = folds_dict[name]
        nfolds = len(fold)

        fold_accuracies = []

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
                fold_accuracies.append(acc)

                #print(f"Split {idx + 1}: Accuracy = {acc:.4f}")

        print(f"Average accuracy {name}: {np.mean(fold_accuracies):.4f}")