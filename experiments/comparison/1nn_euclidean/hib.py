import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import json
import numpy as np

names = ["Earthquakes", "ScreenType", "ShapeletSim",
         "Strawberry", "UWaveGestureLibraryAll", "Wine", "InsectWingbeatSound", "Fish",
         "RefrigerationDevices", "ShapesAll", "Computers", "Car"]

clf = KNeighborsClassifier(n_neighbors=1, metric="euclidean")

with open("experiments/comparison/5fold_cv/folds.json") as f:
    folds_dict = json.load(f)

#output = "hitsberted_mean"
output = "hitsberted_norm"

#model_type = "hitsbert_model_pre"
model_type = "hitsbert_model_nopre"

weights_type = "randinit"
#weights_type = "baseuncase"

if output == "hitsberted_mean":
     print("ejecutando con medias")
else:
     print("Ejecutando con token CLS")

if weights_type == "randinit":
    print("Modelo con pesos iniciales aleatorios")
else:
    print("Modelo con pesos iniciales de BERT base uncase")

if model_type == "hitsbert_model_pre":
    print("Modelo preentrenado")
else:
    print("Modelo NO preentrenado")

for name in names:
        df = pd.read_csv(f"experiments/data/{output}/{weights_type}/{model_type}/{name}/train.csv")

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
