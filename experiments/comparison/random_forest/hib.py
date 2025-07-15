import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

names = [
    "Car",
    "Computers",
    "Earthquakes",
    "Fish",
    "InsectWingbeatSound",
    "RefrigerationDevices",
    "ScreenType",
    "ShapeletSim",
    "ShapesAll",
    "Strawberry",
    "UWaveGestureLibraryAll",
    "Wine"
]


# Random Forest classifier
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

#output = "hitsberted_mean"
output = "hitsberted_norm"

#model_type = "hitsbert_model_pre"
model_type = "hitsbert_model_nopre"

#weights_type = "randinit"
weights_type = "baseuncase"

if output == "hitsberted_mean":
     print("Ejecutando con medias")
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


save_root = f"../memoriatfg/img/experimentos/random_forest/{weights_type}/{model_type}"
os.makedirs(save_root, exist_ok=True)

for name in names:
    df = pd.read_csv(f"experiments/data/{output}/{weights_type}/{model_type}/{name}/train.csv")
    X = df.drop(columns=["label"]).values
    y = df["label"].values

    fold = folds_dict[name]
    nfolds = len(fold)

    fold_accuracies = []
    fold_f1 = []

    y_true_all, y_pred_all = [], []

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

        y_true_all.extend(y_val)
        y_pred_all.extend(y_pred)

        acc = accuracy_score(y_val, y_pred)
        f1  = f1_score(y_val, y_pred, average='macro')

        fold_accuracies.append(acc)
        fold_f1.append(f1)

        #print(f"Split {idx + 1}: Accuracy = {acc:.4f}")

    print(f"Average accuracy {name}: {np.mean(fold_accuracies):.4f}")
    print(f"Average macro-F1: {np.mean(fold_f1):.4f}")

    # --------- crear y guardar la matriz de confusión ------------
    """
    cm = confusion_matrix(y_true_all, y_pred_all, labels=np.unique(y))
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=np.unique(y), yticklabels=np.unique(y),
                cbar=False,
                annot_kws={'size': 18})
    
    plt.title(f'{name}', fontsize=16)
    #plt.tight_layout()
    plt.tick_params(axis='both', labelsize=16)

    out_path = os.path.join(save_root, f"{name}.png")
    plt.savefig(out_path, dpi=400)   # alta resolución para zoom
    plt.close()
    print(f"Matiz guardada en: {out_path}")

    """