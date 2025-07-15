import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import os
from umap import UMAP

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

reducer = UMAP(random_state=42)

#norm = "hitsberted"
norm = "hitsberted_norm"
#norm = "hitsberted_mean"

#weights_type = "randinit"
weights_type = "baseuncase"

#model_type = "hitsbert_model_pre"
model_type = "hitsbert_model_nopre"

if norm == "hitsberted_norm":
    norm_verbose = "Data normalized"
else:
    norm_verbose = "Data not normalized"

if weights_type == "randinit":
    init_verbose = "Weights randomly initialized"
else:
    init_verbose = "Weights initialized from BERT-base-uncased"

if model_type == "hitsbert_model_pre":
    training_verbose = "Model pretrained"
else:
    training_verbose = "Model not pretrained"

for name in names:
    df = pd.read_csv(f"experiments/data/{norm}/{weights_type}/{model_type}/{name}/train.csv")

    X = df.drop(columns=["label"]).values
    y = df["label"].values

    X_2d = reducer.fit_transform(X)
    
    plt.figure(figsize=(6, 6))
    cmap = plt.cm.get_cmap("tab10")            
    classes = np.unique(y)

    for idx, cls in enumerate(classes):
        mask = y == cls
        plt.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            s=40,
            color=cmap(idx % 10),
            label=str(cls),
            alpha=0.75,
            edgecolors="none",
        )
 
    plt.title(f"{name}", fontsize=16)

    plt.tick_params(axis='both', labelsize=16)
    plt.legend(markerscale=2, bbox_to_anchor=(1.04, 1), loc="upper left", borderaxespad=0)
    #plt.tight_layout()

    save_root = f"../memoriatfg/img/experimentos/umap/{weights_type}/{model_type}/"
    os.makedirs(save_root, exist_ok=True)
    plt.savefig(save_root + name + ".png", dpi=300)
