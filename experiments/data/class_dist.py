import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

names = ["Earthquakes", "ScreenType", "ShapeletSim",
         "Strawberry", "UWaveGestureLibraryAll", "Wine", "InsectWingbeatSound", "Fish",
         "RefrigerationDevices", "ShapesAll", "Computers", "Car"]

model_type = "hitsbert_model_pre"
#model_type = "hitsbert_model_nopre"

weights_type = "randinit"
#weights_type = "baseuncase"

for name in names:
    df = pd.read_csv(f"experiments/data/hitsberted_norm/{weights_type}/{model_type}/{name}/train.csv")

    X = df.drop(columns=["label"]).values
    y = df["label"].values

    n_samples = len(df)
    class_counts = df["label"].value_counts().sort_index()

    print(f"\nDataset: {name}")
    print(f"   Total de muestras: {n_samples}")
    print("   Distribución de clases:")
    for cls, cnt in class_counts.items():
        pct = cnt / n_samples * 100
        print(f"      Clase {cls}: {cnt} ({pct:.2f}%)")