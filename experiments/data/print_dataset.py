import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import os

"""
names = ["Car", "Computers", "Earthquakes", "Fish",
"InsectWingbeatSound", "Mallat", "OliveOil", "Phoneme",
"RefrigerationDevices", "ScreenType", "ShapeletSim", "ShapesAll",
"Strawberry", "UWaveGestureLibraryAll", "Wine"]
"""

names = ["InsectWingbeatSound"]

for name in names:


    df = pd.read_csv(f"experiments/data/raw/{name}/train.csv")

    X = df.drop(columns=["label"]).values 
    y = df["label"].values

    classes = np.unique(y)
    colors = plt.get_cmap('tab10', 10)

    offset_step = 5
    N_per_class = 1
    current_offset = 0.0

    plt.figure(figsize=(5, 3))

    for class_idx, class_value in enumerate(classes):
        class_indices = np.where(y == class_value)[0]

        color = colors(class_idx % 10)
        
        selected_indices = class_indices[:N_per_class]
        
        for idx in selected_indices:
            plt.plot(X[idx] + current_offset, color=color)
            
        current_offset += offset_step

    plt.axis("off")
    plt.tight_layout()

    plt.savefig(os.path.join('experiments/data/images', name + ".png"))