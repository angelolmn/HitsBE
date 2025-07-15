import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

names = [
    "Car", "Computers", "Earthquakes", "Fish",
    "InsectWingbeatSound", 
    "RefrigerationDevices", "ScreenType", "ShapeletSim",
    "ShapesAll", "Strawberry", "UWaveGestureLibraryAll", "Wine"
]

n_cols        = 3                               
n_rows        = math.ceil(len(names) / n_cols)      
figsize_cell  = (2.2, 2.2)                           
figsize_total = (figsize_cell[0] * n_cols,
                 figsize_cell[1] * n_rows)

offset_step   = 6.5     
n_per_class   = 1            
cmap_base     = plt.colormaps['tab10']               


fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=figsize_total,
                         squeeze=False)

for ax, name in zip(axes.flat, names):
    csv_path = f"experiments/data/raw/{name}/train.csv"
    df       = pd.read_csv(csv_path)
    X        = df.drop(columns=["label"]).values
    y        = df["label"].values

    classes        = np.unique(y)
    current_offset = 0.0

    for class_idx, c in enumerate(classes):
        color = cmap_base(class_idx % 10)

        idxs  = np.where(y == c)[0][:n_per_class]
        for s in idxs:
            ax.plot(X[s] + current_offset, color=color,
                    linewidth=1.0, antialiased=True)
        current_offset += offset_step

    ax.set_title(name, fontsize=8, pad=2)
    ax.axis("off")

for ax in axes.flat[len(names):]:
    ax.axis("off")

fig.tight_layout()

fig.savefig("experiments/data/images/total.png", dpi=300)  
plt.close(fig)                 
