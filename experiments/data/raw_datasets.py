import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os 

from aeon.datasets import load_classification



names = ["Earthquakes", "ScreenType", "ShapeletSim",
         "Strawberry", "UWaveGestureLibraryAll", "Wine", "InsectWingbeatSound", "Fish",
         "RefrigerationDevices", "ShapesAll", "OliveOil", "Computers", "Mallat",
         "Car", "Phoneme"]

def convert_X_to_dataframe(X):
    series_length = X.shape[2]

    data = X[:, 0, :]  
    df = pd.DataFrame(data, columns=[f"time_{i}" for i in range(series_length)])

    return df

def convert_output_to_dataframe(X):
    n_batches, output_length, word_length = X.size

    data = X.reshape(n_batches, -1)

    df = pd.DataFrame(data, columns=[f"w{i}d{j}" for i in range(output_length) for j in range(word_length)])

    return df


for name in names:
    print(f"Procesando dataset: {name}")
    
    X_train, y_train = load_classification(name, split="train")
    X_test, y_test = load_classification(name, split="test")
    
    df_train = convert_X_to_dataframe(X_train)
    df_train["label"] = y_train 
    
    df_test = convert_X_to_dataframe(X_test)
    df_test["label"] = y_test
    
    os.makedirs(f"experiments/data/raw/{name}", exist_ok=True)

    df_train.to_csv(f"experiments/data/raw/{name}/train.csv", index=False, float_format="%.16g")
    df_test.to_csv(f"experiments/data/raw/{name}/test.csv", index=False, float_format="%.16g")

