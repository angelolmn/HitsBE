import numpy as np
import pandas as pd
import torch
from hitsbe.models.hitsBERT import HitsBERT

import os

def convert_output_to_dataframe(X):
    n_batches, output_length, word_length = X.shape

    data = X.reshape(n_batches, -1)

    df = pd.DataFrame(data.detach().cpu().numpy(), columns=[f"w{i}d{j}" for i in range(output_length) for j in range(word_length)])

    return df


names = ["Car", "Computers", "Earthquakes", "Fish",
"InsectWingbeatSound", "Mallat", "OliveOil", "Phoneme",
"RefrigerationDevices", "ScreenType", "ShapeletSim", "ShapesAll",
"Strawberry", "UWaveGestureLibraryAll", "Wine"]


#names = ["InsectWingbeatSound"]

path = "experiments/pretraining/savemodel"

model = HitsBERT.from_pretrained(path)
#model_base = HitsBERT.from_pretrained(path, filename = "hitsbert_model_base.bin")

for name in names:
    print("Procesando " + name)

    df_train = pd.read_csv(f"experiments/data/raw/{name}/train.csv")
    df_test = pd.read_csv(f"experiments/data/raw/{name}/test.csv")
    
    X_train = df_train.drop(columns=["label"]).values 
    y_train = df_train["label"].values
    
    X_test = df_test.drop(columns=["label"]).values
    y_test = df_test["label"].values

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    
    with torch.no_grad():
        outputs_train = model(X_train_tensor)
        outputs_test = model(X_test_tensor)

    features_train = outputs_train.last_hidden_state
    features_test = outputs_test.last_hidden_state

    df_train_output = convert_output_to_dataframe(features_train)
    df_train_output["label"] = y_train
    
    df_test_output = convert_output_to_dataframe(features_test)
    df_test_output["label"] = y_test

    os.makedirs(f"experiments/data/hitsberted/{name}", exist_ok=True)
    
    df_train_output.to_csv(f"experiments/data/hitsberted/{name}/train.csv", index=False, float_format="%.16g")
    df_test_output.to_csv(f"experiments/data/hitsberted/{name}/test.csv", index=False, float_format="%.16g")
