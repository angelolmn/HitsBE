import pandas as pd
import torch
from hitsbe.models.hitsBERT import HitsBERT

import os

def convert_output_to_dataframe(X):
    n_batches, hidden_size = X.shape

    df = pd.DataFrame(X.detach().cpu().numpy(), columns=[f"d{i}" for i in range(hidden_size) ])

    return df


names = ["Car", "Computers", "Earthquakes", "Fish",
"InsectWingbeatSound", "RefrigerationDevices", 
"ScreenType", "ShapeletSim", "ShapesAll",
"Strawberry", "UWaveGestureLibraryAll", "Wine"]

names = [ "UWaveGestureLibraryAll"]

path = "experiments/pretraining/savemodel/base_uncase"

#filename = "hitsbert_model_nopre"
filename = "hitsbert_model_pre"

model = HitsBERT.from_pretrained(path, filename = filename + ".bin")
model.bert.config.output_hidden_states = True
model.bert.config.return_dict = True

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

    """features_train = outputs_train.last_hidden_state[:, 0, :]
    features_test = outputs_test.last_hidden_state[:, 0, :]"""

    cls_layers_train = [h[:, 0, :] for h in outputs_train.hidden_states]  # lista de B × H

    N = cls_layers_train.shape[0]

    """df_train_output = convert_output_to_dataframe(features_train)
    df_train_output["label"] = y_train

    df_test_output = convert_output_to_dataframe(features_test)
    df_test_output["label"] = y_test"""

    for idx in range(N):
        df_train_output = convert_output_to_dataframe(cls_layers_train[idx])
        df_train_output["label"] = y_train

        df_train_output.to_csv(f"experiments/data/model_forward/cls_{idx}.csv", index=False, float_format="%.16g")
        #df_test_output.to_csv(f"experiments/data/hitsberted/baseuncase/{filename}/{name}/test.csv", index=False, float_format="%.16g")

    #os.makedirs(f"experiments/data/hitsberted/baseuncase/{filename}/{name}", exist_ok=True)


