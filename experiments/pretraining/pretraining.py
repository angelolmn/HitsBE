import os
import torch
import pandas as pd
from datasetsforecast.long_horizon import LongHorizon
from datasets import load_dataset

# Nuevas colecciones de Nixtla
from datasetsforecast.m4 import M4
from datasetsforecast.m3 import M3
from datasetsforecast.m5 import M5

# Configuración
window_size = 1024
long_horizon_data_dir = "experiments/pretraining/datasets/long_horizon_data"
nixtla_data_dir = "experiments/pretraining/datasets/nixtla"
os.makedirs(long_horizon_data_dir, exist_ok=True)
os.makedirs(nixtla_data_dir, exist_ok=True)

def contar_subseries(series_list, window_size):
    total = 0
    for s in series_list:
        length = s.numel()
        if length <= window_size:
            total += 1
        else:
            total += length // window_size
    return total

def analizar_series(series_list, dataset_name):
    lengths = [s.numel() for s in series_list]
    total_subseries = contar_subseries(series_list, window_size)

    if lengths:
        avg_len = sum(lengths) / len(lengths)
        print(f"✅ {dataset_name}:")
        print(f"   📌 Series totales (originales): {len(series_list)}")
        print(f"   📊 Longitud promedio: {avg_len:.1f}")
        print(f"   📦 Subseries generadas (≤ {window_size} o particionadas): {total_subseries}")
    else:
        print(f"⚠️ No se encontraron series válidas en {dataset_name}")

# --------------------------
# Datasets Long Horizon
# --------------------------
long_horizon_datasets = [
    "ETTh1", "ETTh2", "ETTm1", "ETTm2",
    "ECL", "Exchange", "ILI", "Weather"
]

print("\n📦 Analizando datasets Long Horizon:")
for dataset_name in long_horizon_datasets:
    print(f"\n🔍 {dataset_name}")
    try:
        train_df, val_df, test_df = LongHorizon.load(directory=long_horizon_data_dir, group=dataset_name)
        full_df = pd.concat([train_df, val_df, test_df], axis=0)
        series_list = [
            torch.tensor(group["y"].values, dtype=torch.float32)
            for _, group in full_df.groupby("unique_id")
        ]
        analizar_series(series_list, dataset_name)
    except Exception as e:
        print(f"❌ Error en {dataset_name}: {e}")

# --------------------------
# Datasets Monash TSF (HuggingFace)
# --------------------------
monash_configs = [
    "weather", "tourism_yearly", "tourism_quarterly", "tourism_monthly",
    "bitcoin", "vehicle_trips", "nn5_daily", "nn5_weekly",
    "kaggle_web_traffic", "kaggle_web_traffic_weekly", "solar_weekly",
    "traffic_weekly", "covid_deaths", "sunspot", "saugeenday", "us_births"
]

print("\n📦 Analizando datasets Monash TSF:")
for config in monash_configs:
    print(f"\n🔍 {config}")
    try:
        dataset = load_dataset("monash_tsf", config)
        train_split = dataset["train"]

        series_list = []
        for entry in train_split:
            y = entry.get("target", [])
            if y is not None and len(y) > 0:
                series_list.append(torch.tensor(y, dtype=torch.float32))

        analizar_series(series_list, config)
    except Exception as e:
        print(f"❌ Error en {config}: {e}")

# --------------------------
# Datasets univariantes de Nixtla (M1, M3, M4)
# --------------------------
print("\n📦 Analizando datasets univariantes de Nixtla:")

nixtla_datasets = {
    "M5": M5,
    "M3": M3,
    "M4": M4,
}

for name, cls in nixtla_datasets.items():
    print(f"\n🔍 {name}")
    try:
        train_df, test_df = cls.load(directory=nixtla_data_dir, group="Monthly")  # puedes cambiar a Quarterly/Yearly si quieres más
        full_df = pd.concat([train_df, test_df], axis=0)

        series_list = [
            torch.tensor(group["y"].values, dtype=torch.float32)
            for _, group in full_df.groupby("unique_id")
            if len(group["y"].values) > 0
        ]

        analizar_series(series_list, f"nixtla_{name}")
    except Exception as e:
        print(f"❌ Error en {name}: {e}")
