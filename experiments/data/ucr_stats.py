from aeon.datasets import load_classification
import pandas as pd
import numpy as np

# HandOutlines             2709
# InlineSkate              1882
# Haptics                  1092


ucr_datasets = [
    "Adiac", "ArrowHead", "Beef", "BeetleFly", "BirdChicken", "Car", "CBF",
    "ChlorineConcentration", "Coffee", "Computers", "CricketX",
    "CricketY", "CricketZ", "DiatomSizeReduction", "DistalPhalanxTW", "Earthquakes", "ECG200", "ECG5000",
    "ECGFiveDays", "ElectricDevices", "FaceAll", "FaceFour", "FacesUCR",
    "Fish", "FordA", "FordB", "GunPoint", "Ham", "HandOutlines",
    "Haptics", "Herring", "InlineSkate", "InsectWingbeatSound",
    "ItalyPowerDemand", "LargeKitchenAppliances", "Lightning2", "Lightning7",
    "Mallat", "Meat", "MedicalImages", "MiddlePhalanxTW", "MoteStrain",
    "OliveOil", "OSULeaf", "PhalangesOutlinesCorrect", "Phoneme", "Plane",
    "ProximalPhalanxTW", "RefrigerationDevices", "ScreenType", "ShapeletSim",
    "ShapesAll", "SmallKitchenAppliances", "SonyAIBORobotSurface1",
    "SonyAIBORobotSurface2", "Strawberry", "SwedishLeaf", "Symbols",
    "SyntheticControl", "ToeSegmentation1", "ToeSegmentation2", "Trace",
    "TwoLeadECG", "TwoPatterns", "UWaveGestureLibraryX", "UWaveGestureLibraryY",
    "UWaveGestureLibraryZ", "UWaveGestureLibraryAll", "Wafer", "Wine",
    "WordSynonyms", "Worms", "WormsTwoClass", "Yoga"
]

results = []
failed = []

for name in ucr_datasets:
    try:
        X_train, y_train = load_classification(name, split="train")
        X_test, y_test = load_classification(name, split="test")

        _,_,series_length = X_train.shape

        num_classes = len(np.unique(y_train))
        results.append((name, len(X_train), len(X_test), series_length, num_classes))

    except Exception as e:
        print(f"❌ Error cargando {name}: {e}")
        failed.append(name)

# Imprimir resultados
print("\n📊 Estadísticas de datasets univariantes:")
print(f"{'Dataset':<35} {'#Train':>8} {'#Test':>8} {'Length':>8} {'Classes':>8}")
print("-" * 65)

for name, n_train, n_test, length, nclass in results:
    print(f"{name:<35} {n_train:>8} {n_test:>8} {length:>8}{nclass:>8}")

"""                                 Train       Test        length      Classes
Earthquakes                         322         139         512         2
ScreenType                          375         375         720         3
ShapeletSim                         20          180         500         2
Strawberry                          613         370         235         2
UWaveGestureLibraryAll              896         3582        945         8
Wine                                57          54          234         2
InsectWingbeatSound                 220         1980        256         11
Fish                                175         175         463         7
RefrigerationDevices                375         375         720         3
ShapesAll                           600         600         512         60
Computers                           250         250         720         2
Car	                                60	        60	        577	        4
"""