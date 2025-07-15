import torch

dataset_name = [
    "weather_monash_bound", "tourism_monthly", "bitcoin", 
    "vehicle_trips", "nn5_daily", "weather_monash_bound", 
    "traffic_weekly", "covid_deaths", "sunspot", "saugeenday", "us_births",
    "ETTh1", "ETTh2", "ETTm1", "ETTm2", "ECL_1_bound", "Exchange", "ILI", "weather"
]

path_dataset = "experiments/pretraining/data/"
sum = 0

for name in dataset_name:
    input = torch.load(path_dataset + name + "_inputs.pt")
    mask = torch.load(path_dataset + name + "_masks.pt")
    solutions = torch.load(path_dataset + name + "_solutions.pt")
    print(name + " contiene " + str(input.size(0)) + " series temporales.")
    sum += input.size(0)

print("Tamaño del dataset: " + str(sum))
