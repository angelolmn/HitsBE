import torch

dataset_name = [
    "weather_monash", "tourism_monthly", "bitcoin", 
    "vehicle_trips", "nn5_daily", "kaggle_web_traffic", 
    "traffic_weekly", "covid_deaths", "sunspot", "saugeenday", "us_births",
    "ETTh1", "ETTh2", "ETTm1", "ETTm2", "ECL_1", "Exchange", "ILI", "weather"
]

threshold = 100000

path_dataset = "experiments/pretraining/data/"
sum = 0
for name in dataset_name:
    input = torch.load(path_dataset + name + "_inputs.pt")
    mask = torch.load(path_dataset + name + "_masks.pt")
    solutions = torch.load(path_dataset + name + "_solutions.pt")

    if input.size(0) > threshold:
        input = input[:threshold]
        mask = mask[:threshold]
        solutions = solutions[:threshold]

        torch.save(input, path_dataset + name +"_bound_inputs.pt")
        torch.save(mask, path_dataset + name + "_bound_masks.pt")
        torch.save(solutions, path_dataset + name + "_bound_solutions.pt")

        print(name + "_bound")


