import torch

"""
dataset_name = [
    "weather_monash", "tourism_monthly", "bitcoin", 
    "vehicle_trips", "nn5_daily", "kaggle_web_traffic", 
    "traffic_weekly", "covid_deaths", "sunspot", "saugeenday", "us_births",
    "ETTh1", "ETTh2", "ETTm1", "ETTm2", "ECL_1", "Exchange", "ILI", "weather"
]"""

dataset_name = [
    "weather_monash_bound", "tourism_monthly", "bitcoin", 
    "vehicle_trips", "nn5_daily", "kaggle_web_traffic_bound", 
    "traffic_weekly", "covid_deaths", "sunspot", "saugeenday", "us_births",
    "ETTh1", "ETTh2", "ETTm1", "ETTm2", "ECL_1_bound", "Exchange", "ILI", "weather"
]

path_dataset = "experiments/pretraining/data/"
max_size = 270336

all_inputs = []
all_masks = []
all_solutions = []
current_size = 0
chunk_id = 1

for name in dataset_name:
    input = torch.load(path_dataset + name + "_inputs.pt")
    mask = torch.load(path_dataset + name + "_masks.pt")
    solution = torch.load(path_dataset + name + "_solutions.pt")

    data_size = input.size(0)

    # Si al añadir esta parte se supera el límite
    if current_size + data_size > max_size:
        overflow = current_size + data_size - max_size

        # Añadir la parte que cabe
        limit = data_size - overflow
        all_inputs.append(input[:limit])
        all_masks.append(mask[:limit])
        all_solutions.append(solution[:limit])

        # Guardar el fragmento
        inputs_cat = torch.cat(all_inputs, dim=0)
        masks_cat = torch.cat(all_masks, dim=0)
        solutions_cat = torch.cat(all_solutions, dim=0)

        torch.save(inputs_cat, path_dataset + f"inputs{chunk_id}.pt")
        torch.save(masks_cat, path_dataset + f"masks{chunk_id}.pt")
        torch.save(solutions_cat, path_dataset + f"solutions{chunk_id}.pt")

        # Preparar la siguiente tanda
        chunk_id += 1
        all_inputs = [input[limit:]]
        all_masks = [mask[limit:]]
        all_solutions = [solution[limit:]]
        current_size = overflow

    else:
        all_inputs.append(input)
        all_masks.append(mask)
        all_solutions.append(solution)
        current_size += data_size

# Guardar los datos restantes
if all_inputs:
    inputs_cat = torch.cat(all_inputs, dim=0)
    masks_cat = torch.cat(all_masks, dim=0)
    solutions_cat = torch.cat(all_solutions, dim=0)

    torch.save(inputs_cat, path_dataset + f"inputs{chunk_id}.pt")
    torch.save(masks_cat, path_dataset + f"masks{chunk_id}.pt")
    torch.save(solutions_cat, path_dataset + f"solutions{chunk_id}.pt")
