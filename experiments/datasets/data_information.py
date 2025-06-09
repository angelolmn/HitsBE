import torch

# 857775 elementos
# 857775 // 4096 = 209
# 857775 // 2048 = 418
# 857775 // 1024 = 837
# quitarle a ECL_1 1711 series para multiplos perfectos
"""
dataset_name = [
    "weather_monash", "tourism_monthly", "bitcoin", 
    "vehicle_trips", "nn5_daily", "kaggle_web_traffic", 
    "traffic_weekly", "covid_deaths", "sunspot", "saugeenday", "us_births",
    "ETTh1", "ETTh2", "ETTm1", "ETTm2", "ECL_1", "Exchange", "ILI", "weather"
]"""

# 541785 elementos 
# 541785//4096 = 132
# 541785//2048 = 264
# 541785//1024 = 529
# Habria que quitarle 1113
# 371 a weather_monash_bound, weather_monash_bound y ECL_1_bound
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





"""
N elementos 1050 
N pasos 7

train_batch_size 150
minibatch = 25
gradient_accumulation_steps = 6

tiempo total: 544 seg = 9.06 min
              532.79 ZeRO 2

train_batch_size 150
minibatch = 50
gradient_accumulation_steps = 3

tiempo total: 595.6781039237976 seg = 9.93 min


train_batch_size 150
minibatch = 150
gradient_accumulation_steps = 1

tiempo total: ___ seg = min


Preentrenamiento:

Total de elementos: 540672
 540672//4096 = 132 global steps
 540672//2048 = 264 global steps
 540672//1024 = 528 global steps

 
train_batch_size = 1024
minibatch = 32
gradient_accumulation_steps = 32

train_batch_size = 2048
minibatch = 32
gradient_accumulation_steps = 64

train_batch_size = 4096
minibatch = 32
gradient_accumulation_steps = 128
  
"""




"""
tensor([[19.0446, 19.0841, 19.0725],
        [19.3230, 19.3150, 19.3150],
        [19.8123, 19.8415, 19.8487],
        [16.3041, 16.3540, 16.3540],
        [19.1402, 18.9501, 19.0795],
        [18.4806, 18.5122, 18.4806],
        [19.4982, 19.5064, 19.4829],
        [19.8711, 19.8987, 19.8774],
        [16.5244, 16.5207, 16.5244],
        [20.8964, 20.8957, 20.8891],
        [21.9462, 22.1221, 22.1310],
        [15.7817, 15.7817, 15.7664],
        [19.6852, 19.0107, 19.6867],
        [20.4765, 20.4742, 20.4856],
        [17.2016, 15.7920, 17.1862],
        [18.2449, 17.8815, 18.2120],
        [14.3014, 14.2341, 14.2341],
        [19.6155, 19.6364, 19.6457],
        [20.1378, 20.1047, 20.1177],
        [16.9175, 16.9447, 16.9175],
        [19.6739, 19.6571, 19.6571],
        [20.0486, 20.0602, 20.0534],
        [19.3690, 19.3372, 19.3690],
        [16.5447, 16.5447, 16.5447],
        [22.7416, 23.0675, 22.9361],
        [19.2807, 19.2807, 19.2815],
        [19.5157, 19.5144, 19.4933],
        [18.8132, 18.5778, 18.7674],
        [19.0020, 19.1093, 19.1223],
        [18.5559, 18.5132, 18.5559],
        [19.7183, 19.9205, 19.8193],
        [19.4735, 19.4831, 19.4483]], device='cuda:0', grad_fn=<DivBackward0>)
tensor(1.0667, device='cuda:0', grad_fn=<NllLossBackward0>)
tensor([[19.9447, 19.9629, 20.2656],
        [19.5079, 19.2856, 19.4620],
        [17.7576, 17.7576, 17.7576],
        [17.5647, 17.5516, 17.5516],
        [14.8459, 14.8560, 15.7160],
        [18.8576, 18.9176, 18.8806],
        [20.8302, 20.8308, 20.8042],
        [19.8752, 19.8787, 19.8758],
        [15.9888, 15.9650, 15.9888],
        [18.0386, 18.0340, 18.0482],
        [19.3482, 19.2497, 19.3677],
        [19.4909, 19.5229, 19.5232],
        [18.1145, 18.0806, 18.0806],
        [20.1477, 20.1580, 20.1204],
        [20.3317, 20.3200, 20.3349],
        [17.7887, 17.7887, 17.7971],
        [17.9946, 17.9946, 17.9862],
        [16.3535, 16.3535, 16.3474],
        [17.8055, 17.7254, 17.8055],
        [19.3510, 19.3451, 19.4223],
        [22.3058, 22.8367, 22.8657],
        [20.2618, 20.2825, 20.2825],
        [19.7315, 19.6550, 19.7095],
        [15.3952, 15.3952, 15.3952],
        [21.3668, 21.3794, 21.3752],
        [19.4413, 19.4427, 19.4198],
        [19.6404, 19.6315, 19.6276],
        [17.4049, 17.4049, 17.4049],
        [18.3169, 18.2851, 18.2851],
        [13.7436, 14.2674, 14.2674],
        [18.5933, 18.4016, 18.4101],
        [19.3357, 19.2109, 19.2690]], device='cuda:0', grad_fn=<DivBackward0>)
tensor(1.0686, device='cuda:0', grad_fn=<NllLossBackward0>)

"""


"""
                        bsz         steps           samples         days            

--------------------------------------------------------------------------------

Google BERT Base        256         1000K           256M            5.85
                                                                                    16 TPUs 4 days            
Google BERT Large       128         2000k           256M            26.33

--------------------------------------------------------------------------------

                        128         2000k           256M            14.11

                        256         1000k           256M            8.34

BERT academy budget     4096        63k             256M            2.74            8 Nvidia Titan-V 12GB

                        8192        31k             256M            2.53

                        16384       16k             256M            2.41

---------------------------------------------------------------------------------

HitsBE                  512         1k              0.54M                           1 Nvidia RTX 2080 Ti

"""