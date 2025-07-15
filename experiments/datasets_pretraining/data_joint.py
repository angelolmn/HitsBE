import torch

path_dataset = "experiments/pretraining/data/"

input1 = torch.load(path_dataset + "inputs1.pt")
mask1 = torch.load(path_dataset + "masks1.pt")
solutions1 = torch.load(path_dataset + "solutions1.pt")

input2 = torch.load(path_dataset + "inputs2.pt")
mask2 = torch.load(path_dataset + "masks2.pt")
solutions2 = torch.load(path_dataset + "solutions2.pt")

input = torch.cat((input1,input2), dim=0)
torch.save(input, path_dataset + "inputs.pt")

mask = torch.cat((mask1,mask2), dim=0)
torch.save(mask, path_dataset + "masks.pt")

solutions = torch.cat((solutions1,solutions2), dim=0)
torch.save(solutions, path_dataset + "solutions.pt")