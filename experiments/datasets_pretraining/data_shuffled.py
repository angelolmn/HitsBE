import torch

path_dataset = "experiments/pretraining/data/"

input1 = torch.load(path_dataset + "inputs1.pt")
mask1 = torch.load(path_dataset + "masks1.pt")
solutions1 = torch.load(path_dataset + "solutions1.pt")

input2 = torch.load(path_dataset + "inputs2.pt")
mask2 = torch.load(path_dataset + "masks2.pt")
solutions2 = torch.load(path_dataset + "solutions2.pt")

n = input1.size(0)

perm1 = torch.randperm(n)
input1 = input1[perm1]
mask1 = mask1[perm1]
solutions1 = solutions1[perm1]

perm2 = torch.randperm(n)
input2 = input2[perm2]
mask2 = mask2[perm2]
solutions2 = solutions2[perm2]

torch.save(input1, path_dataset + "inputs1.pt")
torch.save(mask1, path_dataset + "masks1.pt")
torch.save(solutions1, path_dataset + "solutions1.pt")

torch.save(input2, path_dataset + "inputs2.pt")
torch.save(mask2, path_dataset + "masks2.pt")
torch.save(solutions2, path_dataset + "solutions2.pt")

