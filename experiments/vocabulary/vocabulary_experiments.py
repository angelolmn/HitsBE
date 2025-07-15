import numpy as np
import torch
import math

import matplotlib.pyplot as plt
import os

from aeon.datasets import load_classification

from hitsbe import Hitsbe

model_embedding = Hitsbe()

X_train, _ = load_classification("Mallat", split="train")

X_train_shaped = np.squeeze(X_train, axis=1)

X_train_tensor = torch.tensor(X_train_shaped, dtype=torch.float)

X = X_train_tensor[0]

seq = model_embedding.get_sequence(X)

X_adj = model_embedding._adjust(X)

domain = np.arange(model_embedding.size)


# Figure 1 

plt.figure(figsize=(20, 10))

plt.plot(domain,X_adj, linewidth=0.5)

for i, s in enumerate(seq):
    word_index = s[0]
    word = model_embedding.vocabulary.words[word_index]

    start = i*8
    end = start + 8

    segment = X_adj[start:end]

    smax = torch.max(segment)
    smin = torch.min(segment)

    wmin = np.min(word)
    wmax = np.max(word)

    norm_word = (word - wmin)/(wmax -wmin)

    traslated_word = X_adj[start] + (norm_word - norm_word[0])  *(smax - smin).item()

    plt.plot(domain[start: end], traslated_word , color='red', linewidth=0.7)


for vgrid in range(0, int(domain[-1]) + 2, 8):
    plt.axvline(x=vgrid, color='gray', linestyle='--', linewidth=0.5)
    
plt.title("Vocabulary representation")


plt.savefig(os.path.join('experiments/vocabulary', "vocabulary_representation.png"), dpi=300)
plt.close()

# Figure 2
cols = 5
rows = math.ceil(model_embedding.dim_seq/ cols)

sub_domain = np.linspace(0,1, num=8)

fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(5 * cols, 5 * rows))
axs = axs.flatten() 

for i, s in enumerate(seq):
    word_index = s[0]
    word = model_embedding.vocabulary.words[word_index]

    start = i*8
    end = start + 8    

    segment = X_adj[start:end]

    smax = torch.max(segment)
    smin = torch.min(segment)

    segment_norm = (segment - smin) / (smax - smin)

    ax = axs[i]
    ax.plot(sub_domain, word, color='red', label=f'Word {word_index+1}')
    ax.plot(sub_domain, segment_norm, color='blue', label="Segment", alpha=0.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f'Segment {i+1}')
    ax.grid(True)
    ax.set_aspect('equal', 'box')  

# Delete empty subgraphs
for j in range(i+1, len(axs)):
    axs[j].axis('off')

plt.tight_layout()
plt.savefig(os.path.join('experiments/vocabulary', "vocabulary_bysegment.png"))
