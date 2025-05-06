from hitsbe import vocabulary2
import matplotlib.pyplot as plt

v50k = vocabulary2.Vocabulary("experiments/vocabulary/uniformwords/100k.vocab")

fig, axes = plt.subplots(nrows=32, ncols=4, figsize=(16, 64))

axes = axes.flatten()

for i, w in enumerate(v50k.words):
    ax = axes[i]
    x = list(range(8))

    for word in w:
        ax.plot(x, word, alpha = 0.2)

    ax.set_title(f"Monotonia {i} -  {format(i, '07b')} - {len(w)} words")
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()

plt.savefig("experiments/vocabulary/uniformwords/100k_dist.png")