from hitsbe import vocabulary2
import matplotlib.pyplot as plt

v50kv2 = vocabulary2.Vocabulary("experiments/vocabulary/uniformwords/50kv2.vocab")
v50k = vocabulary2.Vocabulary("experiments/vocabulary/uniformwords/50k.vocab")

index = [0,1,15,31]

fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(16, 12))

axes = axes.flatten()

for i, idx in enumerate(index):
    x = list(range(8))
    
    op_idx = int(''.join('1' if b == '0' else '0' for b in format(idx, '07b')), 2)
    
    ax = axes[i*4]

    for w in v50k.words[idx]:
        ax.plot(x, w, alpha = 0.2)
    
    ax.set_title(f"Monotonia {idx} -  {format(idx, '07b')} - {len(v50k.words[idx])} words")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axes[i*4+1]

    for w in v50k.words[op_idx]:
        ax.plot(x, w, alpha = 0.2)

    ax.set_title(f"Monotonia {op_idx} -  {format(op_idx, '07b')} - {len(v50k.words[op_idx])} words")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axes[i*4+2]

    for w in v50kv2.words[idx]:
        ax.plot(x, w, alpha = 0.2)
    
    ax.set_title(f"Monotonia {idx} -  {format(idx, '07b')} - {len(v50kv2.words[idx])} words V2")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axes[i*4+3]
    
    for w in v50kv2.words[op_idx]:
        ax.plot(x, w, alpha = 0.2)
    
    ax.set_title(f"Monotonia {op_idx} -  {format(op_idx, '07b')} - {len(v50kv2.words[op_idx])} words V2")
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()

plt.savefig("experiments/vocabulary/uniformwords/50k_vs_50kv2_dist.png")