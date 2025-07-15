import numpy as np
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
import math

import os

size = 100
words = []
gen_points_set = []

domain_len = 8

np.random.seed(42)

for _ in range(size):
    xs = np.linspace(0, 1, 5)
    ys = np.random.rand(5)*0.6 + 0.2
    
    generator_points = np.stack((xs,ys), axis = 1)

    spline = make_interp_spline(xs, ys, k=3)

    domain = np.linspace(0,1,domain_len)
    words.append(np.stack((domain, spline(domain)), axis = 1))
    gen_points_set.append(generator_points)

# Figure 1: All the words in the same XY
plt.figure(figsize=(10, 10))
for curve in words:
    plt.plot(curve[:, 0], curve[:, 1])
    
plt.title("Spline (cubic interpolation) - All in one")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(True)
plt.savefig(os.path.join('experiments/vocabulary', "spline_5points.png"))
plt.close()


# Figure 2: Each spline in a subgraph
cols = 5
rows = math.ceil(size/ cols)

fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(5 * cols, 5 * rows))
axs = axs.flatten() 

for i, (control_points, curve) in enumerate(zip(gen_points_set, words)):
    ax = axs[i]
    ax.plot(curve[:, 0], curve[:, 1], label=f'Curve {i+1}')
    ax.plot(control_points[:, 0], control_points[:, 1], 'ro--', label="Control points", alpha=0.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f'Word {i+1}')
    ax.grid(True)
    ax.set_aspect('equal', 'box')  

# Delete empty subgraphs
for j in range(i+1, len(axs)):
    axs[j].axis('off')

plt.tight_layout()
plt.savefig(os.path.join('experiments/vocabulary', "spline_grid_5points.png"))
