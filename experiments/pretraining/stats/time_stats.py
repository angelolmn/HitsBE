import matplotlib.pyplot as plt

# Datos de tiempos (segundos)
tiempos_hitsbe = [
    9.322060585021973, 8.93831729888916, 9.05681562423706, 11.4262056350708, 9.546890258789062,
    10.229008197784424, 9.903082609176636, 9.441688776016235, 10.642619132995605, 9.69991946220398,
    9.896040678024292, 9.274577856063843, 9.291568279266357
]

tiempos_bert = [
    0.13331961631774902, 0.012255668640136719, 0.011563777923583984, 0.011409759521484375,
    0.009710311889648438, 0.012252569198608398, 0.010498046875, 0.010313034057617188,
    0.009665966033935547, 0.012268543243408203, 0.009186983108520508, 0.008985042572021484,
    0.009311914443969727
]

tiempos_hitsbert = [
    9.510598421096802, 9.024052858352661, 9.142597913742065, 11.512229442596436,
    9.63263201713562, 10.315147876739502, 9.988886833190918, 9.527448892593384,
    10.72831678390503, 9.785804033279419, 9.981565237045288, 9.360170125961304,
    9.377209901809692
]

tiempos_forward = [
    25.446338891983032, 24.873554229736328, 25.037280559539795, 27.36824607849121,
    25.471794843673706, 26.128627061843872, 25.859675884246826, 25.384453773498535,
    26.558815717697144, 25.64987564086914, 25.82154679298401, 25.217523336410522,
    25.241389989852905
]

tiempos_minibatch = [
    25.61374807357788, 25.032516717910767, 25.179197549819946, 27.532856225967407,
    25.62313222885132, 26.311203479766846, 26.02249264717102, 25.526010990142822,
    26.72384762763977, 25.787888765335083, 25.98241424560547, 25.379754781723022,
    25.384355783462524
]

# Calcular promedios
mean_hitsbe = sum(tiempos_hitsbe) / len(tiempos_hitsbe)
mean_bert = sum(tiempos_bert) / len(tiempos_bert)
mean_hitsbert = sum(tiempos_hitsbert) / len(tiempos_hitsbert)
mean_forward = sum(tiempos_forward) / len(tiempos_forward)
mean_minibatch = sum(tiempos_minibatch) / len(tiempos_minibatch)

# Pie chart 1: Tiempo del modelo (HitsBERT)
resto_modelo = mean_hitsbert - (mean_hitsbe + mean_bert)
labels_model = ['HitsBE', 'BERT', 'Otras operaciones']
sizes_model = [mean_hitsbe, mean_bert, resto_modelo]

# Pie chart 2: Tiempo de minibatch
tiempo_actualizacion = mean_forward - mean_hitsbert
resto_batch = mean_minibatch - (mean_hitsbert + tiempo_actualizacion)
labels_batch = ['Modelo (HitsBERT)', 'Actualización', 'Otras operaciones']
sizes_batch = [mean_hitsbert, tiempo_actualizacion, resto_batch]

# Crear subplots
fig, axs = plt.subplots(1, 2, figsize=(11, 5))
radius = 0.8

# --- Gráfico 1: Tiempo del modelo ---
wedges1, _ = axs[0].pie(sizes_model, startangle=90, radius=radius)
axs[0].axis('equal')
axs[0].set_title("Distribución del tiempo he HitsBERT", fontsize=16)

total_model = sum(sizes_model)
legend_labels_model = [
    f"{label}: {size:.2f} s ({(size / total_model) * 100:.1f}%)"
    for label, size in zip(labels_model, sizes_model)
]
axs[0].legend(
    wedges1, legend_labels_model,
    loc='upper center', bbox_to_anchor=(0.5, -0.08),
    fontsize=15, ncol=1, frameon=False
)

# --- Gráfico 2: Tiempo del minibatch ---
wedges2, _ = axs[1].pie(sizes_batch, startangle=90, radius=radius)
axs[1].axis('equal')
axs[1].set_title("Distribución del tiempo de minilote", fontsize=16)

total_batch = sum(sizes_batch)
legend_labels_batch = [
    f"{label}: {size:.2f} s ({(size / total_batch) * 100:.1f}%)"
    for label, size in zip(labels_batch, sizes_batch)
]
axs[1].legend(
    wedges2, legend_labels_batch,
    loc='upper center', bbox_to_anchor=(0.5, -0.08),
    fontsize=15, ncol=1, frameon=False
)

# Ajustar layout para dejar espacio a leyendas
plt.tight_layout()
plt.subplots_adjust(bottom=0.25)

plt.savefig("experiments/pretraining/stats/time.png")
