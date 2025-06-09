import pymannkendall as mk
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
import numpy as np

accuracies = [
    #42.19, 42.58, 41.80, 41.99, 36.13, 36.91, 23.24, 31.05, 34.18, 33.40, 30.86, 34.18, 42.19, 41.60, 39.26,
    40.04, 43.36, 41.41, 40.43, 43.95, 40.23, 38.28, 43.16, 41.41, 42.19, 44.73, 42.19, 44.92, 45.90, 40.62, 42.77,
    41.99, 42.58, 43.95, 39.06, 37.70, 43.16, 40.62, 42.97, 42.38, 40.23, 40.62, 41.99, 39.26, 46.68, 41.99, 40.43,
    41.21, 43.95, 41.02, 43.55, 39.84, 41.41, 42.58, 40.04, 43.55, 40.43, 39.84, 41.99, 40.23, 44.73, 45.51, 38.28,
    42.77, 38.87, 46.09, 43.75, 42.97, 44.34, 44.34, 42.19, 43.95, 38.28, 44.73, 45.12, 45.51, 41.21, 43.75, 40.82,
    45.51, 39.45, 46.09, 46.68, 42.38, 40.23, 45.70, 41.80, 48.44, 42.97, 44.73, 43.95, 44.34, 41.02, 44.14, 41.99,
    44.53, 44.14, 44.73, 46.09, 47.85, 39.45, 43.55, 40.43, 42.97, 40.04, 43.36, 42.58, 45.51, 48.05, 40.82, 43.55,
    41.21, 44.14, 44.92, 44.73, 42.97, 42.38, 44.92, 41.60, 45.70, 46.48, 47.07, 46.29, 45.51, 40.43, 43.16, 42.58, 
    44.53, 42.19, 42.58, 45.90, 43.55, 46.48, 41.41, 41.80, 44.92, 47.27, 46.48, 44.14, 46.68, 42.77, 41.80, 43.95,
    42.77, 46.29, 43.75, 45.51, 46.88, 41.21, 44.14, 45.51, 42.19, 47.27, 42.77, 48.05, 41.99, 42.38, 44.53, 40.82, 
    43.16, 44.34, 42.19, 43.16, 40.43, 45.31, 44.53, 41.99, 44.14, 43.55, 40.62, 43.16, 43.95, 41.21, 41.41, 43.55, 
    46.48, 45.90, 40.23, 45.51, 45.12, 44.92, 45.31, 44.14, 41.80, 44.73, 41.99, 45.12, 42.38, 39.45, 46.09, 45.31, 
    45.90, 45.31, 42.58, 44.53, 38.67, 43.95, 43.95, 42.97, 45.12, 49.02, 41.60, 43.55, 46.68, 47.66, 47.07, 44.34, 
    47.66, 43.36, 45.51, 48.24, 41.02, 43.55, 44.14, 42.19, 41.41, 43.55, 42.77, 42.58, 44.34, 46.09, 44.34, 48.63,
    47.07, 43.95, 45.51, 47.07, 43.36, 46.68, 45.90, 43.95, 50.00, 46.48, 45.51, 39.65, 49.80, 44.73, 44.53, 46.09, 
    46.48, 46.09, 41.80, 43.55, 45.51, 48.44, 45.70, 46.68, 43.95, 48.05, 48.63, 43.16, 42.77, 44.92, 41.41, 47.27, 
    44.34, 46.68, 49.02, 47.66, 43.16, 50.98, 47.46, 47.07, 44.53, 45.12, 44.73, 39.65, 45.51, 46.68, 49.22, 42.97, 
    48.44, 41.02, 49.41, 44.73, 46.09, 46.48, 43.36, 39.06, 42.97, 43.36, 46.48, 41.41, 46.48, 42.77, 47.27, 43.16, 
    41.60, 42.97, 43.95, 46.68, 46.68, 50.59, 44.92, 48.83, 43.16
] # Step 18

x = np.arange(len(accuracies)).reshape(-1, 1)
y = np.array(accuracies)

reg = LinearRegression().fit(x, y)
trend = reg.predict(x)
slope = reg.coef_[0]
intercept = reg.intercept_

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 4.5))

axes = axes.flatten()

ax = axes[0]

domain = list(range(len(accuracies)))

ax.plot(domain, accuracies, alpha = 0.7)
ax.plot(x, trend, color='orange', label=f'Tend (slope = {slope:.4f})')

ax.set_title(f"Accuracy")

accuracies_array = np.array(accuracies)


plot_acf(accuracies_array, lags=20, ax = axes[1])  
axes[1].set_title(f"Accuracy's ACF")

plot_acf(accuracies_array - trend, lags=20, ax = axes[2])  
axes[2].set_title(f"Detrended accuracy's ACF")


#print(mk.hamed_rao_modification_test(accuracies_array))
#print(mk.original_test(accuracies_array))

mk_result = mk.original_test(accuracies_array)

# Crear el texto a mostrar
text = (
    f"Mann-Kendall Test\n"
    f"Trend: {mk_result.trend} | "
    f"p-value: {mk_result.p:.2e} | "
    f"Tau: {mk_result.Tau:.3f} | "
    f"Slope: {mk_result.slope:.4f}"
)


# Añadir texto en la parte inferior de la figura
plt.figtext(
    0.5,               # Posición horizontal (0: izquierda, 1: derecha)
    0.05,             # Posición vertical (valores negativos están fuera del gráfico)
    text,              # El texto
    wrap=True,
    horizontalalignment='center',
    fontsize=10
)
plt.tight_layout(rect=[0, 0.1, 1, 1])

plt.savefig("experiments/pretraining/stats/accuracy.png")
"""
Modified_Mann_Kendall_Test_Hamed_Rao_Approach(trend='increasing', h=True, p=0.0, z=21.797564572941184, 
Tau=0.2990162434225578, s=1307.0, var_s=3589.801280177929, slope=0.0472972972972973, intercept=39.89067567567568)
"""


