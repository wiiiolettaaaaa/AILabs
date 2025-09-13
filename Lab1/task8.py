import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

x = np.linspace(-5, 25, 100)
a = fuzz.gaussmf(x, 8, 3)

notF = 1 - a

fig, ax = plt.subplots(figsize=(16,6))

ax.plot(x, a, label=("Функція"))
ax.plot(x, notF,"--", label=("Доповнення"))
plt.legend()
plt.show()