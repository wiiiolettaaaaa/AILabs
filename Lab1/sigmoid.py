import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

x = np.linspace(-5, 25, 100)

sigm = fuzz.sigmf(x, 3, 6)
dsigm = fuzz.dsigmf(x, 2, 4, 7, 15)
psigm = fuzz.psigmf(x, 3, 5, 8, 14)

fig, axes = plt.subplots(1, 3, figsize=(16,6))
axes[0].plot(x, sigm)
axes[1].plot(x, dsigm)
axes[2].plot(x, psigm)

axes[0].set_title('Основна одностороння')
axes[1].set_title('Додаткова двостороня')
axes[2].set_title('Додаткова несиметрична')

plt.show()
