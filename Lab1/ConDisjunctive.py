import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

x = np.linspace(-5, 25, 100)
a = fuzz.gaussmf(x, 8, 3)
b = fuzz.gaussmf(x, 9, 4)

con = a * b
dis = a + b - con
fig, axes = plt.subplots(1, 2, figsize=(16,6))

axes[0].plot(x, a)
axes[0].plot(x, b)
axes[0].plot(x, con)
axes[1].plot(x, a)
axes[1].plot(x, b)
axes[1].plot(x, dis)

axes[0].set_title("Конʼюнкція")
axes[1].set_title("Дизʼюнкція")

plt.show()