import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

x = np.linspace(-5, 25, 100)

z = fuzz.zmf(x, 2, 6)
pi = fuzz.pimf(x, 2, 5 ,8, 10)
s = fuzz.smf(x, 3, 6)

fig, axes = plt.subplots(1, 3, figsize=(16,6))
axes[0].plot(x, z)
axes[1].plot(x, pi)
axes[2].plot(x, s)

axes[0].set_title("Z-function")
axes[1].set_title("Pi-function")
axes[2].set_title("S-function")

plt.show()