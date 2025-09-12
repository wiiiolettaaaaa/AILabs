import skfuzzy
import numpy
import matplotlib.pyplot as plt

x = numpy.linspace(-40, 40, 100)

gauss = skfuzzy.gaussmf(x, 0, 5)
gauss21 = skfuzzy.gauss2mf(x, -15, 3, -5, 6)
gauss22 = skfuzzy.gauss2mf(x, 5, 4, 15, 7)
gauss23 = skfuzzy.gauss2mf(x, -10, 8, 10, 8)

fig, axes = plt.subplots(1, 2, figsize=(16,6))
axes[0].plot(x, gauss)
axes[1].plot(x, gauss21, label="[-15, 3, -5, 6]")
axes[1].plot(x, gauss22, label="[5, 4, 15, 7]")
axes[1].plot(x, gauss23, label="[-10, 8, 10, 8]")
axes[0].set_title('Gaussian MF')
axes[1].set_title('Gaussian two-combined MF')
plt.legend()
plt.show()