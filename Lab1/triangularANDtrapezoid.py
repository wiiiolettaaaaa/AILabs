import skfuzzy
import numpy
import matplotlib.pyplot as plt

plt.rcParams['axes.grid'] = True

x = numpy.linspace(0, 20, 200)

triMF = skfuzzy.trimf(x, [5, 10, 18])
trapMF = skfuzzy.trapmf(x, [3, 6, 8, 14])

fig, axes = plt.subplots(1, 2, figsize=(16,6))
axes[0].plot(x, triMF)
axes[1].plot(x, trapMF)

axes[0].set_title('Triangle MF')
axes[1].set_title('Trapezoid MF')

plt.show()