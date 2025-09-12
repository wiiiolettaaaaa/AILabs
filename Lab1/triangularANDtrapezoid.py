import skfuzzy
import numpy
import matplotlib.pyplot as plt

plt.rcParams['axes.grid'] = True

x1 = numpy.linspace(0, 20, 200)

triMF = skfuzzy.trimf(x1, [5, 10, 18])
trapMF = skfuzzy.trapmf(x1, [3, 6, 8, 14])

fig, axes = plt.subplots(1, 2, figsize=(16,6))
axes[0].plot(x1, triMF)
axes[1].plot(x1, trapMF)

axes[0].set_title('Triangle MF')
axes[1].set_title('Trapezoid MF')

plt.show()