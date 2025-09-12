import skfuzzy
import numpy
import matplotlib.pyplot as plt

x = numpy.linspace(0, 40, 100)

bell = skfuzzy.gbellmf(x, 4, 8, 15)

fig, ax = plt.subplots()
ax.plot(x, bell)
ax.set_title('Generalized Bell MF')
plt.show()