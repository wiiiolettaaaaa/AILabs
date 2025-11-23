import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms


def func_y(x):
    if x == 0:
        return np.inf
    return np.sin(x**2) / x * np.cos(x - 2)


def func_z(x, y):
    return np.sin(np.cos(x)) + np.cos(y / 2) * np.sin(x / 2) * np.cos(x)


if "FitnessMin" not in creator.__dict__:
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if "IndividualMin" not in creator.__dict__:
    creator.create("IndividualMin", list, fitness=creator.FitnessMin)

toolbox_min = base.Toolbox()
np.random.seed(42)

toolbox_min.register("attr_float", np.random.uniform, -2, 8)
toolbox_min.register("individual", tools.initRepeat, creator.IndividualMin, toolbox_min.attr_float, 1)
toolbox_min.register("population", tools.initRepeat, list, toolbox_min.individual)


def evaluate_min(individual):
    x = individual[0]
    return func_y(x),


toolbox_min.register("evaluate", evaluate_min)
toolbox_min.register("mate", tools.cxBlend, alpha=0.5)
toolbox_min.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox_min.register("select", tools.selTournament, tournsize=3)


if "FitnessMax" not in creator.__dict__:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if "IndividualMax" not in creator.__dict__:
    creator.create("IndividualMax", list, fitness=creator.FitnessMax)

toolbox_max = base.Toolbox()
toolbox_max.register("attr_float", np.random.uniform, -2, 8)
toolbox_max.register("individual", tools.initRepeat, creator.IndividualMax, toolbox_max.attr_float, 2)
toolbox_max.register("population", tools.initRepeat, list, toolbox_max.individual)


def evaluate_max(individual):
    x, y = individual
    if not (-2 <= x <= 8 and -2 <= y <= 8):
        return -np.inf,
    return func_z(x, y),


toolbox_max.register("evaluate", evaluate_max)
toolbox_max.register("mate", tools.cxBlend, alpha=0.5)
toolbox_max.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox_max.register("select", tools.selTournament, tournsize=3)

def run_ga(toolbox, ngen=100, cxpb=0.7, mutpb=0.2):
    population = toolbox.population(n=100)
    algorithms.eaSimple(population, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen, verbose=False)
    best_individual = tools.selBest(population, k=1)[0]
    return best_individual

best_min = run_ga(toolbox_min)
min_x = best_min[0]
min_y = func_y(min_x)

best_max = run_ga(toolbox_max)
max_x, max_y = best_max[0], best_max[1]
max_z = func_z(max_x, max_y)

print("Мінімум func_y(x): x =", min_x, ", y =", min_y)
print("Максимум func_z(x, y): x =", max_x, ", y =", max_y, ", z =", max_z)


x_vals = np.linspace(-2, 8, 400)
y_vals = [func_y(x) for x in x_vals]

plt.figure(figsize=(10, 5))
plt.plot(x_vals, y_vals, label='func_y(x) = sin(x^2)/x * cos(x - 2)')
plt.scatter(min_x, min_y, color='red', label='Мінімум')
plt.title("Графік функції func_y(x)")
plt.xlabel("x")
plt.ylabel("func_y(x)")
plt.legend()
plt.grid(True)
plt.show()

x_vals = np.linspace(-2, 8, 50)
y_vals = np.linspace(-2, 8, 50)

if max_x < x_vals.min() or max_x > x_vals.max():
    x_vals = np.linspace(min(max_x, -2), max(max_x, 8), 50)
if max_y < y_vals.min() or max_y > y_vals.max():
    y_vals = np.linspace(min(max_y, -2), max(max_y, 8), 50)

X, Y = np.meshgrid(x_vals, y_vals)
Z = func_z(X, Y)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.7)

ax.scatter(max_x, max_y, max_z, color='red', s=50, label='Максимум')
ax.set_title("3D-графік функції func_z(x, y)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("func_z(x, y)")
ax.legend()
plt.show()