import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from sklearn.metrics import mean_squared_error, mean_absolute_error

from Lab2.main import rules, functionCompare

def func_y(x):
    return (np.sin(x**2) / x) * np.cos(x - 2)

def func_z(x, y):
    return np.sin(np.cos(x)) + np.cos(y / 2) - np.sin(x / 2) * np.cos(x)


x_values = np.linspace(0.1, 20, 200)
y_values = func_y(x_values)
z_values = func_z(x_values, y_values)


x_means = np.linspace(min(x_values), max(x_values), 6)
y_means = np.linspace(min(y_values), max(y_values), 6)
z_means = np.linspace(min(z_values), max(z_values), 9)

x_sigma = (max(x_values) - min(x_values)) / 6 / 2
y_sigma = (max(y_values) - min(y_values)) / 6 / 2
z_sigma = (max(z_values) - min(z_values)) / 9 / 2

x_mf_gaussian = [fuzz.gaussmf(x_values, x_means[i], x_sigma) for i in range(6)]
y_lin = np.linspace(min(y_values), max(y_values), 200)
y_mf_gaussian = [fuzz.gaussmf(y_lin, y_means[i], y_sigma) for i in range(6)]
z_lin = np.linspace(min(z_values), max(z_values), 200)
z_mf_gaussian = [fuzz.gaussmf(z_lin, z_means[i], z_sigma) for i in range(9)]

plt.figure(figsize=(14, 10))

plt.subplot(3, 1, 1)
for i, mf in enumerate(x_mf_gaussian):
    plt.plot(x_values, mf, label=f"mx{i+1}")
plt.title("Функції приналежності для X")
plt.legend()

plt.subplot(3, 1, 2)
for i, mf in enumerate(y_mf_gaussian):
    plt.plot(y_lin, mf, label=f"my{i+1}")
plt.title("Функції приналежності для Y")
plt.legend()

plt.subplot(3, 1, 3)
for i, mf in enumerate(z_mf_gaussian):
    plt.plot(z_lin, mf, label=f"mz{i+1}")
plt.title("Функції приналежності для Z")
plt.legend()

plt.tight_layout()
plt.show()


print("\nRules:")
for rule in rules.keys():
    print(f"if (x is mx{rule[0] + 1}) and (y is my{rule[1] + 1}) then (z is mf{rules[rule] + 1})")


z_output = []
for x in x_values:
    best_x_func = functionCompare(x, x_means, x_sigma)
    best_y_func = functionCompare(func_y(x), y_means, y_sigma)
    best_z_func = rules[(best_x_func, best_y_func)]
    z_output.append(z_means[best_z_func])


plt.figure(figsize=(10, 6))
plt.plot(x_values, z_output, label="Model")
plt.plot(x_values, z_values, label="True")
plt.legend()
plt.title("Порівняння нечіткої моделі та істинної функції")
plt.show()


mse = mean_squared_error(z_values, z_output)
mae = mean_absolute_error(z_values, z_output)
print(f"\nMean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")