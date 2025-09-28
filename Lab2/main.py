import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from tabulate import tabulate
from sklearn.metrics import mean_squared_error, mean_absolute_error

def func_y(x):
    return (np.sin(x**2) / x) * np.cos(x - 2)

def func_z(x, y):
    return np.sin(np.cos(x)) + np.cos(y / 2) - np.sin(x / 2) * np.cos(x)


x_values = np.linspace(0.1, 20, 200)
y_values = func_y(x_values)
z_values = func_z(x_values, y_values)


plt.plot(x_values, y_values)
plt.title("Y-function: sin(x^2)/x * cos(x-2)")
plt.show()

plt.plot(x_values, z_values)
plt.title("Z-function: sin(cos(x)) + cos(y/2) - sin(x/2)*cos(x)")
plt.show()


x_means = np.linspace(min(x_values), max(x_values), num=6)
y_means = np.linspace(min(y_values), max(y_values), num=6)
z_means = np.linspace(min(z_values), max(z_values), num=9)


mx = [fuzz.trimf(x_values, [x_means[i]-3, x_means[i], x_means[i]+3]) for i in range(6)]
my = [fuzz.trimf(np.linspace(min(y_values), max(y_values), 200),
                 [y_means[i]-3, y_means[i], y_means[i]+3]) for i in range(6)]
mf = [fuzz.trimf(np.linspace(min(z_values), max(z_values), 200),
                 [z_means[i]-4, z_means[i], z_means[i]+4]) for i in range(9)]


for i in range(6):
    plt.plot(x_values, mx[i])
plt.title("X trimf")
plt.show()

for i in range(6):
    plt.plot(np.linspace(min(y_values), max(y_values), 200), my[i])
plt.title("Y trimf")
plt.show()

for i in range(9):
    plt.plot(np.linspace(min(z_values), max(z_values), 200), mf[i])
plt.title("Z trimf")
plt.show()


def calculate_trimf(x, a, b, c):
    if a <= x < b:
        return (x - a) / (b - a)
    elif b <= x <= c:
        return (c - x) / (c - b)
    else:
        return 0


def functionCompare(value, means, diff):
    best_func_value = -float("inf")
    best_index = -1
    for index, mean in enumerate(means):
        ff = calculate_trimf(value, mean - diff, mean, mean + diff)
        if ff > best_func_value:
            best_func_value = ff
            best_index = index
    return best_index


print("Таблиця значень")
table = [["y\\x"] + [str(round(x, 2)) for x in x_means]]
for y in y_means:
    row = [round(y, 2)]
    for x in x_means:
        z = func_z(x, y)
        row.append(round(z, 2))
    table.append(row)

print(tabulate(table, tablefmt="grid"))


print("Таблиця з назвами функцій")
table = [["y\\x"] + ["mx" + str(i+1) for i in range(6)]]
rules = {}

for i in range(6):
    row = ["my" + str(i+1)]
    for j in range(6):
        z = func_z(x_means[j], y_means[i])
        best_func = functionCompare(z, z_means, diff=4)
        row.append("mf" + str(best_func + 1))
        rules[(j, i)] = best_func
    table.append(row)

print(tabulate(table, tablefmt="grid"))

print("\nRules:")
for rule in rules.keys():
    print(
        f"if (x is mx{rule[0] + 1}) and (y is my{rule[1] + 1}) "
        f"then (z is mf{rules[rule] + 1})"
    )

z_output = []
for x in x_values:
    best_x_func = functionCompare(x, x_means, diff=3)
    y_val = func_y(x)
    best_y_func = functionCompare(y_val, y_means, diff=3)
    best_z_func = rules[(best_x_func, best_y_func)]
    z_output.append(z_means[best_z_func])

plt.plot(x_values, z_output, label="Model")
plt.plot(x_values, z_values, label="True")
plt.title("Справжня і змодельована функції")
plt.legend()
plt.show()

mse = mean_squared_error(z_values, z_output)
mae = mean_absolute_error(z_values, z_output)

print(f"\nMean Squared Error (MSE): {mse:.6f}")
print(f"Mean Absolute Error (MAE): {mae:.6f}")


