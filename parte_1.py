import pandas as pd
import matplotlib.pyplot as plt
import funciones_matematicas as func

dataset = pd.read_csv("players_21.csv")

value_eur = []
potential = []
overall = []
age = []
n = 0

for index, row in dataset.iterrows():
    value_eur.append(row["value_eur"] / 1000000)
    potential.append(row["potential"])
    overall.append(row["overall"])
    age.append(row["age"])
    n += 1

# 👇 Cambiar esto para probar con otros campos (y_values no debería cambiar)
x_values = age
y_values = value_eur

# Gráfico de dispersión (habría que cambiar xlabel al cambiar el criterio para x)
plt.scatter(x_values, y_values)
plt.xlabel("Edad actual del jugador")
plt.ylabel("Valor de mercado (en millones de euros)")

# Cálculo de estimación
b1 = func.get_s_xy(x_values, y_values) / func.get_s_xx(x_values)
b0 = func.get_mean(y_values) - (b1 * func.get_mean(x_values))

# Gráfico de recta de ajuste
x_line = [i for i in range(0, 100)]
y_line = [b0 + b1*x for x in x_line]

plt.plot(x_line, y_line, label="Recta de ajuste", c="red")

# Cálculo de varianza
variance = func.get_sce(x_values,y_values) / n - 2

print(f"La recta de regresión tiene los parámetros beta_0={b0} y beta_1={b1}")
print(f"El gráfico tiene una varianza de {variance}")

plt.show()