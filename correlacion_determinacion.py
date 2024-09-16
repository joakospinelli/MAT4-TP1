import pandas as pd
from scipy.stats import t
import funciones_reg_simple as func

dataset = pd.read_csv("players_21.csv")

dataset = dataset.dropna(axis=1, how="any").select_dtypes(include=["number"])

y = dataset['value_eur']
n = len(y)

def get_determination(column):
    x = dataset[column]
    return func.get_r_2(x,y)

def get_t_stat(r):
    return r*((n-2)**0.5) / ((1-r**2)**0.5)

correlations = {}
determinations = {}
p_value = {}

for col in dataset.columns:
    if col == 'value_eur':
        continue
    determinations[col] = get_determination(col)
    correlations[col] = determinations[col]**0.5
    p_value[col] = 2 * (1 - t.cdf(get_t_stat(correlations[col]), n-2))

print("---- COEFICIENTE DETERMINACIÓN ----")

for col, value in sorted(determinations.items(), key=lambda el: el[1], reverse=True):
    print(f"{col}: {value}")

print("---- CORRELACIÓN LINEAL ----")

for col, value in sorted(correlations.items(), key=lambda el: el[1], reverse=True):
    print(f"{col}: {value}")

print("---- P-VALOR ----")

for col, value in sorted(p_value.items(), key=lambda el: el[1], reverse=False):
    print(f"{col}: {value}")