import pandas as pd
import funciones_reg_simple as func

dataset = pd.read_csv("players_21.csv")

dataset = dataset.dropna(axis=1, how="any").select_dtypes(include=["number"])

y = dataset['value_eur']

def get_determination(column):
    x = dataset[column]
    return func.get_r_2(x,y)

def get_correlation(column):
    x = dataset[column]
    return func.get_r_2(x,y)**0.5

correlations = {}
determinations = {}

for col in dataset.columns:
    determinations[col] = get_determination(col)
    correlations[col] = determinations[col]**0.5

for col, value in sorted(determinations.items(), key=lambda el: el[1], reverse=True):
    print(f"{col}: {value}")

print("--------------")

for col, value in sorted(correlations.items(), key=lambda el: el[1], reverse=True):
    print(f"{col}: {value}")