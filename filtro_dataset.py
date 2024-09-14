import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("players_21.csv")

dataset_filtrado = dataset.dropna(axis=1, how="any").select_dtypes(include=["number"])
print(dataset_filtrado.columns)