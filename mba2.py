import pandas as pd
import itertools

file_location = 'groceries_small.csv'

MIN_SUPPORT = 2

# data must be in the format [transaction, item]

df = pd.read_csv(file_location)
# print(df.head())
