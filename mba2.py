import pandas as pd
import itertools

file_location = 'groceries_small.csv'

MIN_SUPPORT = 2

# data must be in the format [transaction, item]

df = pd.read_csv(file_location)
# print(df.head())

# function is used to delete infrequent item/item-sets
def delete(x, infrequent, axis=1):
    if x['item' not in list(infrequent['item'])]:
        return x

# function is used for making combinations of items/item-sets
def combinations(x, axis=1, k=1):
    name = x.iloc[0]['Person']
    combos = list(itertools.combinations(x['item'], k))
    return pd.DataFrame({'Person': name,
                         'item': combos})