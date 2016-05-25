import pandas as pd
import numpy as np
import itertools
import time
import sys

file_location = 'groceries.csv'

# data must be in the format [transaction, item]

# reading the csv file + dropping duplicate items in a given transaction
df = pd.read_csv(file_location).drop_duplicates(subset=['Person', 'item'], keep='last')

# making the items comma separated
df['item'] += ','

# support count
support=pd.value_counts(df['item'].values, sort=False)

#printing support
print(support)

#printing supoort of a particular item
print(sup['baby food,'])

# grouping the items based on person or transaction-id
df = df.groupby('Person').sum()
print(df.head())

# support count
#support = df.groupby(['item'], as_index=False).count()

#print(support)















