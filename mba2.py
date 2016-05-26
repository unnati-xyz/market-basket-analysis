import pandas as pd
import itertools

file_location = 'simple.csv'

MIN_SUPPORT = 1

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

for k in [2, 3, 4]:

    print("DATAFRAME ITERATION %s" % (k-1))
    drop_duplicate = df.drop_duplicates(subset=['Person', 'item'], keep='last')
    # print(drop_duplicate)

    # support gives the # of times each item/item-set is purchased
    support = drop_duplicate.groupby(['item'], as_index=False).count()
    # print(support)

    # infrequent item/item-sets. ie., item/item-sets where support < MIN_SUPPORT
    infrequent = support[support['Person'] <= MIN_SUPPORT]
    # print(infrequent)

    # frequent item/item-sets. ie., item/item-sets where support > MIN_SUPPORT
    frequent = support[support['Person'] > MIN_SUPPORT]
    #print(frequent)

    # deletes infrequent item/item-sets
    delete_infrequent = drop_duplicate.apply(delete, axis=1, infrequent=infrequent).dropna()

    # Dataframe of all combinations of items from a transaction
    df = drop_duplicate.groupby('Person', as_index=True).apply(combinations, axis=1, k=k)
    #print(df)

    mba = pd.DataFrame({'item_set': frequent['item'],
                        'support': frequent['Person'],
                        #'confidence': ,
                        #'lift':
                         })
    mba = mba.set_index(['item_set'])
    print(mba)