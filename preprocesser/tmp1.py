import pandas as pd
import itertools
import math

# create sample DataFrame
df = pd.DataFrame({'x': [0, 3, 5, 8], 'y': [0, 4, 7, 12], 'serial_number': [1, 2, 3, 4]})

# generate pairs of rows
pairs = list(itertools.combinations(df.index, 2))

# calculate distance between pairs
distances = []
for i, j in pairs:
    distance = math.sqrt((df.loc[i, 'x'] - df.loc[j, 'x']) ** 2 + (df.loc[i, 'y'] - df.loc[j, 'y']) ** 2)
    distances.append(distance)

# create new DataFrame with distances
df_distances = pd.DataFrame({'from': [str(df.loc[i, 'serial_number']).zfill(3) for i, j in pairs],
                             'to': [str(df.loc[j, 'serial_number']).zfill(3) for i, j in pairs],
                             'distance': distances})

# display resulting DataFrame
print(df_distances)