import pandas as pd
from settings import FEATURE_CHIN_ENG_DICT
import matplotlib.pyplot as plt
import numpy as np

path = '/media/reid/ext_disk1/blue_alage/dushu/geography_point.csv'

df = pd.read_csv(path)

lon_min, lon_max = df['lon'].min(), df['lon'].max()
lat_min, lat_max = df['lat'].min(), df['lat'].max()
range_step = 0.001

grid_lon = np.arange(lon_min, lon_max, range_step)
grid_lat = np.arange(lat_min, lat_max, range_step)

result = np.array([[(xi, yj) for yj in grid_lat] for xi in grid_lon])

whole_data = result.reshape(-1, 2)

lake_data = df[['lon', 'lat']].to_numpy()



result = np.any(np.all(whole_data[:, np.newaxis, :] == lake_data, axis=2), axis=1).astype(int)
ss = np.intersect1d(whole_data, lake_data)

fig, ax = plt.subplots()

# ax.scatter(combined_coordinates[:, 0], combined_coordinates[:, 1], color='green')

df.plot(kind='scatter', x='lon', y='lat', ax=ax)

plt.legend()

plt.show()
