import pandas as pd
from preprocesser.settings import FEATURE_CHIN_ENG_DICT
import argparse
import geopandas
from pykrige.ok import OrdinaryKriging
import numpy as np
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt


def load(data_path):
    data_frame = pd.read_csv(data_path).rename(columns=FEATURE_CHIN_ENG_DICT)[FEATURE_CHIN_ENG_DICT.values()]

    return data_frame


def remove_outlier(data_frame):
    data_frame = data_frame.drop(['time', 'an', 'sal', 'depth'], axis=1)

    data_frame = data_frame[data_frame['chl'] > 0]
    data_frame = data_frame[data_frame['do'] > 0]

    df_norm = (data_frame - data_frame.mean()) / (data_frame.std())

    data_frame = data_frame[abs(df_norm[:]) <= 3].dropna()

    return data_frame


def get_krige_grid(data_lake, polygon):
    data_lake = data_lake.to_numpy()
    range_step = 0.0005
    lon_min, lon_max = min(polygon[0]), max(polygon[0])
    lat_min, lat_max = min(polygon[1]), max(polygon[1])
    grid_lon = np.arange(lon_min, lon_max, range_step)
    grid_lat = np.arange(lat_min, lat_max, range_step)
    ok3d = OrdinaryKriging(data_lake[:, 0], data_lake[:, 1], data_lake[:, 2], variogram_model="spherical")
    k3d1, _ = ok3d.execute("grid", grid_lon, grid_lat)
    result_whole = np.array([[(xi, yj) for xi in grid_lon] for yj in grid_lat])
    k3d1 = k3d1.reshape(-1, 1)
    result_whole = result_whole.reshape(-1, 2)

    output = np.concatenate((result_whole, k3d1), axis=1)
    return output


def filter_krige_output(krige_output, polygon):
    polygon = Polygon(zip(polygon[0], polygon[1]))

    filtered_krige_output = []

    for point in krige_output:
        x, y = point[:2]
        if polygon.contains(Point(x, y)):
            filtered_krige_output.append(point)

    filtered_krige_output = np.array(filtered_krige_output)

    return filtered_krige_output


def visual(filtered_krige_output):
    plt.scatter(filtered_krige_output[:, 0], filtered_krige_output[:, 1],
                c=filtered_krige_output[:, 2], cmap='cool')
    plt.colorbar()

    # Set plot title and axis labels
    plt.title('Cyanobacteria concentration distribution')
    plt.xlabel('lon')
    plt.ylabel('lat')

    # Display the plot
    plt.show()


def main(data_path, lake_path):
    df_data = load(data_path)
    df_data = remove_outlier(df_data)

    df_lake = geopandas.read_file(lake_path, rows=1000)

    polygon = df_lake.loc[0, 'geometry'].exterior.coords.xy

    df_lake = df_data[['lon', 'lat', 'pc']]

    krige_output = get_krige_grid(df_lake, polygon)

    filtered_kriger_output = filter_krige_output(krige_output, polygon)

    visual(filtered_kriger_output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="blue green alage prediction")
    parser.add_argument(
        "--data_path",
        default="/media/reid/ext_disk1/blue_alage/dushu/orginial_data/20210929.csv",
        metavar="FILE",
        type=str,
    )

    parser.add_argument(
        "--lake_path",
        default="/media/reid/ext_disk1/dushu/dushu1.shp",
        metavar="FILE",
        type=str,
    )

    args = parser.parse_args()

    main(args.data_path, args.lake_path)
