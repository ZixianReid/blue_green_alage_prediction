import glob
import os
from pykrige.ok import OrdinaryKriging
import pandas as pd
import numpy as np
from settings import FEATURE_USED_IN_KRIGE
import matplotlib.pyplot as plt
import ntpath
from tqdm import tqdm
from utils import generate_serial_number


class KrigeParameterLoader:
    def __init__(self, geography_point_path, lake_point_path):
        self.geography_point = pd.read_csv(geography_point_path)
        self.lake_point = pd.read_csv(lake_point_path)

    @property
    def lon_range(self):
        lon_s = self.geography_point['lon'].to_numpy()
        lon = np.sort(np.unique(lon_s))
        return lon

    @property
    def lat_range(self):
        lat_s = self.geography_point['lat'].to_numpy()
        lat = np.sort(np.unique(lat_s))
        return lat

    @property
    def lake_mask(self):
        """
        build lake mask and add lake lat and lon feature extracting from whole geography_point features
        """
        result_whole = np.array([[(xi, yj) for xi in self.lon_range] for yj in self.lat_range])
        result_whole_flatten = result_whole.reshape(-1, 2)
        lake_result = self.lake_point[['lon', 'lat']].to_numpy()

        result = np.any(np.all(result_whole_flatten[:, np.newaxis, :] == lake_result, axis=2), axis=1).astype(int)
        result = result.reshape((result.shape[0], 1))
        result = np.concatenate((result_whole_flatten, result), axis=1)
        result = result.reshape(result_whole.shape[0], result_whole.shape[1], 3)

        return result


class KrigeCommiter:

    def __init__(self, source_path, geography_point_path, lake_point_path, save_path):
        self.data_paths = glob.glob(os.path.join(source_path, "*"))
        self.KrigeParameterLoader = KrigeParameterLoader(geography_point_path, lake_point_path)
        self.save_path = save_path

    def execute(self):
        grid_lon = self.KrigeParameterLoader.lon_range
        grid_lat = self.KrigeParameterLoader.lat_range

        lake_mask = self.KrigeParameterLoader.lake_mask

        data_paths = tqdm(self.data_paths)
        for data_path in data_paths:
            df = pd.read_csv(data_path)
            k3d1_features = []
            for feature in FEATURE_USED_IN_KRIGE:
                df_extracted = df[['lon', 'lat', feature]]
                data = df_extracted.to_numpy()
                k3d1 = self.__get_kridge_grid(data, grid_lon, grid_lat)
                k3d1_features.append(k3d1)
                # self.__visual(k3d1)
            k3d1_features = np.stack(k3d1_features, axis=-1)
            mask_k3d1 = self.__filter_by_lake_mask(k3d1_features, lake_mask)
            self.__save(mask_k3d1, data_path, self.save_path)

    def __save(self, mask_k3d1, data_path, save_path):
        base_name = ntpath.basename(data_path)
        column_names = FEATURE_USED_IN_KRIGE[:]
        column_names.append('lon')
        column_names.append('lat')
        df = pd.DataFrame(mask_k3d1, columns=column_names)
        df = df[['lon', 'lat', 'chl', 'cond', 'do', 'pc']]
        df = df.sort_values(['lon', 'lat']).reset_index(drop=True)
        df['id'] = (df.index + 1).map(generate_serial_number)
        df.to_csv(os.path.join(save_path, base_name), index=False)

    def __visual(self, data):
        fig, (ax1) = plt.subplots(1)
        ax1.imshow(data, origin="lower")
        ax1.set_title("ordinary kriging")
        plt.tight_layout()
        plt.show()

    def __get_kridge_grid(self, data, grid_lon, grid_lat):
        ok3d = OrdinaryKriging(data[:, 0], data[:, 1], data[:, 2], variogram_model="spherical")
        k3d1, _ = ok3d.execute("grid", grid_lon, grid_lat)
        return k3d1

    def __filter_by_lake_mask(self, k3d1, lake_mask):
        filter_condition = lake_mask[:, :, 2] == 1
        filter_k3d1 = k3d1[filter_condition]
        filter_lake_mask = lake_mask[filter_condition][:, 0:2]
        result = np.concatenate((filter_k3d1, filter_lake_mask), axis=1) \
            .reshape(-1, filter_k3d1.shape[1] + filter_lake_mask.shape[1])
        return result


if __name__ == '__main__':
    source_path = "/media/reid/ext_disk1/blue_alage/dushu/cleaning_data"
    save_path = "/media/reid/ext_disk1/blue_alage/dushu/kridge_data"
    geography_point_path = '/media/reid/ext_disk1/blue_alage/dushu/whole_data.csv'
    lake_point_path = "/media/reid/ext_disk1/blue_alage/dushu/lack_data.csv"
    krigeCommiter = KrigeCommiter(source_path, geography_point_path, lake_point_path, save_path)
    krigeCommiter.execute()
    pass
