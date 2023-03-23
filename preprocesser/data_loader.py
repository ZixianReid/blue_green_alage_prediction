import pandas as pd
import glob
import os
from settings import FEATURE_CHIN_ENG_DICT
import ntpath


class DataLoader(object):
    def __init__(self, data_root, save_path):
        self.data_paths = glob.glob(os.path.join(data_root, "*"))
        self.save_path = save_path

    def load(self, data_path):
        """
        load_data,
        """
        data_frame = pd.read_csv(data_path).rename(columns=FEATURE_CHIN_ENG_DICT)[FEATURE_CHIN_ENG_DICT.values()]

        return data_frame

    def remove_outlier(self, data_frame):
        """
        remove value of chlo < 0 and do < 0, remove outliers that z-score > 3
        """

        data_frame = data_frame.drop(['time', 'an', 'sal', 'depth'], axis=1)

        data_frame = data_frame[data_frame['chl'] > 0]
        data_frame = data_frame[data_frame['do'] > 0]

        df_norm = (data_frame - data_frame.mean()) / (data_frame.std())

        tmp = abs(df_norm[:])
        data_frame = data_frame[abs(df_norm[:]) <= 3].dropna()

        return data_frame

    def save(self, data_frame, data_path, save_path):
        base_name = ntpath.basename(data_path)
        data_frame.to_csv(os.path.join(save_path, base_name), index=False)

    def run(self):
        """

        """
        for data_path in self.data_paths:
            data_frame = self.load(data_path)
            data_frame = self.remove_outlier(data_frame)
            self.save(data_frame, data_path, self.save_path)


if __name__ == '__main__':
    loader = DataLoader("/media/reid/ext_disk1/blue_alage/dushu/orginial_data",
                        "/media/reid/ext_disk1/blue_alage/dushu/cleaning_data")
    loader.run()
