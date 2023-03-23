from . import utils
import os


class FileLocations:

    def __init__(self, root_dir):
        self.original_path = utils.mkDataDirs(root_dir, 'orginial_data')
        self.cleaning_path = utils.mkDataDirs(root_dir, 'cleaning_data')
        self.krige_path = utils.mkDataDirs(root_dir, 'krige_data')
        self.geography_points_path = os.path.join(root_dir, 'whole_data.csv')
        self.lake_points_path = os.path.join(root_dir, 'lack_data.csv')
        self.distance_path = os.path.join(root_dir, 'distances_dushu.csv')
        self.ids_path = os.path.join(root_dir, 'graph_ids.txt')
        self.adj_path = os.path.join(root_dir, 'adj_mat.pkl')
