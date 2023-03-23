import argparse
from data_loader import DataLoader
from file_locations import FileLocations
from krige_commiter import KrigeCommiter
from graph_maker import generate_network_distance, get_adjacency_matrix


def run(fileLocations):
    # dataLoader = DataLoader(fileLocations.original_path, fileLocations.cleaning_path)
    # dataLoader.run()
    #
    # krigeCommiter = KrigeCommiter(fileLocations.cleaning_path, fileLocations.geography_points_path,
    #                               fileLocations.lake_points_path, fileLocations.krige_path)
    # krigeCommiter.execute()
    # generate_network_distance(fileLocations.krige_path)

    get_adjacency_matrix(fileLocations.distance_path, fileLocations.ids_path, fileLocations.adj_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="proprecess alge data",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--data_root', type=str, default="/media/reid/ext_disk1/blue_alage/dushu")

    args = parser.parse_args()

    fileLocations = FileLocations(args.data_root)

    run(fileLocations)
