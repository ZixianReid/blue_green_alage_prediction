import itertools
import math
import pandas as pd
import os
from utils import generate_serial_number
import numpy as np
import pickle

def generate_network_distance(file_path):
    """
    generate distnace between each points (must be executed after krige_commiter)
    :param file_path:
    """
    df = pd.read_csv(os.path.join(file_path, '20210901.csv'))
    pairs = list(itertools.combinations(df.index, 2))
    distances = []
    for i, j in pairs:
        distance = math.sqrt((df.loc[i, 'lon'] - df.loc[j, 'lon']) ** 2 + (df.loc[i, 'lat'] - df.loc[j, 'lat']) ** 2)
        distances.append(distance)

    # create new DataFrame with distances
    df_distances = pd.DataFrame({'from': [str(df.loc[i, 'id']).zfill(3) for i, j in pairs],
                                 'to': [str(df.loc[j, 'id']).zfill(3) for i, j in pairs],
                                 'distance': distances})
    # display resulting DataFrame
    df_distances.to_csv(os.path.join(os.path.dirname(file_path), 'distances_dushu.csv'), index=False)

    df['id'] = df['id'].astype(str).str.zfill(3)
    with open(os.path.join(os.path.dirname(file_path), 'graph_ids.txt'), 'w') as f:
        f.write(','.join(df['id'].tolist()))


def get_adjacency_matrix(distance_path, ids_path, adj_path, normalized_k=0.1):
    """
    :param distance_df: data frame with three columns: [from, to, distance].
    :param sensor_ids: list of sensor ids.
    :param normalized_k: entries that become lower than normalized_k after normalization are set to zero for sparsity.
    :return:
    """

    distance_df = pd.read_csv(distance_path, dtype={'from': 'str', 'to': 'str'})
    with open(ids_path) as f:
        sensor_ids = f.read().strip().split(',')
    num_sensors = len(sensor_ids)
    dist_mx = np.zeros((num_sensors, num_sensors), dtype=np.float32)
    dist_mx[:] = np.inf
    # Builds sensor id to index map.
    sensor_id_to_ind = {}
    for i, sensor_id in enumerate(sensor_ids):
        sensor_id_to_ind[sensor_id] = i

    # Fills cells in the matrix with distances.
    for row in distance_df.values:
        if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind:
            continue
        dist_mx[sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]] = row[2]

    # Calculates the standard deviation as theta.
    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(dist_mx / std))
    # Make the adjacent matrix symmetric by taking the max.
    # adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])

    # Sets entries that lower than a threshold, i.e., k, to zero for sparsity.
    adj_mx[adj_mx < normalized_k] = 0

    with open(adj_path, 'wb') as f:
        pickle.dump([sensor_ids, sensor_id_to_ind, adj_mx], f, protocol=2)
