import argparse
from preprocesser.file_locations import FileLocations
from glob import glob
import os
import pandas as pd
import ntpath
import numpy as np


def organize_data(path):
    data_paths = glob(os.path.join(path, "*"))
    data = dict()
    for data_path in data_paths:
        date = str.split(ntpath.basename(data_path), ".")[0]
        df = pd.read_csv(data_path)[['chl', 'cond', 'do', 'pc']]
        data[date] = df
    df_list = list(data.values())
    result = np.stack(df_list, axis=0)
    return result


def generate_graph_seq2seq_io_data(data, x_offsets, y_offsets):
    num_samples, num_nodes = data.shape[0], data.shape[1]
    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))
    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...]
        y_t = data[t + y_offsets, ...][..., -1]
        x.append(x_t)
        y.append(y_t)

    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y


def generate_train_test_data(data, args):
    x_offsets = np.sort(
        # np.concatenate(([-week_size + 1, -day_size + 1], np.arange(-11, 1, 1)))
        np.concatenate((np.arange(-4, 1, 1),))
    )

    y_offsets = np.sort(np.arange(1, 2, 1))

    x, y = generate_graph_seq2seq_io_data(data, x_offsets, y_offsets)

    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.8)
    # train
    x_train, y_train = x[:num_train], y[:num_train]

    # test
    x_test, y_test = x[-num_test:], y[-num_test:]

    # x_2022, y_2022 = generate_graph_seq2seq_io_data(data_2022, x_offsets, y_offsets)

    print("x shape: ", x.shape, ", y shape: ", y.shape)

    # # -----------------------------------------------------------------------------
    # # 2021
    # # -----------------------------------------------------------------------------
    #
    # num_samples_2021 = x_2021.shape[0]
    # num_test_2021 = round(num_samples_2021 * 0.2)
    # num_train_2021 = round(num_samples_2021 * 0.8)
    # # train
    # x_train_2021, y_train_2021 = x_2021[:num_train_2021], y_2021[:num_train_2021]
    #
    # # test
    # x_test_2021, y_test_2021 = x_2021[-num_test_2021:], y_2021[-num_test_2021:]
    #
    # # -----------------------------------------------------------------------------
    # # 2022
    # # -----------------------------------------------------------------------------
    # num_samples_2022 = x_2022.shape[0]
    # num_test_2022 = round(num_samples_2022 * 0.2)
    # num_train_2022 = round(num_samples_2022 * 0.8)
    # # train
    # x_train_2022, y_train_2022 = x_2022[:num_train_2022], y_2022[:num_train_2022]
    #
    # # test
    # x_test_2022, y_test_2022 = x_2022[-num_test_2022:], y_2022[-num_test_2022:]
    #
    # # -----------------------------------------------------------------------------
    # # concat
    # # -----------------------------------------------------------------------------
    # x_train = np.concatenate([x_train_2021, x_train_2022], axis=0)
    # y_train = np.concatenate([y_train_2021, y_train_2022], axis=0)
    #
    # x_test = np.concatenate([x_test_2021, x_test_2022], axis=0)
    # y_test = np.concatenate([y_test_2021, y_test_2022], axis=0)


    for cat in ["train", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, "%s.npz" % cat),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )


def main(args):
    fileLocations = FileLocations(args.data_root)
    data = organize_data(fileLocations.krige_path)
    # data_2021 = data[0:5, ...]
    # data_2022 = data[5:, ...]
    generate_train_test_data(data, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="/mnt/develop/PycharmProjects/blue_green_alage_prediction/pred_core/data",
        help="Output directory."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/media/reid/ext_disk1/blue_alage/dushu",
        help="Raw traffic readings.",
    )

    args = parser.parse_args()

    main(args)
