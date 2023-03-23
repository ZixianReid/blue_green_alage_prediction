import numpy as np
import os


class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True, shuffle=False):
        """

        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        if shuffle:
            permutation = np.random.permutation(self.size)
            xs, ys = xs[permutation], ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()


class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data, data_type):
        if data_type == 'x':
            return (data - self.mean) / self.std

        if data_type == 'y':
            return (data - self.mean[-1]) / self.std[-1]

    def inverse_transform(self, data, data_type):
        if data_type == 'x':
            return (data * self.std) + self.mean
        if data_type == 'y':
            return (data * self.std[-1]) + self.mean[-1]


def make_data_loader(cfg):
    data = {}
    batch_size = cfg.DATA.BATCH_SIZE
    test_batch_size = cfg.DATA.TEST_BATCH_SIZE
    dataset_dir = cfg.DATA.DATASET_DIR
    for category in ['train', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']

    scaler = StandardScaler(mean=np.mean(data['x_train'], axis=(0, 1, 2)), std=np.std(data['x_train'], axis=(0, 1, 2)))

    for category in ['train', 'test']:
        data['x_' + category] = scaler.transform(data['x_' + category], 'x')
        data['y_' + category] = scaler.transform(data['y_' + category], 'y')
        pass

    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size, shuffle=True)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size, shuffle=False)
    data['scaler'] = scaler

    return data
