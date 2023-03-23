import os


def mkDataDirs(out_location, fold_name):
    path = os.path.join(out_location, fold_name)
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def generate_serial_number(x):
    return '{:03d}'.format(x)
