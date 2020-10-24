import sys
sys.path.append('/home/zankov/dev/miqsar')

import os
import shutil
import torch
from multiprocessing import Pool
from sklearn.model_selection import train_test_split
from utils import DataReader, ModelBuilder, scale_data

RANDOM_STATE = 45
DATA_DIR = 'descriptors'
OUT_DIR = 'model'
MAX_CONF = 100
INIT_CUDA = False

if os.path.exists(OUT_DIR):
   shutil.rmtree(OUT_DIR)
   os.mkdir(OUT_DIR)
else:
   os.mkdir(OUT_DIR)


def run(dataset):
    os.mkdir(os.path.join(OUT_DIR, dataset))

    data_reader = DataReader(DATA_DIR, dataset)
    data = data_reader.read_3d(DATA_DIR, MAX_CONF)

    # tune
    bags, labels, idx = data['dsc']['3d'][MAX_CONF], data['labels'], data['idx']
    x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(bags, labels, idx, test_size=0.2, random_state=RANDOM_STATE)
    x_train, x_val, y_train, y_val, idx_train, idx_val = train_test_split(x_train, y_train, idx_train, test_size=0.25, random_state=RANDOM_STATE)
    _, x_test = scale_data(x_train, x_test)
    x_train, x_val = scale_data(x_train, x_val)

    model_builder = ModelBuilder(init_cuda=INIT_CUDA)
    model_builder.local_dir = os.path.join(OUT_DIR, dataset)
    nets_default, nets_tuned = model_builder.tune_nets(x_train, x_val, y_train, y_val)

    # 3d build
    for n_conf in [1, MAX_CONF]:
        model_builder.local_dir = os.path.join(OUT_DIR, dataset, 'pmapper_{}'.format(n_conf))
        os.mkdir(model_builder.local_dir)
        #
        data = data_reader.read_3d(DATA_DIR, n_conf)
        bags, labels, idx = data['dsc']['3d'][n_conf], data['labels'], data['idx']
        x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(bags, labels, idx, test_size=0.2, random_state=RANDOM_STATE)
        x_train, x_val, y_train, y_val, idx_train, idx_val = train_test_split(x_train, y_train, idx_train, test_size=0.25, random_state=RANDOM_STATE)
        _, x_test = scale_data(x_train, x_test)
        x_train, x_val = scale_data(x_train, x_val)
        #
        model_builder.train_nets(nets_default, x_train, x_val, x_test, y_train, y_val, y_test, idx_val, idx_test, mode='3d')
        model_builder.train_nets(nets_tuned, x_train, x_val, x_test, y_train, y_val, y_test, idx_val, idx_test, mode='3d')

    # 2d tune and build
    data = data_reader.read_2d(DATA_DIR)
    for dsc in data['dsc']['2d']:
        model_builder = ModelBuilder(init_cuda=INIT_CUDA)
        model_builder.local_dir = os.path.join(OUT_DIR, dataset, '{}_0'.format(dsc))
        os.mkdir(model_builder.local_dir)
        #
        bags, labels, idx = data['dsc']['2d'][dsc], data['labels'], data['idx']
        x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(bags, labels, idx, test_size=0.2, random_state=RANDOM_STATE)
        x_train, x_val, y_train, y_val, idx_train, idx_val = train_test_split(x_train, y_train, idx_train, test_size=0.25, random_state=RANDOM_STATE)
        _, x_test = scale_data(x_train, x_test)
        x_train, x_val = scale_data(x_train, x_val)
        #
        nets_default, nets_tuned = model_builder.tune_nets(x_train, x_val, y_train, y_val)
        #
        model_builder.train_nets(nets_default, x_train, x_val, x_test, y_train, y_val, y_test, idx_val, idx_test, mode='2d')
        model_builder.train_nets(nets_tuned, x_train, x_val, x_test, y_train, y_val, y_test, idx_val, idx_test, mode='2d')

    # 2d_3d build
    data = data_reader.read_2d_3d(DATA_DIR, MAX_CONF)

    bags, labels, idx = data['dsc']['3d'][MAX_CONF], data['labels'], data['idx']
    x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(bags, labels, idx, test_size=0.2, random_state=RANDOM_STATE)
    x_train, x_val, y_train, y_val, idx_train, idx_val = train_test_split(x_train, y_train, idx_train, test_size=0.25, random_state=RANDOM_STATE)
    _, x_test = scale_data(x_train, x_test)
    x_train, x_val = scale_data(x_train, x_val)

    model_builder = ModelBuilder(init_cuda=INIT_CUDA)
    model_builder.local_dir = os.path.join(OUT_DIR, dataset)
    nets_default, nets_tuned = model_builder.tune_nets(x_train, x_val, y_train, y_val)

    # 3d build
    for n_conf in [1, MAX_CONF]:
        model_builder.local_dir = os.path.join(OUT_DIR, dataset, 'concat_{}'.format(n_conf))
        os.mkdir(model_builder.local_dir)
        #
        data = data_reader.read_2d_3d(DATA_DIR, n_conf)
        bags, labels, idx = data['dsc']['3d'][n_conf], data['labels'], data['idx']
        x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(bags, labels, idx, test_size=0.2, random_state=RANDOM_STATE)
        x_train, x_val, y_train, y_val, idx_train, idx_val = train_test_split(x_train, y_train, idx_train, test_size=0.25, random_state=RANDOM_STATE)
        _, x_test = scale_data(x_train, x_test)
        x_train, x_val = scale_data(x_train, x_val)
        #
        model_builder.train_nets(nets_default, x_train, x_val, x_test, y_train, y_val, y_test, idx_val, idx_test, mode='3d')
        model_builder.train_nets(nets_tuned, x_train, x_val, x_test, y_train, y_val, y_test, idx_val, idx_test, mode='3d')

    return

datasets = os.listdir(DATA_DIR)
if __name__ == '__main__':
    with Pool(len(datasets)) as p:
        p.map(run, datasets, chunksize=1)



