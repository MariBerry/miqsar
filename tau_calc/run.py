import sys
sys.path.append('/home/zankov/dev/miqsar')

import os
import shutil
from multiprocessing import Pool
from sklearn.model_selection import train_test_split
from utils import DataReader, ModelBuilder, scale_data

RANDOM_STATE = 45
DATA_DIR = 'descriptors'
OUT_DIR = 'models'

if os.path.exists(OUT_DIR):
   shutil.rmtree(OUT_DIR)
   os.mkdir(OUT_DIR)
else:
   os.mkdir(OUT_DIR)

def run(dataset):
    os.mkdir(os.path.join(OUT_DIR, dataset))

    # 2d tune and build
    for alg in ['sil', 'mil']:
        data = DataReader(DATA_DIR, dataset).read_data(DATA_DIR)

        model_builder = ModelBuilder(init_cuda=False)
        with open('enzyme_list.txt', 'r') as f:
            enz_list = f.read().split(',')
        if dataset in enz_list:
            model_builder.tresh = 6.5
        else:
            model_builder.tresh = 7

        for dsc in data['dsc'][alg]:
            CONF_DIR = '{}_{}'.format(dsc, alg)
            CONF_DIR = os.path.join(OUT_DIR, dataset, CONF_DIR)
            os.mkdir(CONF_DIR)
            model_builder.local_dir = CONF_DIR
            #
            bags, labels, idx = data['dsc'][alg][dsc], data['labels'], data['idx']
            x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(bags, labels, idx,
                                                                                             test_size=0.2,
                                                                                             random_state=RANDOM_STATE)
            x_train, x_val, y_train, y_val, idx_train, idx_val = train_test_split(x_train, y_train, idx_train,
                                                                                          test_size=0.25,
                                                                                          random_state=RANDOM_STATE)
            _, x_test = scale_data(x_train, x_test)
            x_train, x_val = scale_data(x_train, x_val)
            #
            nets_default, nets_tuned = model_builder.tune_nets(x_train, x_val, y_train, y_val)
            #
            model_builder.train_nets(nets_default, x_train, x_val, x_test, y_train, y_val, y_test, idx_val, idx_test,
                                             mode=alg)
            model_builder.train_nets(nets_tuned, x_train, x_val, x_test, y_train, y_val, y_test, idx_val, idx_test,
                                             mode=alg)

            return

datasets = os.listdir(DATA_DIR)
if __name__ == '__main__':
    with Pool(len(datasets)) as p:
        p.map(run, datasets, chunksize=1)
