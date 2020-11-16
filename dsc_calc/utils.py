import sys
sys.path.append('/home/zankov/dev/miqsar')

import pickle
import joblib
import pandas as pd
from miqsar.descriptor_calculation.rdkit_3d import calc_3d_descriptors
from miqsar.descriptor_calculation.pmapper_3d import calc_pmapper_descriptors


def read_pkl(fname):
    with open(fname, 'rb') as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


def calc_3d_rdkit(conf_files, nconfs_list=[1], stereo=False, path='.', ncpu=10):

    res = {}
    for file, n_conf in zip(conf_files, nconfs_list):
        confs = list(read_pkl(file))
        res.setdefault(n_conf, [])
        for _, conf, _, _ in confs:
            res[n_conf].append(conf)

    dsc_file = calc_3d_descriptors(conf_files[-1], path=path, ncpu=ncpu, del_log=True)

    for n_conf, idx in res.items():
        data = pd.read_csv(dsc_file)
        data.index = data['mol_title']
        name = dsc_file.replace('_{}.csv'.format(max(nconfs_list)), '_{}.csv'.format(n_conf))
        data.loc[idx].to_csv(name, index=False)

    return


def calc_3d_pmapper(conf_files, nconfs_list=[1], stereo=False, path='.', ncpu=10):

    for conf in conf_files:
        dsc_file = calc_pmapper_descriptors(conf, path=path, ncpu=ncpu, col_clean=None, del_undef=True)

        with open(dsc_file, 'rb') as inp:
            data = joblib.load(inp)

        if 'mol_title' not in data.columns:
            data = data.reset_index()
        data['mol_id'] = data['mol_id'].str.lower()
        fname = dsc_file.split
        data.to_csv(dsc_file.replace('_proc.pkl', '.csv'), index=False)

    return

