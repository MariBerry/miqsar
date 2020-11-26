import sys
sys.path.append('/home/zankov/dev/miqsar')

import os
import shutil
from miqsar.descriptor_calculation.rdkit_2d import calc_2d_descriptors
from miqsar.descriptor_calculation.rdkit_ph import calc_ph_descriptors
from miqsar.descriptor_calculation.rdkit_morgan import calc_morgan_descriptors

from miqsar.conformer_generation.gen_conformers import gen_confs

from utils import calc_3d_rdkit, calc_3d_pmapper

# NCPU = 15
# INP_DIR = 'descriptors'
# NCONFS_LIST = [1, 100]
#

# for chembl in os.listdir(INP_DIR):
#
#     # remove
#     tmp = list(range(10, 200, 10))
#     tmp.remove(100)
#     for i in tmp:
#         try:
#             os.remove(os.path.join(INP_DIR, chembl, 'conf-{}_{}.pkl'.format(chembl, i)))
#             os.remove(os.path.join(INP_DIR, chembl, 'PhFprPmapper_conf-{}_{}.csv'.format(chembl, i).format(chembl, i)))
#         except:
#             pass
#
#     # conf files
#     conf_files = ['conf-{}_1.pkl'.format(chembl), 'conf-{}_100.pkl'.format(chembl)]
#     if not all([i in os.listdir(os.path.join(INP_DIR, chembl)) for i in conf_files]):
#         continue
#     conf_files = [os.path.join(INP_DIR, chembl, 'conf-{}_1.pkl'.format(chembl)),
#                   os.path.join(INP_DIR, chembl, 'conf-{}_100.pkl'.format(chembl))]
#     dsc_dir = os.path.join(INP_DIR, chembl)
#
#     # calc 2d
#     #calc_2d_descriptors(fname=data_file, ncpu=NCPU, path=dsc_dir)
#     #calc_ph_descriptors(fname=data_file, ncpu=NCPU, path=dsc_dir)
#     #calc_morgan_descriptors(fname=data_file, path=dsc_dir)
#
#     # calc 3d
#     #calc_3d_pmapper(conf_files, nconfs_list=NCONFS_LIST, stereo=False, ncpu=NCPU, path=dsc_dir)
#     calc_3d_rdkit(conf_files, nconfs_list=NCONFS_LIST, stereo=False, ncpu=NCPU, path=dsc_dir)

NCPU = 15
INP_DIR = 'datasets'
OUT_DIR = 'descriptors'
NCONFS_LIST = [1, 100]

for chembl in os.listdir(INP_DIR):

    dsc_dir = os.path.join(OUT_DIR, chembl.split('.')[0])

    if os.path.exists(dsc_dir):
        shutil.rmtree(dsc_dir)
        os.mkdir(dsc_dir)
    else:
        os.mkdir(dsc_dir)

    chembl = os.path.join(INP_DIR, chembl)

    # calc 2d
    calc_2d_descriptors(fname=chembl, ncpu=NCPU, path=dsc_dir)
    calc_ph_descriptors(fname=chembl, ncpu=NCPU, path=dsc_dir)
    calc_morgan_descriptors(fname=chembl, path=dsc_dir)

    # calc 3d
    conf_files = gen_confs(chembl, nconfs_list=NCONFS_LIST, stereo=False, path=dsc_dir, ncpu=NCPU)

    calc_3d_pmapper(conf_files, nconfs_list=NCONFS_LIST, stereo=False, ncpu=NCPU, path=dsc_dir)
    calc_3d_rdkit(conf_files, nconfs_list=NCONFS_LIST, stereo=False, ncpu=NCPU, path=dsc_dir)


