import os
import sys
import string
import random
import itertools
from multiprocessing import Pool, cpu_count
from collections import defaultdict, Counter

from .read_input import read_input
from pmapper.utils import load_multi_conf_mol


class SvmSaver:

    def __init__(self, file_name):
        self.__fname = file_name
        self.__varnames_fname = os.path.splitext(file_name)[0] + '.colnames'
        self.__molnames_fname = os.path.splitext(file_name)[0] + '.rownames'
        self.__varnames = []
        if os.path.isfile(self.__fname):
            os.remove(self.__fname)
        if os.path.isfile(self.__molnames_fname):
            os.remove(self.__molnames_fname)
        if os.path.isfile(self.__varnames_fname):
            os.remove(self.__varnames_fname)

    def save_mol_descriptors(self, mol_name, mol_descr_dict):

        values = []

        for i, varname in enumerate(self.__varnames):
            if varname in mol_descr_dict:
                values.append((i, mol_descr_dict[varname]))

        new_varnames = list(sorted(set(mol_descr_dict) - set(self.__varnames)))
        for i, varname in enumerate(new_varnames):
            values.append((len(self.__varnames) + i, mol_descr_dict[varname]))

        if values:  # values can be empty if all descriptors are zero

            self.__varnames.extend(new_varnames)

            with open(self.__molnames_fname, 'at') as f:
                f.write(mol_name + '\n')

            if new_varnames:
                with open(self.__varnames_fname, 'at') as f:
                    f.write('\n'.join(new_varnames) + '\n')

            with open(self.__fname, 'at') as f:
                values = sorted(values)
                values_str = ('%i:%i' % (i, v) for i, v in values)
                f.write(' '.join(values_str) + '\n')

            return tuple(i for i, v in values)

        return tuple()


def process_mol(mol, mol_title):
    ps = load_multi_conf_mol(mol)
    res = [p.get_descriptors() for p in ps]
    ids = [c.GetId() for c in mol.GetConformers()]
    ids, res = zip(*sorted(zip(ids, res)))  # reorder output by conf ids
    return mol_title, res


def process_mol_map(items):
    return process_mol(*items)


def calc_pmapper_descriptors(input, output, remove=False, keep_temp=False, ncpu=1, verbose=False):

    if remove < 0 or remove > 1:
        raise ValueError('Value of the "remove" argument is out of range [0, 1]')

    pool = Pool(max(min(ncpu, cpu_count()), 1))

    tmp_fname = os.path.splitext(output)[0] + '.' + ''.join(random.sample(string.ascii_lowercase, 6)) + '.svm'
    svm = SvmSaver(tmp_fname)

    stat = defaultdict(set)

    # create temp file with all descriptors
    mols = [(i[0], i[1]) for i in list(read_input(input))]
    for i, (mol_title, desc) in enumerate(pool.imap(process_mol_map, mols, chunksize=10), 1):
        for desc_dict in desc:
            if desc_dict:
                ids = svm.save_mol_descriptors(mol_title, desc_dict)
                stat[mol_title].update(ids)
        if verbose and i % 10 == 0:
            sys.stderr.write(f'\r{i} molecule records were processed')
    sys.stderr.write('\n')

    if remove == 0:  # if no remove - rename temp files to output files
        os.rename(tmp_fname, output)
        os.rename(os.path.splitext(tmp_fname)[0] + '.colnames', os.path.splitext(output)[0] + '.colnames')
        os.rename(os.path.splitext(tmp_fname)[0] + '.rownames', os.path.splitext(output)[0] + '.rownames')

    else:
        # determine frequency of descriptors occurrence and select frequently occurred
        c = Counter(itertools.chain.from_iterable(stat.values()))
        threshold = len(stat) * remove
        desc_ids = {k for k, v in c.items() if v >= threshold}

        # create output files with removed descriptors

        replace_dict = dict()  # old_id, new_id
        with open(os.path.splitext(output)[0] + '.colnames', 'wt') as fout:
            with open(os.path.splitext(tmp_fname)[0] + '.colnames') as fin:
                for i, line in enumerate(fin):
                    if i in desc_ids:
                        replace_dict[i] = len(replace_dict)
                        fout.write(line)

        with open(os.path.splitext(output)[0] + '.rownames', 'wt') as fmol, open(output, 'wt') as ftxt:
            with open(os.path.splitext(tmp_fname)[0] + '.rownames') as fmol_tmp, open(tmp_fname) as ftxt_tmp:
                for line1, line2 in zip(fmol_tmp, ftxt_tmp):
                    desc_str = []
                    for item in line2.strip().split(' '):
                        i, v = item.split(':')
                        i = int(i)
                        if i in replace_dict:
                            desc_str.append(f'{replace_dict[i]}:{v}')
                    if desc_str:
                        fmol.write(line1)
                        ftxt.write(' '.join(desc_str) + '\n')

        if not keep_temp:
            os.remove(tmp_fname)
            os.remove(os.path.splitext(tmp_fname)[0] + '.colnames')
            os.remove(os.path.splitext(tmp_fname)[0] + '.rownames')

    return output
