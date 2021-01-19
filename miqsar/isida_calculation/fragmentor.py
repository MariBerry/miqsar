import os
import shutil
import numpy as np
import pandas as pd
from CIMtools.preprocessing import Fragmentor
from CGRtools.files.SDFrw import SDFRead

class BagGenerator(Fragmentor):

    def __init__(self, min_length=3, max_length=3, fragment_type=1, marked_atom=2):
        super().__init__(version='2017.x', min_length=min_length, max_length=max_length,
                         fragment_type=fragment_type, useformalcharge=True)

        self.marked_atom = marked_atom
        self.frag_names = None

    def get_bag_df(self, mol):
        dsc = self.transform([mol])
        df = pd.DataFrame(dsc[0], columns=self.frag_names)
        df = df.loc[:, (df != 0).any(axis=0)]
        return df

    def select_marked_frags(self, bags):
        idxes = [i for i, f in enumerate(self.frag_names) if '*' in f]
        self.frag_names = [self.frag_names[i] for i in idxes]
        bags = bags[:, :, idxes]
        return bags

    def mark_atoms(self, mols):
        marked_mols = []
        for m1 in mols:
            marked_atoms = []
            for i, atom in enumerate(m1.atoms(), start=1):
                m2 = m1.copy()
                m2.atom(i).charge = m2.atom(i).charge + 1
                cgr = m1.compose(m2)
                marked_atoms.append(cgr)
            marked_mols.append(marked_atoms)
        return marked_mols

    def fit(self, mols):
        mols = self.mark_atoms(mols)
        super().fit([j for i in mols for j in i])
        self.frag_names = self.get_feature_names()
        self.frag_names = [i.replace('c+1', '*') for i in self.frag_names]
        return self

    def transform(self, mols):
        mols = self.mark_atoms(mols)
        bag_size = max([len(m) for m in mols])

        bags = []
        for n, mol in enumerate(mols):
            bag = super().transform(mol).values
            if len(bag) < bag_size:
                padding = np.ones((bag_size - bag.shape[0], bag.shape[1]))
                bag = np.vstack((bag, padding))
            bags.append(bag)
        bags = np.array(bags)

        if self.marked_atom == 2:
            bags = self.select_marked_frags(bags)

        return bags


def bags_from_sdf(input_sdf_file=None, fragmentor_path='./',
                  min_length=1, max_length=3, fragment_type=1, marked_atom=2, label_name=None, ids_name=None):
    from os import environ
    environ['PATH'] += ':{}'.format(fragmentor_path)

    # read mols and init fragmentor
    mols = SDFRead(input_sdf_file, remap=False).read()

    labels = np.array([float(mol.meta[label_name]) for mol in mols])
    if ids_name:
        ids = np.array([mol.meta.get(ids_name, '') for mol in mols])
    else:
        ids = None

    frag = BagGenerator(min_length=min_length, max_length=max_length, fragment_type=fragment_type,
                        marked_atom=marked_atom)
    frag.fit(mols)
    bags = frag.transform(mols)
    header = frag.frag_names

    # delete frg files
    del frag
    frg_files = [i for i in os.listdir() if i.startswith('frg')]
    for file in frg_files:
        if os.path.isfile(file):
            os.remove(file)
        else:
            shutil.rmtree(file)
    return bags, labels, ids, header