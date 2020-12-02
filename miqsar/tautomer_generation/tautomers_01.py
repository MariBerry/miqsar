from rdkit import Chem
from rdkit.Chem.MolStandardize import Standardizer, tautomer
from rdkit.Chem.rdmolfiles import MolToSmiles
import argparse
import os


def enumerate_tautomers_smiles(smiles, stereo=False):
    """Return a set of tautomers as SMILES strings, given a SMILES string.

    :param smiles: A SMILES string.
    :param stereo: if true  - returns  tautomers with the same stereo as input, otherwise drops stereo info .
    :returns: A set containing SMILES strings for every possible tautomer.
    :rtype: set of strings.
    """
    # Skip sanitize as standardize does this anyway
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    mol = Standardizer().standardize(mol)
    tmp = Chem.MolToSmiles(mol, isomericSmiles=stereo) # smiles of original input with/wihout stereo
    tautomers = tautomer.TautomerEnumerator().enumerate(mol)
    res = {Chem.MolToSmiles(m, isomericSmiles=stereo) for m in tautomers}
    res.remove(tmp)
    return tmp,res

def gen_tautomers(data_file,  path, max_n_tau=None):
    out_fname = os.path.join(path, os.path.basename(data_file).split('.')[0] + "_tau.smi")
    #load dataet
    with open(data_file) as f:
        smiles=[]
        names=[]
        act = []
        for i in f:
            tmp = i.strip().split(",")
            smiles.append(tmp[0])
            names.append(tmp[1])
            act.append(tmp[2])


    #generate tautomers
    with open(out_fname, "wt") as out:
        for i, j, a in zip(smiles, names, act):
            orig,smile_t = enumerate_tautomers_smiles(i, stereo = False)
            smile_t = list(smile_t)
            # trick to put input smiles to the 1st place in smile_t:  put it in start and  remove its dupl
            # form the list
            smile_t.insert(0, orig)
            for n, tautomer in enumerate(smile_t):
                k = j + "_{tautomer_id}"
                out.write(str(tautomer) + "," + k.format(tautomer_id=n) + "," + str(a) + ","+ j + "\n")
                if max_n_tau is not None and (n + 1) >= max_n_tau: break
    return out_fname

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enumerate tautomers.")
    parser.add_argument("-i", "--input_file", metavar=" ", required=True, help="Input .smi file .")
    parser.add_argument("-p", "--path", metavar=" ", required=True, help="Output path.")
    args = parser.parse_args()

    gen_tautomers(args.input_file, args.path)
