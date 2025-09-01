# 00_utils/rdkit_features.py

from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
import pandas as pd
from tqdm import tqdm

def get_rdkit_descriptors(smiles_list):
    descriptor_names = [desc[0] for desc in Descriptors._descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)

    feature_list = []
    valid_idx = []

    for idx, smi in enumerate(tqdm(smiles_list)):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            descs = calculator.CalcDescriptors(mol)
            feature_list.append(descs)
            valid_idx.append(idx)
        else:
            feature_list.append([None] * len(descriptor_names))

    df = pd.DataFrame(feature_list, columns=descriptor_names)
    return df

# 00_utils/rdkit_features.py (기존 파일에 이어서 추가)

from rdkit.Chem import AllChem
import numpy as np

def get_morgan_fingerprint(smiles_list, radius=2, n_bits=2048):
    from rdkit import Chem
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            arr = np.zeros((n_bits,))
            AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr)
        else:
            fps.append(np.zeros((n_bits,)))
    return np.array(fps)
