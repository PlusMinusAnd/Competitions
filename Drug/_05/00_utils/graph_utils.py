# 00_utils/graph_utils.py

import dgl
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops

ATOM_FEATS = ['atomic_num', 'degree', 'formal_charge', 'chiral_tag', 'num_explicit_hs', 'is_aromatic']
BOND_FEATS = ['bond_type', 'is_conjugated', 'is_in_ring']

def atom_features(atom):
    return torch.tensor([
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        int(atom.GetChiralTag()),
        atom.GetTotalNumHs(),
        atom.GetIsAromatic()
    ], dtype=torch.float)

def bond_features(bond):
    return torch.tensor([
        bond.GetBondTypeAsDouble(),
        bond.GetIsConjugated(),
        bond.IsInRing()
    ], dtype=torch.float)

def smiles_to_dgl(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)
    g = dgl.DGLGraph()
    num_atoms = mol.GetNumAtoms()
    g.add_nodes(num_atoms)

    atom_feats = [atom_features(mol.GetAtomWithIdx(i)) for i in range(num_atoms)]
    g.ndata['h'] = torch.stack(atom_feats)

    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        
        # 엣지 피처 제거 → 그냥 연결만 함
        g.add_edges(u, v)
        g.add_edges(v, u)

    return g

