from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import pandas as pd

# CSV 파일 불러오기
file_path = './_data/dacon/drug/'
df_train = pd.read_csv(file_path + 'train.csv')
df_test = pd.read_csv(file_path + 'test.csv')

# 함수 정의: 고리 제거 및 분자식 계산
def remove_rings_and_get_formula(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    ring_info = mol.GetRingInfo()
    ring_atoms = set([atom_idx for ring in ring_info.AtomRings() for atom_idx in ring])

    emol = Chem.EditableMol(mol)
    atom_indices = list(range(mol.GetNumAtoms()))

    for idx in reversed(atom_indices):
        if idx in ring_atoms:
            emol.RemoveAtom(idx)

    new_mol = emol.GetMol()

    formula = rdMolDescriptors.CalcMolFormula(new_mol)
    return formula if formula else None

# 적용
df_train['No_Ring_Formula'] = df_train['Canonical_Smiles'].apply(remove_rings_and_get_formula)
df_test['No_Ring_Formula'] = df_test['Canonical_Smiles'].apply(remove_rings_and_get_formula)

# 필요한 컬럼만 저장
result_train = df_train[['ID', 'No_Ring_Formula']]
result_test = df_test[['ID', 'No_Ring_Formula']]
result_train.to_csv(file_path + 'data/no_ring_formula_train.csv', index=False)
result_test.to_csv(file_path + 'data/no_ring_formula_test.csv', index=False)

print(result_train.head())
print(result_test.head())
