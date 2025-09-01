from rdkit import Chem
from rdkit.Chem import Draw

# 입력 SMILES
smiles = 'CC(=O)C1=C(NC2CCCc3ccccc23)Nc4c(cccc4c5ccccc5)C1=O'
mol = Chem.MolFromSmiles(smiles)

# 고리에 속하지 않은 원자 인덱스 추출
non_ring_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if not atom.IsInRing()]

# 새로운 분자 만들기 (곁가지만)
editable_mol = Chem.RWMol()
atom_map = {}  # 원래 인덱스 → 새 인덱스

for idx in non_ring_atoms:
    atom = mol.GetAtomWithIdx(idx)
    new_idx = editable_mol.AddAtom(Chem.Atom(atom.GetSymbol()))
    atom_map[idx] = new_idx

# 곁가지 원자 간의 결합 추가
for bond in mol.GetBonds():
    a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    if a1 in atom_map and a2 in atom_map:
        editable_mol.AddBond(atom_map[a1], atom_map[a2], bond.GetBondType())

# 최종 분자 생성
non_ring_mol = editable_mol.GetMol()

# 이미지로 저장
img = Draw.MolToImage(non_ring_mol, size=(300, 300))
img.save("non_ring_structure.png")

print("✔ 고리 제외한 구조 이미지를 'non_ring_structure.png'로 저장했습니다.")
