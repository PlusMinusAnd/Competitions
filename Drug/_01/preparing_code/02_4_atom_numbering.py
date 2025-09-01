import pandas as pd
from rdkit import Chem

# ===============================
# 데이터 불러오기
# ===============================
train = pd.read_csv('./_data/dacon/drug/train.csv')
test = pd.read_csv('./_data/dacon/drug/test.csv')

# ===============================
# SMILES 열 이름 가져오기 (2번째 열)
# ===============================
smiles_col_train = train.columns[1]
smiles_col_test = test.columns[1]

# ===============================
# SMILES → 원자 기호 리스트 추출 함수
# ===============================
def extract_atom_sequence(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    return [atom.GetSymbol() for atom in mol.GetAtoms()]

# ===============================
# 데이터에 적용 (원자 기호 시퀀스)
# ===============================
train['atom_sequence'] = train[smiles_col_train].apply(extract_atom_sequence)
test['atom_sequence'] = test[smiles_col_test].apply(extract_atom_sequence)

# ===============================
# 전체 원자 목록 만들기
# ===============================
all_atoms = set()
for seq in train['atom_sequence'].tolist() + test['atom_sequence'].tolist():
    all_atoms.update(seq)

all_atoms = sorted(list(all_atoms))  # 정렬
print('전체 원소 목록:', all_atoms)

# ===============================
# 원자 기호 → 숫자 매핑
# ===============================
atom_to_index = {atom: idx for idx, atom in enumerate(all_atoms)}
print('원소 인덱스 매핑:', atom_to_index)

# ===============================
# 원자 기호 시퀀스를 숫자 시퀀스로 변환
# ===============================
def atom_sequence_to_numeric(seq):
    return [atom_to_index[a] for a in seq]

train['atom_index_sequence'] = train['atom_sequence'].apply(atom_sequence_to_numeric)
test['atom_index_sequence'] = test['atom_sequence'].apply(atom_sequence_to_numeric)

# ===============================
# 결과 저장
# ===============================
train.to_csv('./_data/dacon/drug/data/train_with_atom_sequence.csv', index=False)
test.to_csv('./_data/dacon/drug/data/test_with_atom_sequence.csv', index=False)

print('저장 완료!')
