import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
import os

# CSV 파일 경로
csv_path = './_data/dacon/drug/train.csv'
df = pd.read_csv(csv_path)

# 이미지 저장 디렉토리
save_dir = './_data/dacon/drug/image/'
os.makedirs(save_dir, exist_ok=True)

# SMILES 컬럼 (2열)
smiles_list = df.iloc[:, 1]  # 또는 df["Canonical_Smiles"]

# 구조식 이미지 생성 및 저장
for idx, smiles in enumerate(smiles_list):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        img = Draw.MolToImage(mol, size=(300, 300))
        img.save(os.path.join(save_dir, f'molecule_{idx:04d}.png'))
    else:
        print(f"[{idx}] Invalid SMILES: {smiles}")
