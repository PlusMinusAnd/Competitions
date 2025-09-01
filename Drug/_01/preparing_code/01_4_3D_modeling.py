from rdkit import Chem
from rdkit.Chem import AllChem

mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
AllChem.EmbedMolecule(mol, AllChem.ETKDG())
positions = mol.GetConformer().GetPositions()

print(positions.shape)
print(positions)

# (9, 3)
# [[-0.95345518  0.04780422  0.04249938]
#  [ 0.48791869 -0.32153932 -0.18904974]
#  [ 1.27697753  0.32478368  0.73767324]
#  [-1.35701261  0.7339752  -0.73322256]
#  [-1.59332094 -0.86256983  0.02175948]
#  [-1.07103128  0.59643936  1.01116276]
#  [ 0.78740605 -0.14060868 -1.23577432]
#  [ 0.59002788 -1.42610966 -0.04403049]
#  [ 1.83248987  1.04782504  0.38898225]]

# from rdkit import Chem
# from rdkit.Chem import AllChem

# # 분자 생성 및 3D 좌표 생성
# smiles = 'CCO'
# mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
# AllChem.EmbedMolecule(mol, AllChem.ETKDG())
# AllChem.UFFOptimizeMolecule(mol)

# # SDF 파일로 저장
# w = Chem.SDWriter('./molecule_3D.sdf')
# w.write(mol)
# w.close()

# print("SDF 파일 저장 완료")


# import pandas as pd
# from rdkit import Chem
# from rdkit.Chem import AllChem

# # ===============================
# # CSV 파일 불러오기
# # ===============================
# path = './_data/dacon/drug/'  # 파일 경로 설정
# file = 'train.csv'

# df = pd.read_csv(path + file)

# # SMILES가 있는 열 (2번째 열)
# smiles_list = df.iloc[:, 1]  # 0부터 시작하므로 2번째 열은 인덱스 1

# # 이름 설정용 인덱스 또는 다른 열 (원하는 경우)
# id_list = df.index.tolist()  # 파일명이 0,1,2... 형태로 저장됨

# # ===============================
# # 저장 폴더 설정
# # ===============================
# output_path = './_data/dacon/drug/data/'

# import os
# os.makedirs(output_path, exist_ok=True)

# # ===============================
# # 변환 및 저장
# # ===============================
# for idx, smiles in zip(id_list, smiles_list):
#     mol = Chem.MolFromSmiles(smiles)
#     if mol is None:
#         print(f"[Warning] Invalid SMILES at index {idx}: {smiles}")
#         continue

#     # 3D 구조 생성
#     mol = Chem.AddHs(mol)
#     AllChem.EmbedMolecule(mol, AllChem.ETKDG())
#     AllChem.UFFOptimizeMolecule(mol)

#     # SDF로 저장
#     writer = Chem.SDWriter(f'{output_path}molecule_{idx}.sdf')
#     writer.write(mol)
#     writer.close()

#     print(f"[Saved] molecule_{idx}.sdf")

# print("✅ 모든 파일 저장 완료")



# import pandas as pd
# import numpy as np
# import os
# from rdkit import Chem
# from rdkit.Chem import AllChem, Draw
# from PIL import Image

# # ===============================
# # 파일 및 폴더 설정
# # ===============================
# path = './_data/dacon/drug/'
# file = 'train.csv'
# save_path = './_data/dacon/drug/data/'

# os.makedirs(save_path, exist_ok=True)

# # ===============================
# # 데이터 불러오기
# # ===============================
# df = pd.read_csv(path + file)
# smiles_list = df.iloc[:, 1]  # 2번째 열 (인덱스 1)
# id_list = df.index.tolist()

# # ===============================
# # 이미지 파라미터
# # ===============================
# img_size = (224, 224)  # 이미지 사이즈 (높이, 너비)
# channel = 3  # RGB

# # ===============================
# # 변환 및 저장
# # ===============================
# image_list = []

# for idx, smiles in zip(id_list, smiles_list):
#     mol = Chem.MolFromSmiles(smiles)
#     if mol is None:
#         print(f"[Warning] Invalid SMILES at index {idx}: {smiles}")
#         continue

#     # 3D 구조 생성
#     mol = Chem.AddHs(mol)
#     AllChem.EmbedMolecule(mol, AllChem.ETKDG())
#     AllChem.UFFOptimizeMolecule(mol)

#     # 이미지 생성 (2D 그림)
#     img = Draw.MolToImage(mol, size=img_size)

#     # PIL 이미지를 넘파이 배열로 변환
#     img_array = np.array(img)

#     # (224, 224, 3) 형태로 변환됨
#     image_list.append(img_array)

#     print(f"[Saved] image for index {idx}")

# # ===============================
# # 넘파이 배열로 변환
# # ===============================
# image_array = np.stack(image_list, axis=0)  # (샘플 수, 높이, 너비, 채널)

# print(f"✅ 최종 배열 형태: {image_array.shape}")

# # ===============================
# # 저장
# # ===============================
# np.save(save_path + 'molecule_images.npy', image_array)
# print("✅ 저장 완료: molecule_images.npy")


import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import os

# ===============================
# 파일 및 폴더 설정
# ===============================
path = './_data/dacon/drug/'
file = 'train.csv'
save_path = './_data/dacon/drug/data/'

os.makedirs(save_path, exist_ok=True)

# ===============================
# 데이터 불러오기
# ===============================
df = pd.read_csv(path + file)
smiles_list = df.iloc[:, 1]  # 2번째 열 (인덱스 1)
id_list = df.index.tolist()

# ===============================
# Voxel 파라미터
# ===============================
grid_size = 32  # (32x32x32) 그리드
voxel_resolution = 2.0  # 옹스트롬당 그리드 간격 (높을수록 촘촘)

# 원자 종류 매핑 (C, O, N, H)
atom_types = ['C', 'O', 'N', 'H', 'S', 'P', 'Cl', 'F', 'Br', 'I']
num_channels = len(atom_types)

atom_type_to_channel = {atom: idx for idx, atom in enumerate(atom_types)}

# ===============================
# 변환 함수
# ===============================
def mol_to_voxel(mol):
    try:
        conf = mol.GetConformer()
        positions = conf.GetPositions()

        voxel = np.zeros((grid_size, grid_size, grid_size, num_channels), dtype=np.float32)

        # 중심 좌표 정규화
        center = positions.mean(axis=0)
        positions -= center  # 중심 이동

        # 스케일링: 그리드 중앙에 위치
        scale = (grid_size / 2) - 1
        scaled_positions = positions / voxel_resolution + scale

        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            symbol = atom.GetSymbol()

            if symbol not in atom_type_to_channel:
                continue  # 무시

            channel = atom_type_to_channel[symbol]

            x, y, z = scaled_positions[idx]
            x, y, z = int(round(x)), int(round(y)), int(round(z))

            if 0 <= x < grid_size and 0 <= y < grid_size and 0 <= z < grid_size:
                voxel[z, y, x, channel] = 1.0  # 위치에 채널값 기록

        return voxel

    except:
        return None

# ===============================
# 변환 및 저장
# ===============================
voxel_list = []

for idx, smiles in zip(id_list, smiles_list):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"[Warning] Invalid SMILES at index {idx}: {smiles}")
        continue

    # 3D 구조 생성
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.UFFOptimizeMolecule(mol)

    voxel = mol_to_voxel(mol)

    if voxel is None:
        print(f"[Warning] Failed voxelize at index {idx}")
        continue

    voxel_list.append(voxel)

    print(f"[Saved] voxel for index {idx}")

# ===============================
# 넘파이 배열로 변환
# ===============================
voxel_array = np.stack(voxel_list, axis=0)  # (샘플 수, 깊이, 높이, 너비, 채널)

print(f"✅ 최종 배열 형태: {voxel_array.shape}")

# ===============================
# 저장
# ===============================
np.save(save_path + 'molecule_3D.npy', voxel_array)
print("✅ 저장 완료: molecule_3D.npy")
