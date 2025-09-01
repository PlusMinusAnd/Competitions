import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import rdFingerprintGenerator
import os
import random

# Seed 고정
r = random.randint(1,1000)
random.seed(r)
np.random.seed(r)

# 🔁 변수 초기화 (재실행 안전)
train_mols, test_mols = [], []
train_descriptor_df, test_descriptor_df = None, None
train_fingerprint_df, test_fingerprint_df = None, None

# 📂 파일 불러오기
train_csv = pd.read_csv('./Drug/train.csv', index_col=0)
test_csv = pd.read_csv('./Drug/test.csv', index_col=0)

# 📦 SMILES → Mol 객체 변환
train_mols = [Chem.MolFromSmiles(smi) for smi in train_csv['Canonical_Smiles']]
test_mols = [Chem.MolFromSmiles(smi) for smi in test_csv['Canonical_Smiles']]

# ❗ SMILES 오류 확인
invalid_train = sum([mol is None for mol in train_mols])
invalid_test = sum([mol is None for mol in test_mols])

# 🧪 디스크립터 계산기
descriptor_names = [desc[0] for desc in Descriptors._descList]
calc = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)

def calc_descriptors(mols):
    return [calc.CalcDescriptors(mol) if mol is not None else [None]*len(descriptor_names) for mol in mols]

# 🧬 최신 Morgan Fingerprint 생성기
morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def get_fingerprint(mol):
    if mol is None:
        return [0]*2048
    fp = morgan_gen.GetFingerprint(mol)
    return [int(b) for b in fp.ToBitString()]

def calc_fingerprints(mols):
    return [get_fingerprint(mol) for mol in mols]

# 🧮 디스크립터 + 핑거프린트 계산
train_descriptor_df = pd.DataFrame(calc_descriptors(train_mols), columns=descriptor_names)
test_descriptor_df = pd.DataFrame(calc_descriptors(test_mols), columns=descriptor_names)

train_fingerprint_df = pd.DataFrame(calc_fingerprints(train_mols), columns=[f'FP_{i}' for i in range(2048)])
test_fingerprint_df = pd.DataFrame(calc_fingerprints(test_mols), columns=[f'FP_{i}' for i in range(2048)])



# 스마일즈 목록 분리
train_smiles = train_csv['Canonical_Smiles']
test_smiles = test_csv['Canonical_Smiles']

# 🧷 인덱스 정렬 후 병합
train = pd.concat([
    train_smiles.reset_index(drop=True).rename('Canonical_Smiles'),
    train_descriptor_df.reset_index(drop=True),
    train_fingerprint_df.reset_index(drop=True),
    train_csv[['Inhibition']].reset_index(drop=True)
], axis=1)

test = pd.concat([
    test_smiles.reset_index(drop=True).rename('Canonical_Smiles'),
    test_descriptor_df.reset_index(drop=True),
    test_fingerprint_df.reset_index(drop=True)
], axis=1)

# 📈 확인
print("✅ Final Train shape:", train.shape)
print("✅ Final Test shape :", test.shape)

# 💡 수치형 / 범주형 분리 기준
descriptor_cols = descriptor_names  # 수치형
fingerprint_cols = [f'FP_{i}' for i in range(2048)]  # 범주형
# RDKit 디스크립터와 Fingerprint 열 목록
descriptor_cols = descriptor_names
fingerprint_cols = [f'FP_{i}' for i in range(2048)]

# 수치형 피처: float이거나 유니크 값이 많음
numeric_cols = []
categorical_cols = []

# 모든 열 검사
for col in descriptor_cols + fingerprint_cols:
    series = train[col]
    nunique = series.nunique()
    dtype = series.dtype

    if pd.api.types.is_float_dtype(dtype):
        numeric_cols.append(col)
    elif nunique <= 2:
        categorical_cols.append(col)
    elif nunique < 20:
        # 원하면 여기서도 범주형으로 넣을 수 있음
        categorical_cols.append(col)
    else:
        numeric_cols.append(col)

# SMILES와 Inhibition은 따로 저장
meta_cols = ['Canonical_Smiles']
if 'Inhibition' in train.columns:
    meta_cols.append('Inhibition')

# 분리해서 붙여두기
train_numeric = pd.concat([train[meta_cols], train[numeric_cols]], axis=1)
train_categorical = pd.concat([train[meta_cols], train[categorical_cols]], axis=1)

test_numeric = pd.concat([test[['Canonical_Smiles']], test[numeric_cols]], axis=1)
test_categorical = pd.concat([test[['Canonical_Smiles']], test[categorical_cols]], axis=1)

train_numeric.to_csv('./Drug/_03/train_nu_siles.csv')
train_categorical.to_csv('./Drug/_03/train_ca_siles.csv')
test_numeric.to_csv('./Drug/_03/test_nu_siles.csv')
test_categorical.to_csv('./Drug/_03/test_ca_siles.csv')

exit()

# print(train_numeric.shape)          (1681, 111)
# print(test_numeric.shape)           (100, 110)
# print(train_categorical.shape)      (1681, 2158)
# print(test_categorical.shape)       (100, 2157)

# 두 데이터프레임에서 NaN 없는 열을 공통적으로 추출
train_nan_cols = train_numeric.columns[train_numeric.isna().any()].tolist()
test_nan_cols = test_numeric.columns[test_numeric.isna().any()].tolist()

# 공통적으로 제거할 열
nan_cols_to_drop = list(set(train_nan_cols + test_nan_cols))

# 동시에 제거
train_numeric = train_numeric.drop(columns=nan_cols_to_drop)
test_numeric = test_numeric.drop(columns=nan_cols_to_drop)

# NaN이 있는 열 확인
def drop_nan_columns(df, df_name):
    nan_cols = df.columns[df.isna().any()].tolist()
    return df.drop(columns=nan_cols)

train_categorical = drop_nan_columns(train_categorical, "train_categorical")
test_categorical = drop_nan_columns(test_categorical, "test_categorical")

# print(train_numeric.shape)        #  (1681, 103)
# print(test_numeric.shape)         #  (100, 102)
# print(train_categorical.shape)    #  (1681, 2158)
# print(test_categorical.shape)     #  (100, 2157)


# feature_importance
print('feature_importance 시작')
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor
import numpy as np

# 🎯 수치형
x_nu = train_numeric.drop(['Canonical_Smiles', 'Inhibition'], axis=1)
y_nu = train_numeric['Inhibition']
test_nu = test_numeric.drop(['Canonical_Smiles'], axis=1)

x1_train, x1_test, y1_train, y1_test = train_test_split(
    x_nu, y_nu, random_state=r
)

model1 = XGBRegressor(random_state=r) 
model1.fit(x1_train, y1_train)
print("==========", model1.__class__.__name__, "==========")
print('R2 :', model1.score(x1_test, y1_test))

# 중요도 상위 50개 선택
top50_nu_idx = np.argsort(model1.feature_importances_)[::-1][:50]
top50_nu_cols = x_nu.columns[top50_nu_idx]

# 🎯 범주형
x_ca = train_categorical.drop(['Canonical_Smiles', 'Inhibition'], axis=1)
y_ca = train_categorical['Inhibition']
test_ca = test_categorical.drop(['Canonical_Smiles'], axis=1)

from sklearn.preprocessing import LabelEncoder
y_ca_rint = np.rint(y_ca).astype(int)
le = LabelEncoder()
y_ca_rint_enc = le.fit_transform(y_ca_rint)

x2_train, x2_test, y2_train, y2_test = train_test_split(
    x_ca, y_ca_rint_enc, random_state=r,
)

model2 = XGBClassifier(random_state=r) 
model2.fit(x2_train, y2_train)
print("==========", model2.__class__.__name__, "==========")
print('ACC :', model2.score(x2_test, y2_test))

# 중요도 상위 1000개 선택
top1000_ca_idx = np.argsort(model2.feature_importances_)[::-1][:1000]
top1000_ca_cols = x_ca.columns[top1000_ca_idx]

# ✅ 정렬 및 열 제한 후 병합
train_nu_sorted = pd.concat([
    train_numeric[['Canonical_Smiles', 'Inhibition']].reset_index(drop=True),
    x_nu[top50_nu_cols].reset_index(drop=True)
], axis=1)

test_nu_sorted = pd.concat([
    test_numeric[['Canonical_Smiles']].reset_index(drop=True),
    test_nu[top50_nu_cols].reset_index(drop=True)
], axis=1)

train_ca_sorted = pd.concat([
    train_categorical[['Canonical_Smiles', 'Inhibition']].reset_index(drop=True),
    x_ca[top1000_ca_cols].reset_index(drop=True)
], axis=1)

test_ca_sorted = pd.concat([
    test_categorical[['Canonical_Smiles']].reset_index(drop=True),
    test_ca[top1000_ca_cols].reset_index(drop=True)
], axis=1)

# ✅ 확인
# print("📐 Shapes after top-k filtering")
# print("  train_numeric:", train_nu_sorted.shape)      #  train_numeric: (1681, 52)
# print("  test_numeric :", test_nu_sorted.shape)       #  test_numeric : (100, 51)
# print("  train_categorical:", train_ca_sorted.shape)  #  train_categorical: (1681, 1002)
# print("  test_categorical :", test_ca_sorted.shape)   #  test_categorical : (100, 1001)

# numeric 데이터 PCA
numeric_x = train_nu_sorted.drop(
    ['Canonical_Smiles', 'Inhibition'], axis=1
)
numeric_test = test_nu_sorted.drop(
    ['Canonical_Smiles'], axis=1
)
numeric_y = train_nu_sorted['Inhibition']

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
numeric_x_scaled = ss.fit_transform(numeric_x)
numeric_test_scaled = ss.transform(numeric_test)

pca = PCA(n_components=35)
numeric_x_pca = pca.fit_transform(numeric_x_scaled)
numeric_test_pca = pca.transform(numeric_test_scaled)
# evr = np.cumsum(pca.explained_variance_ratio_)
# # print(evr)
# num = [0.999, 0.99, 0.95, 0.90]
# for i in num :
#     threshold = i            
#     compo_1 = np.argmax(evr>= threshold) +1 
#     print(f'{i} :',compo_1) 
# 0.999 : 44
# 0.99  : 35
# 0.95  : 27
# 0.9   : 21

pca_train = pd.concat([
    train_numeric[['Canonical_Smiles']].reset_index(drop=True),
    pd.DataFrame(numeric_x_pca, columns=[f'PC_{i+1}' for i in range(numeric_x_pca.shape[1])]),
    train_numeric[['Inhibition']].reset_index(drop=True)
], axis=1)

pca_test = pd.concat([
    test_numeric[['Canonical_Smiles']].reset_index(drop=True),
    pd.DataFrame(numeric_test_pca, columns=[f'PC_{i+1}' for i in range(numeric_test_pca.shape[1])])
], axis=1)

# Categorical 데이터 LDA
categorical_x = train_categorical.drop(
    ['Canonical_Smiles', 'Inhibition'], axis=1
)
categorical_test = test_categorical.drop(
    ['Canonical_Smiles'], axis=1
)
categorical_y = np.rint(train_categorical['Inhibition']).astype(int)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=97)
categorical_x_lda = lda.fit_transform(categorical_x, categorical_y)
categorical_test_lda = lda.transform(categorical_test)

# evr = np.cumsum(lda.explained_variance_ratio_)
# num = [0.999, 0.99, 0.95, 0.90]
# for i in num :
#     threshold = i            
#     compo_1 = np.argmax(evr>= threshold) +1 
#     print(compo_1) 
# 0.999     99
# 0.99      97
# 0.95      87
# 0.90      77

lda_train = pd.concat([
    train_categorical[['Canonical_Smiles']].reset_index(drop=True),
    pd.DataFrame(categorical_x_lda, columns=[f'PC_{i+1}' for i in range(categorical_x_lda.shape[1])]),
    train_categorical[['Inhibition']].reset_index(drop=True)
], axis=1)

lda_test = pd.concat([
    test_categorical[['Canonical_Smiles']].reset_index(drop=True),
    pd.DataFrame(categorical_test_lda, columns=[f'PC_{i+1}' for i in range(categorical_test_lda.shape[1])])
], axis=1)

# print(lda_train.shape)     #  (1681, 99)
# print(pca_train.shape)     #  (1681, 37)
# print(lda_test.shape)      #  (100, 98)
# print(pca_test.shape)      #  (100, 36)

lda_train.to_csv('./Drug/_03/1_data/train_category.csv')
lda_test.to_csv('./Drug/_03/1_data/test_category.csv')
pca_train.to_csv('./Drug/_03/1_data/train_numeric.csv')
pca_test.to_csv('./Drug/_03/1_data/test_numeric.csv')

print(f'저장완료! 데이터 제작 랜덤값은 [{r}] 이야!')


exit()
lda_x = lda_train.drop(['Canonical_Smiles','Inhibition'], axis=1)
lda_y = lda_train['Inhibition']
pca_x = pca_train.drop(['Canonical_Smiles','Inhibition'], axis=1)
pca_y = pca_train['Inhibition']

lda_y = np.rint(lda_y).astype(int)

lda_x_train, lda_x_test, lda_y_train, lda_y_test = train_test_split(
    lda_x, lda_y, random_state=r
)
pca_x_train, pca_x_test, pca_y_train, pca_y_test = train_test_split(
    pca_x, pca_y, random_state=r
)


