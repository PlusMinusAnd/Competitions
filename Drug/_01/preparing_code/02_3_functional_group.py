# ======================= 라이브러리 =======================
import pandas as pd
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import os
from collections import Counter


# ======================= 기능기 팩토리 불러오기 =======================
fdef_path = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
factory = ChemicalFeatures.BuildFeatureFactory(fdef_path)


# ======================= 기능기 추출 함수 =======================
def extract_feature_counts(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return Counter({'Invalid_SMILES': 1})
    feats = factory.GetFeaturesForMol(mol)
    counts = Counter([f.GetType() for f in feats])
    return counts


# ======================= 데이터 로드 =======================
train_path = './_data/dacon/drug/train.csv'
test_path = './_data/dacon/drug/test.csv'

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)


# ======================= 기능기 카운팅 적용 =======================
train['functional_group'] = train['Canonical_Smiles'].apply(extract_feature_counts)
test['functional_group'] = test['Canonical_Smiles'].apply(extract_feature_counts)


# ======================= 전체 기능기 종류 추출 =======================
all_features = set()

for counts in pd.concat([train['functional_group'], test['functional_group']]):
    all_features.update(counts.keys())

all_features = sorted(list(all_features))


# ======================= 카운팅 테이블로 변환 =======================
def counts_to_series(counts):
    return pd.Series({f: counts.get(f, 0) for f in all_features})


train_encoded = train['functional_group'].apply(counts_to_series)
test_encoded = test['functional_group'].apply(counts_to_series)


# ======================= 기존 데이터와 병합 =======================
train_result = pd.concat([train.drop(columns=['functional_group']), train_encoded], axis=1)
test_result = pd.concat([test.drop(columns=['functional_group']), test_encoded], axis=1)


# ======================= 저장 =======================
train_result.to_csv('./_data/dacon/drug/data/train_with_functional_groups.csv', index=False)
test_result.to_csv('./_data/dacon/drug/data/test_with_functional_groups.csv', index=False)

print('완료! 카운트형 기능기 파일 저장됨.')
