'''
import pandas as pd
import numpy as np

# 데이터 로드
df = pd.read_csv('./Drug/data/features_rdkit.csv')

# 독립 변수만 추출
features = df.drop(['index', 'SMILES', 'Inhibition'], axis=1)

# 상관 행렬 계산
corr_matrix = features.corr().abs()  # 절대값 상관계수

# 자기 자신 제외 (상삼각 행렬만 사용)
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# 상관계수가 threshold 이상인 피처 찾기
threshold = 0.85  # 상관계수 기준

to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]

print(f"제거할 피처 수: {len(to_drop)}")
print(f"제거할 피처 목록:\n{to_drop}")

# 피처 제거
reduced_features = features.drop(columns=to_drop)

# 최종 데이터프레임 생성 (index, SMILES, Inhibition 포함)
final_df = pd.concat([df[['index', 'SMILES']], reduced_features, df['Inhibition']], axis=1)

# 저장
final_df.to_csv('./Drug/data/features_rdkit_reduced.csv', index=False)

print("저장 완료: ./Drug/data/features_rdkit_reduced.csv")


import pandas as pd

# 데이터 로드
df = pd.read_csv('./Drug/data/features_rdkit.csv')

# 이전에 구한 feature importance 데이터 로드 또는 직접 리스트 입력
# 여기서는 리스트로 직접 지정합니다.
top20_features = [
    'MolLogP', 'AvgIpc', 'VSA_EState4', 'MolMR', 'PEOE_VSA8',
    'BCUT2D_MRLOW', 'PEOE_VSA9', 'EState_VSA6', 'qed', 'VSA_EState3',
    'VSA_EState7', 'MinAbsPartialCharge', 'BCUT2D_LOGPLOW', 'BCUT2D_MRHI',
    'VSA_EState2', 'VSA_EState5', 'SMR_VSA7', 'BCUT2D_LOGPHI', 'PEOE_VSA7', 'BalabanJ'
]

# 메타 데이터 (index, SMILES, Inhibition) + 상위 20개 피처만 선택
selected_columns = ['index', 'SMILES'] + top20_features + ['Inhibition']
reduced_df = df[selected_columns]

# 저장
reduced_df.to_csv('./Drug/data/features_rdkit_top20.csv', index=False)

print("상위 20개 피처만 포함된 데이터 저장 완료: ./Drug/data/features_rdkit_top20.csv")'''


import pandas as pd

# 데이터 로드
df = pd.read_csv('./Drug/data/features_rdkit_top20.csv')

# 결측치 개수 확인
missing = df.isnull().sum()

print("NaN 결측치 개수:\n", missing)

# '빈 문자열' 형태의 결측치가 있는지 확인
empty_string = (df == '').sum()

print("\n빈 문자열 개수:\n", empty_string)

# '공백 문자열' 형태 확인
space_string = (df == ' ').sum()

print("\n공백 문자열 개수:\n", space_string)
