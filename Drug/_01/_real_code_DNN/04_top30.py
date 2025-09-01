import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# 경로
train_path = './Drug/_engineered_data_DNN/train_filled_dnn.csv'
test_path = './Drug/_engineered_data_DNN/test_with_rdkit_features.csv'
save_path = './Drug/_engineered_data_DNN/'

# 데이터 로드
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# 수치형 + 범주형 분리
feature_cols = train.columns.difference(['Inhibition'])  # ID, SMILES도 포함됨
X_all = pd.concat([train[feature_cols], test[feature_cols]], axis=0)

numeric_cols = X_all.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X_all.select_dtypes(exclude=[np.number]).columns.difference(['ID', 'Canonical_Smiles']).tolist()

# 전처리 파이프라인
numeric_transformer = SimpleImputer(strategy='mean')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# train/test 나누기
X_train_raw = train[feature_cols]
X_test_raw = test[feature_cols]
y_train = train['Inhibition']

# 전처리 후 피처 추출
X_train = preprocessor.fit_transform(X_train_raw)
X_test = preprocessor.transform(X_test_raw)

# 1. OneHotEncoder 별도로 fit
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_cat = train[categorical_cols]
ohe.fit(X_cat)

# 2. 피처 이름 생성
cat_feature_names = ohe.get_feature_names_out(categorical_cols)
all_feature_names = numeric_cols + list(cat_feature_names)

# 중요도 학습
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
importances = pd.Series(model.feature_importances_, index=all_feature_names)
importances_sorted = importances.sort_values(ascending=False)

# 상위 피처 후보 100개 중 상관계수 0.9 이상 제거
top_candidates = importances_sorted.head(100).index.tolist()
X_all_encoded = pd.DataFrame(np.vstack([X_train, X_test]), columns=all_feature_names)
X_top = X_all_encoded[top_candidates]
corr_matrix = X_top.corr().abs()

threshold = 0.9
selected_features = []
for feature in top_candidates:
    if not selected_features or all(corr_matrix.loc[feature, selected_features] < threshold):
        selected_features.append(feature)
    if len(selected_features) == 30:
        break

# 최종 선택된 피처만 추출
X_train_selected = pd.DataFrame(X_train, columns=all_feature_names)[selected_features]
X_test_selected = pd.DataFrame(X_test, columns=all_feature_names)[selected_features]

# ID, SMILES 포함하여 저장
train_final = pd.concat([train[['ID', 'Canonical_Smiles']].reset_index(drop=True),
                         X_train_selected.reset_index(drop=True),
                         y_train.reset_index(drop=True)], axis=1)

test_final = pd.concat([test[['ID', 'Canonical_Smiles']].reset_index(drop=True),
                        X_test_selected.reset_index(drop=True)], axis=1)

train_final.to_csv(f"{save_path}train_final.csv", index=False)
test_final.to_csv(f"{save_path}test_final.csv", index=False)
