import numpy as np
import pandas as pd
import warnings, os, json

warnings.filterwarnings('ignore')
# seed 상태 저장용 파일
seed_file = "./Stress/seed_count/01.json"

# 파일이 없으면 처음 생성
if not os.path.exists(seed_file):
    seed_state = {"seed": 1}
else:
    with open(seed_file, "r") as f:
        seed_state = json.load(f)

# 현재 seed 값 사용
SEED = seed_state["seed"]
print(f"[Current Run SEED]: {SEED}")

# 다음 실행을 위해 seed 값 1 증가
seed_state["seed"] += 1
with open(seed_file, "w") as f:
    json.dump(seed_state, f)

data_path = './Stress/'

train = pd.read_csv(data_path + 'train.csv', index_col=0)
test = pd.read_csv(data_path + 'test.csv', index_col=0)
submission = pd.read_csv(data_path + 'sample_submission.csv')

print("[1] 데이터 로딩 및 초기 전처리 완료")

# print(train.columns)
# Index(['gender', 'age', 'height', 'weight', 'cholesterol',
#        'systolic_blood_pressure', 'diastolic_blood_pressure', 'glucose',
#        'bone_density', 'activity', 'smoke_status', 'medical_history',
#        'family_medical_history', 'sleep_pattern', 'edu_level', 'mean_working',
#        'stress_score'],
#       dtype='object')

def classify_bp(sys, dia):
    level_sys = 0
    if sys >= 180:
        level_sys = 4
    elif sys >= 160:
        level_sys = 3
    elif sys >= 140:
        level_sys = 2
    elif sys >= 120:
        level_sys = 1
    else:
        level_sys = 0

    level_dia = 0
    if dia >= 120:
        level_dia = 4
    elif dia >= 100:
        level_dia = 3
    elif dia >= 90:
        level_dia = 2
    elif dia >= 80:
        level_dia = 1
    else:
        level_dia = 0

    return max(level_sys, level_dia)

train['bp_category'] = train.apply(lambda row: classify_bp(row['systolic_blood_pressure'], row['diastolic_blood_pressure']), axis=1)
test['bp_category'] = test.apply(lambda row: classify_bp(row['systolic_blood_pressure'], row['diastolic_blood_pressure']), axis=1)

train = train.drop(['systolic_blood_pressure', 'diastolic_blood_pressure'], axis=1)
test = test.drop(['systolic_blood_pressure', 'diastolic_blood_pressure'], axis=1)

print("[2] 혈압 카테고리 분류 및 불필요 컬럼 제거 완료")

gender_map = {'M':0, 'F':1}
smoke_map = {'non-smoker':0,'ex-smoker':1, 'current-smoker':2}
activity_map = {'moderate':0, 'intense':1, 'light':2}
med_map = {'heart disease':3 , 'diabetes':1 ,'high blood pressure':2}
sleep_map = {'normal':0, 'oversleeping':1, 'sleep difficulty':2}
edu_map = {'bachelors degree':0, 'graduate degree':1, 'high school diploma':2}

train['gender'] = train['gender'].map(gender_map)
test['gender'] = test['gender'].map(gender_map)

train['smoke_status'] = train['smoke_status'].map(smoke_map)
test['smoke_status'] = test['smoke_status'].map(smoke_map)

train['activity'] = train['activity'].map(activity_map)
test['activity'] = test['activity'].map(activity_map)

train['medical_history'] = train['medical_history'].map(med_map).fillna(0)
test['medical_history'] = test['medical_history'].map(med_map).fillna(0)

train['family_medical_history'] = train['family_medical_history'].map(med_map).fillna(0)
test['family_medical_history'] = test['family_medical_history'].map(med_map).fillna(0)

train['sleep_pattern'] = train['sleep_pattern'].map(sleep_map)
test['sleep_pattern'] = test['sleep_pattern'].map(sleep_map)

train['edu_level'] = train['edu_level'].map(edu_map).fillna(3)
test['edu_level'] = test['edu_level'].map(edu_map).fillna(3)

oh_col = ['edu_level', 'smoke_status', 'activity', 'medical_history', 'family_medical_history', 'bp_category']

print("[3] 범주형 데이터 인코딩 완료")

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# 1. 결측치 있는 행 / 없는 행 분리
train_known = train[train['mean_working'].notna()]
train_unknown = train[train['mean_working'].isna()]

print("[4] mean_working 결측치 보간 완료 (train/test)")

# 2. 예측에 사용할 feature들
features = [col for col in train.columns if col not in ['mean_working', 'stress_score']]

# 3. 학습 데이터 준비
X = train_known[features]
y = train_known['mean_working']

from lightgbm import LGBMRegressor
# 4. 모델 학습
model = RandomForestRegressor(random_state=SEED, n_estimators=1000)
model.fit(X, y)

# 5. 결측치 예측
train.loc[train['mean_working'].isna(), 'mean_working'] = model.predict(train_unknown[features])

# 6. test 데이터도 동일하게 결측치 채움
test_known = test[test['mean_working'].notna()]
test_unknown = test[test['mean_working'].isna()]

test.loc[test['mean_working'].isna(), 'mean_working'] = model.predict(test_unknown[features])

# 7. 확인
# print("남은 train 결측치:\n", train.isna().sum())
# print("남은 test 결측치:\n", test.isna().sum())

from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

import numpy as np
import pandas as pd


# 1. 데이터 분리
X = train.drop(columns=['stress_score'])
y = train['stress_score']
X_test = test.copy()
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import StackingRegressor

from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge

estimators = [
    ('lgbm', LGBMRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, random_state=SEED, verbose=-1)),
    ('xgb', XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, random_state=SEED, verbosity=0)),
    ('cat', CatBoostRegressor(n_estimators=1000, learning_rate=0.05, depth=6, random_state=SEED, verbose=0))
]

stack_model = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge(),
    passthrough=True,
    cv=5,
    n_jobs=-1
)


kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

print("[5] 스태킹 모델 KFold 학습 시작")

for fold,(train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"    > Fold {fold+1}/5 시작")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    stack_model.fit(X_train, y_train)
    oof_preds[val_idx] = stack_model.predict(X_val)
    test_preds += stack_model.predict(X_test) / kf.n_splits
    print(f"    > Fold {fold+1}/5 완료")

# 6. 평가
mae = mean_absolute_error(y, oof_preds)
print("[6] 최종 OOF MAE:", mae)

# 파일명 포맷팅
mae_str = f"{mae:.5f}".replace('.', '_')  # 소수점 → 언더바
filename = f"energy_{SEED}_{mae_str}.csv"

# 저장
submission['stress_score'] = test_preds
submission.to_csv(os.path.join(data_path, 'submission/', filename), index=False)

print(f"[7] 결과 저장 완료 → {filename}")
print("[8] 사용된 SEED:", SEED)
print(f"✅ 최종 MAE 점수 : {mae:.6f}")


with open("./Stress/submission/result_log.txt", "a") as f:
    f.write(f"<{SEED} 회차>\n")
    f.write(f"✅ 저장 완료: {filename}\n")
    f.write(f"최종 MAE 점수 : {mae}\n")
    f.write("="*40 + "\n")