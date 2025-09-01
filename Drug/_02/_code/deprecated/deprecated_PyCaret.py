import pandas as pd
import numpy as np
from pycaret.regression import *
import datetime

# 데이터 로드
train = pd.read_csv("./Drug/_02/full_pipeline/0_dataset/train_descriptor.csv")
test = pd.read_csv("./Drug/_02/full_pipeline/0_dataset/test_descriptor.csv")
test_id = pd.read_csv("./Drug/test.csv")["ID"]

# 시드 고정
r = 238

# PyCaret 설정
setup(
    data=train,
    target='Inhibition',
    session_id=r,
    normalize=True,
    verbose=False,
    use_gpu=False,
    feature_selection=False,
    remove_outliers=False,
    polynomial_features=False
)

# 모델 비교 및 최종 선택
best_model = compare_models(include=['lr', 'ridge', 'lasso'], sort='R2')
final_model = finalize_model(best_model)

# 🔹 train OOF 예측
train_preds = predict_model(final_model, data=train)

# ✅ 예측 컬럼명 자동 감지
label_col = None
for col in ['Label', 'prediction_label', 'Prediction']:
    if col in train_preds.columns:
        label_col = col
        break
if label_col is None:
    label_col = train_preds.columns[-1]  # fallback: 마지막 컬럼

train_pred_values = train_preds[label_col].values
np.save("./Drug/_02/full_pipeline/pycaret_oof.npy", train_pred_values)

# 🔹 test 예측
test_preds = predict_model(final_model, data=test)
if label_col in test_preds.columns:
    y_pred = test_preds[label_col].values
else:
    y_pred = test_preds.iloc[:, -1].values  # fallback

np.save("./Drug/_02/full_pipeline/pycaret_preds.npy", y_pred)

# 🔹 제출 파일 생성
submission = pd.DataFrame({
    "ID": test_id,
    "Inhibition": y_pred
})
now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
submission.to_csv(f"./Drug/_02/full_pipeline/submission_pycaret_{r}_{now}.csv", index=False)

# 🔹 모델 저장
save_model(final_model, "./Drug/_02/full_pipeline/pycaret_final_model")

print("✅ PyCaret 예측 및 저장 완료")
