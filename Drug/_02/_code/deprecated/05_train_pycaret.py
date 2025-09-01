import pandas as pd
import numpy as np
from pycaret.regression import *
import datetime

r = 394  # 시드 고정

# 데이터 로드
train_df = pd.read_csv("./Drug/_02/full_pipeline/0_dataset/train_descriptor.csv")
test_df = pd.read_csv("./Drug/_02/full_pipeline/0_dataset/test_descriptor.csv")
test_id = pd.read_csv("./Drug/test.csv")["ID"]

# ✅ PyCaret 설정
setup(
    data=train_df,
    target='Inhibition',
    session_id=r,
    normalize=True,
    verbose=False,
    use_gpu=False,
    feature_selection=True,    # 불필요한 피처 제거
    remove_outliers=True,      # 이상치 제거
    polynomial_features=False,
    fold=3                     # 속도 최적화용
)

# ✅ 튜닝 없이 Ridge 모델 생성 및 고정
model = create_model('ridge')
final_model = finalize_model(model)

# 🔹 OOF 예측 저장
train_preds = predict_model(final_model, data=train_df)
label_col = [col for col in train_preds.columns if col.lower() in ['label', 'prediction_label', 'prediction']][-1]
train_oof = train_preds[label_col].values
np.save("./Drug/_02/full_pipeline/pycaret_oof.npy", train_oof)

# 🔹 Test 예측 저장
test_preds = predict_model(final_model, data=test_df)
test_pred_values = test_preds[label_col].values
np.save("./Drug/_02/full_pipeline/pycaret_preds.npy", test_pred_values)

# 🔹 제출 파일 저장
submission = pd.DataFrame({
    "ID": test_id,
    "Inhibition": test_pred_values
})
now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
filename = f"submission_pycaret_{r}_{now}.csv"
submission.to_csv(f"./Drug/_02/full_pipeline/{filename}", index=False)

# 🔹 모델 저장
save_model(final_model, "./Drug/_02/full_pipeline/pycaret_final_model")

print("✅ 튜닝 없이 Ridge 모델 학습 및 저장 완료")
