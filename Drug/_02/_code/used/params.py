import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import datetime
import os

# 시드 고정
r = 394
np.random.seed(r)

# 📌 평가 함수 정의
def print_scores(y_true, y_pred, label=""):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nrmse = rmse / (np.max(y_true) - np.min(y_true))
    pearson = pearsonr(y_true, y_pred)[0]
    score = 0.5 * (1 - min(nrmse, 1)) + 0.5 * np.clip(pearson, 0, 1)
    print(f"📊 {label} 결과")
    print(f"RMSE     : {rmse:.5f}")
    print(f"NRMSE    : {nrmse:.5f}")
    print(f"Pearson  : {pearson:.5f}")
    print(f"Score📈  : {score:.5f}")
    return score

# 🔹 데이터 로드
train = pd.read_csv("./Drug/_02/full_pipeline/0_dataset/train_descriptor.csv")
test = pd.read_csv("./Drug/_02/full_pipeline/0_dataset/test_descriptor.csv")
X = train.drop(columns=["Inhibition"])
y = train["Inhibition"].values
X_test = test.copy()

# ✅ 최적 파라미터 (Optuna 튜닝 결과 입력)
param_lgbm = {
    "n_estimators": 1000,
    "learning_rate": 0.024615467196152204,
    "max_depth": 3,
    "num_leaves": 31,
    "min_child_samples": 20,
    "colsample_bytree": 0.6411145386746391,
    "random_state": r,
    "verbosity": -1
}

param_xgb = {
    "n_estimators": 1000,
    'learning_rate': 0.04018614797174392,
    'max_depth': 3,
    'subsample': 0.7086680178296805,
    'colsample_bytree': 0.6201094943343927,
    'reg_alpha': 3.056666214937805,
    'reg_lambda': 2.370292354775882,
    "random_state": r,
    "verbosity": 0
}

param_cat = {
    "iterations": 1000,
    "learning_rate": 0.03905462846751998,
    "depth": 4,
    "l2_leaf_reg": 6.738080359145052,
    "bagging_temperature": 0.4929876007310619,
    "verbose": 0,
    "random_seed": r
}

# 🔹 모델 정의
models = {
    "lgbm": LGBMRegressor(**param_lgbm),
    "xgb": XGBRegressor(**param_xgb, early_stopping_rounds=20,),
    "cat": CatBoostRegressor(**param_cat)
}

kf = KFold(n_splits=5, shuffle=True, random_state=r)
oof_preds = {name: np.zeros(len(X)) for name in models}
test_preds = {name: np.zeros(len(X_test)) for name in models}

print("🔹 Boosting 모델별 5-Fold 학습 시작")

for name, model in models.items():
    print(f"✅ {name.upper()} 시작")
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        if name == "lgbm":
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="rmse",
                callbacks=[early_stopping(20), log_evaluation(period=0)]
            )
        elif name == "xgb":
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                
                verbose=False
            )
        elif name == "cat":
            model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                early_stopping_rounds=20,
                use_best_model=True
            )

        oof_preds[name][val_idx] = model.predict(X_val)
        test_preds[name] += model.predict(X_test) / kf.n_splits

    print_scores(y, oof_preds[name], label=name.upper())

# 🔹 앙상블 최적 가중치 탐색 (OOF 기준)
best_score = -np.inf
best_weights = None
alphas = np.linspace(0, 1, 11)

for a in alphas:
    for b in alphas:
        if a + b > 1: continue
        c = 1 - a - b
        blended = a * oof_preds["lgbm"] + b * oof_preds["xgb"] + c * oof_preds["cat"]
        score = print_scores(y, blended, label=f"LGBM={a:.2f}, XGB={b:.2f}, CAT={c:.2f}")
        if score > best_score:
            best_score = score
            best_weights = (a, b, c)

a, b, c = best_weights
print(f"✅ 최적 가중치: LGBM={a:.2f}, XGB={b:.2f}, CAT={c:.2f} | Score: {best_score:.5f}")

final_oof = a * oof_preds["lgbm"] + b * oof_preds["xgb"] + c * oof_preds["cat"]
final_test = a * test_preds["lgbm"] + b * test_preds["xgb"] + c * test_preds["cat"]

print_scores(y, final_oof, label="Final Stacked Ensemble")

# 🔹 저장
submission = pd.DataFrame({
    "ID": pd.read_csv("./Drug/test.csv")["ID"],
    "Inhibition": final_test
})

now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
filename = f"submission_boost_best_{r}_{now}.csv"
save_path = f"./Drug/_02/full_pipeline/{filename}"
submission.to_csv(save_path, index=False)

np.save("./Drug/_02/full_pipeline/boost_oof_best.npy", final_oof)
np.save("./Drug/_02/full_pipeline/boost_preds_best.npy", final_test)

print(f"✅ 최종 예측 저장 완료 → {filename}")
