
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import os

r = 73

np.random.seed(r)

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

train = pd.read_csv("./Drug/_02/0_dataset/train_descriptor.csv")
test = pd.read_csv("./Drug/_02/0_dataset/test_descriptor.csv")
X = train.drop(columns=["Inhibition"])
y = train["Inhibition"].values
X_test = test.copy()

kf = KFold(n_splits=5, shuffle=True, random_state=r)

param_lgbm = {
    "n_estimators": 1000,
    "learning_rate": 0.015875869112729403,
    "max_depth": 5,
    "num_leaves": 77,
    "min_child_samples": 21,
    "colsample_bytree": 0.8956019941712514,
    "random_state": r
}

param_cat = {
    "iterations": 1000,
    "learning_rate": 0.020782688572110002,
    "depth": 4,
    "l2_leaf_reg": 9.322000582507451,
    "verbose": 0,
    "random_seed": r
}

param_xgb = {
    "n_estimators": 1000,
    "learning_rate": 0.010607554597937956,
    "max_depth": 3,
    "subsample": 0.7961002707424782,
    "colsample_bytree": 0.8105132963922017,
    "verbosity": 0,
    "random_state": r
}

models = {
    "lgbm": LGBMRegressor(**param_lgbm, verbosity=-1),
    "xgb": XGBRegressor(**param_xgb, early_stopping_rounds=50),
    "cat": CatBoostRegressor(**param_cat)
}

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
                callbacks=[early_stopping(stopping_rounds=10), log_evaluation(period=0)],
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
                early_stopping_rounds=50,
                use_best_model=True
            )

        oof_preds[name][val_idx] = model.predict(X_val)
        test_preds[name] += model.predict(X_test) / kf.n_splits

    print_scores(y, oof_preds[name], label=f"{name.upper()}")

# OOF stacking 앙상블
print("🔹 OOF 앙상블 최적 alpha 탐색")
best_score = -np.inf
best_weights = None
alphas = np.linspace(0, 1, 11)

for a in alphas:
    for b in alphas:
        if a + b > 1: continue
        c = 1 - a - b
        blended = a * oof_preds["lgbm"] + b * oof_preds["xgb"] + c * oof_preds["cat"]
        rmse = np.sqrt(mean_squared_error(y, blended))
        nrmse = rmse / (np.max(y) - np.min(y))
        pearson = pearsonr(y, blended)[0]
        score = 0.5 * (1 - min(nrmse, 1)) + 0.5 * np.clip(pearson, 0, 1)
        if score > best_score:
            best_score = score
            best_weights = (a, b, c)

a, b, c = best_weights
print(f"✅ 최적 가중치: LGBM={a:.2f}, XGB={b:.2f}, CAT={c:.2f} | Score: {best_score:.5f}")

final_oof = a * oof_preds["lgbm"] + b * oof_preds["xgb"] + c * oof_preds["cat"]
final_test = a * test_preds["lgbm"] + b * test_preds["xgb"] + c * test_preds["cat"]

print_scores(y, final_oof, label="Stacked Ensemble")

submission = pd.DataFrame({
    "ID": pd.read_csv("./Drug/test.csv")["ID"],
    "Inhibition": final_test
})

import datetime

# 현재 시간 포맷
now = datetime.datetime.now().strftime("%Y%m%d_%H%M")

# 파일명 생성
filename = f"pre_DNN_{r}_({now}).csv"
save_path = f"./Drug/_02/1_pre/{filename}"

# 저장
submission.to_csv(save_path, index=False)
print(f"✅ {filename} 저장 완료")

np.save("./Drug/_02/2_npy/boost_oof.npy", final_oof)
np.save("./Drug/_02/2_npy/boost_preds.npy", final_test)
print("✅ boost_oof.npy, boost_preds.npy 저장 완료")