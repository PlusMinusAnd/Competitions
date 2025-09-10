# ============================================================
# Toss CTR (Parquet) - Batch Preprocess → 64D Encoding → LGBM/XGB/Cat (GPU) Ensemble
#   * 입력: Parquet (train/test)
#   * 전처리(요구 반영):
#       - 카테고리: ["gender","age_group","inventory_id"] + 모든 l_feat_* (NaN→-1)
#       - 시간주기: ["hour"(0~23), "day_of_week"(1~7)] 수치로 사용
#       - 수치형: feat_[a~e]_*, history_[a|b]_*
#       - 결측(수치): (hour,dow) 그룹 median → 전역 median
#       - 이상치: 글로벌 q0.5~99.5% clip
#       - 표준화: min>=0면 log1p 후 Robust((x-med)/IQR), 음수 존재시 Robust만
#       - 인코더: IncrementalPCA → 64차원
#       - 수치64 + 카테고리 코드를 concat
#       - 다운샘플링: 배치마다 양성비 ~30%로 음성 부분추출(가중치 보정)
#   * 검증: 마지막 10% 배치를 시간 홀드아웃
#   * 앙상블: LGBM / XGBoost / CatBoost (GPU)
#   * 지표: AP, Weighted LogLoss(50:50), 최종 Score = 0.5*AP + 0.5*(1/(1+WLL))
# ============================================================
import os, re, gc, math, json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import average_precision_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool

# ---------------------- Config ----------------------
SEED = 1
ver  = "ver_proto"
train_path = "./Project/Toss/train.parquet"
test_path  = "./Project/Toss/test.parquet"
submission_path = "./Project/Toss/sample_submission.csv"
save_path = f"./Project/Toss/submission/{ver}_{SEED}_submission/"
os.makedirs(save_path, exist_ok=True)

BATCH_ROWS = 400_000            # 배치 크기
SAMPLE_MAX = 300_000            # 글로벌 통계용 샘플 크기(근사)
IPCA_COMP  = 64
TARGET_COL = "clicked"
ID_COL     = "ID"
rng = np.random.default_rng(SEED)

# 명세
CAT_BASE  = ["gender","age_group","inventory_id"]
TIME_COLS = ["hour","day_of_week"]

IS_LFEAT  = re.compile(r"^l_feat_\d+$")
IS_FA = re.compile(r"^feat_a_\d+$")
IS_FB = re.compile(r"^feat_b_\d+$")
IS_FC = re.compile(r"^feat_c_\d+$")
IS_FD = re.compile(r"^feat_d_\d+$")
IS_FE = re.compile(r"^feat_e_\d+$")
IS_HA = re.compile(r"^history_a_\d+$")
IS_HB = re.compile(r"^history_b_\d+$")

# ---------------------- Metrics ----------------------
def weighted_logloss_5050(y_true, y_pred):
    y_true = y_true.astype(int)
    eps = 1e-12
    p = np.clip(y_pred, eps, 1-eps)
    n_pos = max(1, int((y_true==1).sum()))
    n_neg = max(1, int((y_true==0).sum()))
    w_pos = 0.5 / n_pos
    w_neg = 0.5 / n_neg
    w = np.where(y_true==1, w_pos, w_neg)
    ll = - (w * (y_true*np.log(p) + (1-y_true)*np.log(1-p))).sum()
    return ll

def leaderboard_score(y_true, y_pred):
    ap  = float(average_precision_score(y_true, y_pred))
    wll = float(weighted_logloss_5050(y_true, y_pred))
    score = 0.5*ap + 0.5*(1.0/(1.0 + wll))
    return score, ap, wll

# ---------------------- Stage 0: schema/roles ----------------------
pf = pq.ParquetFile(train_path)
first_df = next(pf.iter_batches(batch_size=1)).to_pandas()

all_cols = list(first_df.columns)
lfeat_cols = [c for c in all_cols if IS_LFEAT.match(c)]
num_cols = [c for c in all_cols if (
    IS_FA.match(c) or IS_FB.match(c) or IS_FC.match(c) or IS_FD.match(c) or IS_FE.match(c) or IS_HA.match(c) or IS_HB.match(c)
)]
num_cols += [c for c in TIME_COLS if c in all_cols]  # 시간형도 수치로 유지
cat_cols = [c for c in CAT_BASE if c in all_cols] + lfeat_cols

print(f"[INFO] categorical={len(cat_cols)} / numeric={len(num_cols)}")

# ---------------------- Stage A: sample stats + fit IPCA ----------------------
print("[StageA] Sampling for global stats & fitting IncrementalPCA...")
# (1) 샘플링으로 글로벌 통계 추정
samples = []
sampled = 0
for rb in pf.iter_batches(batch_size=BATCH_ROWS, columns=num_cols):
    df = rb.to_pandas()
    n = len(df)
    if n == 0: continue
    take = min(n, SAMPLE_MAX - sampled)
    if take > 0:
        idx = rng.choice(n, size=take, replace=False)
        samples.append(df.iloc[idx])
        sampled += take
    if sampled >= SAMPLE_MAX:
        break

S = pd.concat(samples, axis=0, ignore_index=True) if samples else first_df[num_cols]
S = S.apply(pd.to_numeric, errors="coerce")

q_lo_s = S.quantile(0.005)
q_hi_s = S.quantile(0.995)
med_s  = S.median()
q25    = S.quantile(0.25); q75 = S.quantile(0.75)
iqr_s  = (q75 - q25).replace(0, 1.0)
min_s  = S.min()

q_lo = {c: float(q_lo_s.get(c, np.nan)) for c in S.columns}
q_hi = {c: float(q_hi_s.get(c, np.nan)) for c in S.columns}
med  = {c: float(med_s.get(c, 0.0))     for c in S.columns}
iqr  = {c: float(iqr_s.get(c, 1.0))     for c in S.columns}
nonneg = {c: bool(min_s.get(c, 0.0) >= 0.0) for c in S.columns}

del samples, S; gc.collect()

# (2) IPCA partial_fit (전처리 적용한 수치로)
ipca = IncrementalPCA(n_components=IPCA_COMP, whiten=False)

pbar = tqdm(pf.iter_batches(batch_size=BATCH_ROWS, columns=num_cols+TIME_COLS), desc="[StageA] IPCA fit", unit="batch")
for rb in pbar:
    df = rb.to_pandas()
    Xn = df[num_cols].apply(pd.to_numeric, errors="coerce")

    # clip
    for c in Xn.columns:
        lo = q_lo.get(c); hi = q_hi.get(c)
        if lo is not None and not np.isnan(lo): Xn[c] = np.maximum(Xn[c], lo)
        if hi is not None and not np.isnan(hi): Xn[c] = np.minimum(Xn[c], hi)

    # impute: (hour,dow) median → global
    if all(col in df.columns for col in TIME_COLS):
        gb0, gb1 = df[TIME_COLS[0]], df[TIME_COLS[1]]
        for c in Xn.columns:
            s = Xn[c]
            med_by = s.groupby([gb0, gb1]).transform("median")
            s = s.fillna(med_by).fillna(med.get(c, 0.0))
            Xn[c] = s
    else:
        Xn = Xn.fillna(pd.Series(med)).fillna(0.0)

    # log1p for nonneg
    for c in Xn.columns:
        if nonneg.get(c, False):
            Xn[c] = np.log1p(np.maximum(Xn[c], 0.0))

    # robust scale
    for c in Xn.columns:
        denom = iqr.get(c, 1.0); 
        if not np.isfinite(denom) or denom == 0: denom = 1.0
        Xn[c] = (Xn[c] - med.get(c, 0.0)) / denom

    Xn = Xn.replace([np.inf,-np.inf], 0.0).fillna(0.0).to_numpy(dtype=np.float32)
    ipca.partial_fit(Xn)

    del df, Xn, rb; gc.collect()
print("[StageA] IPCA fitted.")

# ---------------------- Stage B: build processed train/valid (with downsampling) ----------------------
proc_train = os.path.join(save_path, "train_proc.csv")
proc_valid = os.path.join(save_path, "valid_proc.csv")
for p in [proc_train, proc_valid]:
    if os.path.exists(p): os.remove(p)

# 카테고리 코드북: 값→코드 (-1은 결측)
cat_codebook = {c:{} for c in cat_cols}
cat_nextcode = {c:0 for c in cat_cols}

# 총 배치 수(홀드아웃 경계)
total_batches = sum(1 for _ in pq.ParquetFile(train_path).iter_batches(batch_size=BATCH_ROWS))
valid_from = math.floor(total_batches * 0.9)
print(f"[StageB] total_batches={total_batches}, valid_from={valid_from}")

TARGET_POS_RATE = 0.30  # 다운샘플 목표 양성비

def cat_encode(col_name, series):
    # NaN→-1, 그 외: codebook에 부여
    out = np.full(len(series), -1, dtype=np.int64)
    mask = series.notna()
    if mask.any():
        vals = series[mask].astype(str).values
        book = cat_codebook[col_name]; nxt = cat_nextcode[col_name]
        idxs = np.where(mask)[0]
        for i, v in enumerate(vals):
            code = book.get(v)
            if code is None:
                book[v] = nxt; code = nxt; nxt += 1
            out[idxs[i]] = code
        cat_nextcode[col_name] = nxt
    return out

batch_idx = 0
pbar = tqdm(pq.ParquetFile(train_path).iter_batches(batch_size=BATCH_ROWS), desc="[StageB] proc train", unit="batch")
for rb in pbar:
    df = rb.to_pandas()
    is_valid = (batch_idx >= valid_from)

    # ---------- Numeric ----------
    Xn = df[num_cols].apply(pd.to_numeric, errors="coerce")
    for c in Xn.columns:
        lo = q_lo.get(c); hi = q_hi.get(c)
        if lo is not None and not np.isnan(lo): Xn[c] = np.maximum(Xn[c], lo)
        if hi is not None and not np.isnan(hi): Xn[c] = np.minimum(Xn[c], hi)

    if all(col in df.columns for col in TIME_COLS):
        gb0, gb1 = df[TIME_COLS[0]], df[TIME_COLS[1]]
        for c in Xn.columns:
            s = Xn[c]
            med_by = s.groupby([gb0, gb1]).transform("median")
            s = s.fillna(med_by).fillna(med.get(c, 0.0))
            Xn[c] = s
    else:
        Xn = Xn.fillna(pd.Series(med)).fillna(0.0)

    for c in Xn.columns:
        if nonneg.get(c, False):
            Xn[c] = np.log1p(np.maximum(Xn[c], 0.0))
    for c in Xn.columns:
        denom = iqr.get(c, 1.0); 
        if not np.isfinite(denom) or denom == 0: denom = 1.0
        Xn[c] = (Xn[c] - med.get(c, 0.0)) / denom

    Xn = Xn.replace([np.inf,-np.inf], 0.0).fillna(0.0)
    X64 = ipca.transform(Xn.to_numpy(dtype=np.float32))
    enc_cols = [f"enc_{i:02d}" for i in range(IPCA_COMP)]
    X64_df = pd.DataFrame(X64, columns=enc_cols)

    # ---------- Categorical ----------
    Xc_dict = {}
    for c in cat_cols:
        Xc_dict[c] = cat_encode(c, df[c])
    Xc_df = pd.DataFrame(Xc_dict, dtype=np.int64)

    # ---------- Concat + Label / Downsample ----------
    out = pd.concat([X64_df, Xc_df], axis=1)
    y = df[TARGET_COL].astype(int).values

    pos_mask = (y==1)
    n_pos = int(pos_mask.sum())
    n_neg = len(y) - n_pos
    if n_pos>0 and n_neg>0:
        r = TARGET_POS_RATE
        f = (n_pos*(1-r)) / max(1, (r*n_neg))
        f = np.clip(f, 0.0, 1.0)
        keep_neg = rng.random(n_neg) < f
        mask = np.zeros(len(y), dtype=bool)
        mask[pos_mask] = True
        mask[np.where(~pos_mask)[0][keep_neg]] = True
        out = out.loc[mask].reset_index(drop=True)
        y = y[mask]
        w = np.ones(len(y), dtype=np.float32)
        w[y==0] = (1.0/float(f)) if f>0 else 1.0
    else:
        w = np.ones(len(y), dtype=np.float32)

    out[TARGET_COL] = y
    out["weight"] = w

    # write
    path = proc_valid if is_valid else proc_train
    header = not os.path.exists(path)
    out.to_csv(path, mode="a", index=False, header=header)

    del df, Xn, X64_df, Xc_df, out, rb
    gc.collect()
    batch_idx += 1

print("[StageB] processed train/valid saved.")

# ---------------------- Stage C: train models (GPU with LGBM CPU fallback) ----------------------
print("[StageC] load processed files...")
tr = pd.read_csv(proc_train)
va = pd.read_csv(proc_valid)

enc_cols = [c for c in tr.columns if c.startswith("enc_")]
cat_model_cols = cat_cols[:]  # 동일 이름
feat_cols = enc_cols + cat_model_cols

Xtr, ytr, wtr = tr[feat_cols], tr[TARGET_COL].values, tr["weight"].values
Xva, yva, wva = va[feat_cols], va[TARGET_COL].values, va["weight"].values

# ---------- LightGBM ----------
# LGBM 용으로만 -1을 NaN으로 바꿔 경고 제거
Xl_tr = Xtr.copy()
Xl_va = Xva.copy()
for c in cat_model_cols:
    mask_tr = Xl_tr[c] < 0
    mask_va = Xl_va[c] < 0
    if mask_tr.any(): Xl_tr.loc[mask_tr, c] = np.nan
    if mask_va.any(): Xl_va.loc[mask_va, c] = np.nan

cat_idx_lgb = [Xl_tr.columns.get_loc(c) for c in cat_model_cols]
lgb_train = lgb.Dataset(Xl_tr, label=ytr, weight=wtr,
                        categorical_feature=cat_idx_lgb, free_raw_data=False)
lgb_valid = lgb.Dataset(Xl_va, label=yva, weight=wva,
                        categorical_feature=cat_idx_lgb, free_raw_data=False)

lgb_params_gpu = dict(
    objective="binary",
    metric=["binary_logloss","auc"],
    learning_rate=0.05,
    num_leaves=512,
    max_depth=-1,
    min_data_in_leaf=100,
    feature_fraction=0.9,
    bagging_fraction=0.8,
    bagging_freq=1,
    max_bin=255,
    device_type="gpu",      # 먼저 GPU 시도
    seed=SEED,
    verbose=-1
)

try:
    print("[LGBM] trying GPU...")
    lgbm = lgb.train(
        lgb_params_gpu, lgb_train, num_boost_round=10000,
        valid_sets=[lgb_train, lgb_valid], valid_names=["train","valid"],
        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(100)]
    )
except Exception as e:
    print("[LGBM] GPU failed → fallback to CPU:", e)
    lgb_params_cpu = dict(lgb_params_gpu)
    lgb_params_cpu["device_type"] = "cpu"
    lgbm = lgb.train(
        lgb_params_cpu, lgb_train, num_boost_round=10,
        valid_sets=[lgb_train, lgb_valid], valid_names=["train","valid"],
        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(100)]
    )
    

# ---- XGBoost (GPU, 2.x 스타일) ----
dtrain = xgb.DMatrix(Xtr, label=ytr, weight=wtr, enable_categorical=True)
dvalid = xgb.DMatrix(Xva, label=yva, weight=wva, enable_categorical=True)

xgb_params = {
    "objective": "binary:logistic",
    "eval_metric": ["logloss", "auc"],
    "tree_method": "hist",   # ← hist로 두고
    "device": "cuda",        # ← GPU는 이걸로 지정
    "max_depth": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "learning_rate": 0.05,
    "max_cat_to_onehot": 16,
    "seed": SEED,
    # "early_stopping_rounds" : 200
}
xgbm = xgb.train(
    xgb_params, dtrain, num_boost_round=10000,
    evals=[(dtrain, "train"), (dvalid, "valid")],
    verbose_eval=100, early_stopping_rounds=200
)


# ---------- CatBoost (GPU 유지) ----------
cat_idx_cb = [Xtr.columns.get_loc(c) for c in cat_model_cols]
train_pool = Pool(Xtr, label=ytr, weight=wtr, cat_features=cat_idx_cb)
valid_pool = Pool(Xva, label=yva, weight=wva, cat_features=cat_idx_cb)

cbc = CatBoostClassifier(
    loss_function="Logloss",
    eval_metric="AUC",
    depth=8,
    learning_rate=0.05,
    l2_leaf_reg=3.0,
    iterations=10000,
    random_seed=SEED,
    od_type="Iter",
    od_wait=200,
    task_type="GPU",
    verbose=100,
)
cbc.fit(train_pool, eval_set=valid_pool, verbose=100)


# ---------- Validation ensemble ----------
pred_lgb = lgbm.predict(Xl_va, num_iteration=lgbm.best_iteration)
pred_xgb = xgbm.predict(dvalid, iteration_range=(0, xgbm.best_iteration+1))
pred_cat = cbc.predict_proba(Xva)[:,1]
pred_va  = (pred_lgb + pred_xgb + pred_cat) / 3.0

score, ap, wll = leaderboard_score(yva, pred_va)
print(f"[VALID] AP={ap:.6f}  WLL={wll:.6f}  SCORE={score:.6f}")

del lgb_train, lgb_valid; gc.collect()
del dtrain, dvalid; gc.collect()
del train_pool, valid_pool; gc.collect()

# ---------------------- Stage D: process TEST & predict ----------------------
print("[StageD] Process TEST & predict...")
out_chunks = []
pbar = tqdm(pq.ParquetFile(test_path).iter_batches(batch_size=BATCH_ROWS), desc="[StageD] test", unit="batch")
for rb in pbar:
    df = rb.to_pandas()

    # Numeric
    Xn = df[num_cols].apply(pd.to_numeric, errors="coerce")
    for c in Xn.columns:
        lo = q_lo.get(c); hi = q_hi.get(c)
        if lo is not None and not np.isnan(lo): Xn[c] = np.maximum(Xn[c], lo)
        if hi is not None and not np.isnan(hi): Xn[c] = np.minimum(Xn[c], hi)

    if all(col in df.columns for col in TIME_COLS):
        gb0, gb1 = df[TIME_COLS[0]], df[TIME_COLS[1]]
        for c in Xn.columns:
            s = Xn[c]
            med_by = s.groupby([gb0, gb1]).transform("median")
            s = s.fillna(med_by).fillna(med.get(c, 0.0))
            Xn[c] = s
    else:
        Xn = Xn.fillna(pd.Series(med)).fillna(0.0)

    for c in Xn.columns:
        if nonneg.get(c, False):
            Xn[c] = np.log1p(np.maximum(Xn[c], 0.0))
    for c in Xn.columns:
        denom = iqr.get(c, 1.0); 
        if not np.isfinite(denom) or denom == 0: denom = 1.0
        Xn[c] = (Xn[c] - med.get(c, 0.0)) / denom

    X64 = ipca.transform(Xn.replace([np.inf,-np.inf], 0.0).fillna(0.0).to_numpy(dtype=np.float32))
    X64_df = pd.DataFrame(X64, columns=[f"enc_{i:02d}" for i in range(IPCA_COMP)])

    # Categorical
    Xc_dict = {}
    for c in cat_cols:
        # 테스트에서 새 값이면 새 코드 부여 (일관성을 위해 학습 codebook에 추가)
        Xc_dict[c] = cat_encode(c, df[c])
    Xc_df = pd.DataFrame(Xc_dict, dtype=np.int64)

    X = pd.concat([X64_df, Xc_df], axis=1)

    p_lgb = lgbm.predict(X, num_iteration=lgbm.best_iteration)
    p_xgb = xgbm.predict(xgb.DMatrix(X, enable_categorical=True), iteration_range=(0, xgbm.best_iteration+1))
    p_cat = cbc.predict_proba(X)[:,1]
    p = (p_lgb + p_xgb + p_cat) / 3.0

    part = pd.DataFrame({ID_COL: df[ID_COL].values, "clicked": p})
    out_chunks.append(part)

    del df, Xn, X64_df, Xc_df, X, rb
    gc.collect()

pred_all = pd.concat(out_chunks, axis=0, ignore_index=True)
sub = pd.read_csv(submission_path)[[ID_COL]]
out = sub.merge(pred_all, on=ID_COL, how="left")
out_path = os.path.join(save_path, f"{ver}_{SEED}_submission.csv")
out.to_csv(out_path, index=False)

# 로그 저장
summary = {
    "SEED": SEED,
    "ver": ver,
    "IPCA_COMP": IPCA_COMP,
    "BATCH_ROWS": BATCH_ROWS,
    "TARGET_POS_RATE": TARGET_POS_RATE,
    "lgb_best_iter": int(lgbm.best_iteration),
    "xgb_best_iter": int(xgbm.best_iteration),
    "cat_best_iter": int(cbc.get_best_iteration()),
    "valid_AP": float(ap),
    "valid_WLL": float(wll),
    "valid_SCORE": float(score),
}

summary_path = os.path.join(save_path, "run_summary.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("[SAVE]", summary_path)
