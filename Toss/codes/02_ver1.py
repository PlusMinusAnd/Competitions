import pyarrow as pa
import pyarrow.dataset as ds
import gc

# ===== 기존 코드 =====
dset = ds.dataset("./Project/Toss/train.parquet", format="parquet")
dset_test = ds.dataset("./Project/Toss/test.parquet", format="parquet")
# NOTE: 학습/검증을 위해서는 label + feature들이 필요합니다.
#       아래에서 'scanner'는 교체하고, 루프는 그대로 둡니다.

print("[train load]")

# ====== 여기부터 이어붙임 ======
# 필요 패키지
import numpy as np
import pandas as pd
import hashlib
import xgboost as xgb

LABEL_COL = "clicked"
ID_COL = "ID"              # 없으면 자동으로 fallback
BATCH_SIZE = 200_000
RATIO_TRAIN = 0.8
VALID_MAX_ROWS = 300_000    # 검증셋 최대 행 수(메모리 제한)

# pyarrow 버전 호환 스캐너
def make_scanner(dset, columns, batch_size, filter=None):
    try:
        return dset.scanner(columns=columns, batch_size=batch_size, filter=filter)
    except Exception:
        return ds.Scanner.from_dataset(dset, columns=columns, batch_size=batch_size, filter=filter)

# 피처 자동 추출 (label/ID 제외)
schema_names = [f.name for f in dset.schema]
features = [c for c in schema_names if c not in {LABEL_COL, ID_COL}]
print(f"[INFO] #features={len(features)}")

# 간단 전처리(필요 최소): float64 -> float32 (메모리↓)
def preprocess_batch_inplace(df: pd.DataFrame):
    f64 = [c for c in df.columns if c != LABEL_COL and str(df[c].dtype) == "float64"]
    if f64:
        df[f64] = df[f64].astype("float32")
    return df

# 결정적 해시(항상 같은 split) : blake2b 32bit
def hash_mod_u32(values: np.ndarray, m: int = 1000, salt: str = "v1") -> np.ndarray:
    def h(s: str) -> int:
        x = (salt + s).encode("utf-8")
        return int(hashlib.blake2b(x, digest_size=4).hexdigest(), 16) % m
    return np.fromiter((h(str(v)) for v in values), dtype=np.uint32, count=len(values))

# (1) 검증셋 소량 수집 (ID 해시 기반, 전량 적재 없음)
def build_validation_holdout(dset, features, label_col=LABEL_COL, id_col=ID_COL,
                             ratio_train=RATIO_TRAIN, max_rows=VALID_MAX_ROWS,
                             batch_size=BATCH_SIZE):
    rows = 0
    Xs, Ys = [], []
    m = 1000
    th = int(m * ratio_train)

    cols = features + [label_col] + ([id_col] if id_col in schema_names else [])
    sc = make_scanner(dset, columns=cols, batch_size=batch_size)

    for rb in sc.to_batches():
        tbl = pa.Table.from_batches([rb])
        df  = tbl.to_pandas()

        if id_col in df.columns:
            h = hash_mod_u32(df[id_col].astype(str).values, m=m, salt="v1")
            mask_valid = (h >= th)
        else:
            # ID가 없다면 검증을 만들 수 없으니 skip
            mask_valid = np.zeros(len(df), dtype=bool)

        if mask_valid.any():
            dv = df.loc[mask_valid, :].copy()
            yv = dv.pop(label_col).astype("int8").values
            dv = preprocess_batch_inplace(dv)
            Xs.append(dv)
            Ys.append(yv)
            rows += len(dv)
            if rows >= max_rows:
                break

        del rb, tbl, df
        gc.collect()

    if not Xs:
        return None, None, 0

    Xv = pd.concat(Xs, axis=0, ignore_index=True)
    yv = np.concatenate(Ys, axis=0)
    return Xv, yv, len(Xv)

print("[valid build]")
X_valid, y_valid, n_valid = build_validation_holdout(dset, features)
print(f"[VALID] rows={n_valid:,}")

dvalid = xgb.DMatrix(X_valid, label=y_valid, enable_categorical=True) if n_valid > 0 else None

# (2) 학습용 DataIter (ID 해시로 train만 흘려보냄)
class ArrowTrainIter(xgb.core.DataIter):
    def __init__(self, dset, features, label_col=LABEL_COL, id_col=ID_COL,
                 ratio_train=RATIO_TRAIN, batch_size=BATCH_SIZE):
        super().__init__()
        self.dset = dset
        self.feats = list(features)
        self.label = label_col
        self.id = id_col
        self.ratio_train = ratio_train
        self.batch_size = batch_size
        self.m = 1000
        self.th = int(self.m * self.ratio_train)
        self._reset_scanner()

    def _reset_scanner(self):
        cols = self.feats + [self.label] + ([self.id] if self.id in schema_names else [])
        self.scanner = make_scanner(self.dset, columns=cols, batch_size=self.batch_size)
        self._it = iter(self.scanner.to_batches())

    def reset(self):
        self._reset_scanner()

    def next(self, input_data):
        while True:
            try:
                rb = next(self._it)
            except StopIteration:
                return 0

            tbl = pa.Table.from_batches([rb])
            df  = tbl.to_pandas()

            # train 마스크: ID 해시 < 임계
            if self.id in df.columns:
                h = hash_mod_u32(df[self.id].astype(str).values, m=self.m, salt="v1")
                mask = (h < self.th)
            else:
                mask = np.ones(len(df), dtype=bool)

            if not mask.any():
                del rb, tbl, df
                continue

            y = df.loc[mask, self.label].astype("int8").values
            X = df.loc[mask, self.feats].copy()
            X = preprocess_batch_inplace(X)

            input_data(data=X, label=y)

            del rb, tbl, df, X, y
            gc.collect()
            return 1

# (3) XGBoost 학습 (QuantileDMatrix: 대용량/스트리밍 최적화)
params = {
    "objective": "binary:logistic",
    "eval_metric": ["logloss","auc"],
    "tree_method": "gpu_hist",  # GPU 없으면 "hist"
    "max_depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 73,
    "verbosity": 1,
    "max_bin": 128,
}

print("[train build]")
train_iter = ArrowTrainIter(dset, features, LABEL_COL, ID_COL, RATIO_TRAIN, BATCH_SIZE)
dtrain = xgb.QuantileDMatrix(train_iter, enable_categorical=True)

evals = [(dtrain, "train")]
if dvalid is not None:
    evals.append((dvalid, "valid"))

print("[train fit]")
bst = xgb.train(
    params,
    dtrain,
    num_boost_round=800,
    evals=evals,
    early_stopping_rounds=(50 if dvalid is not None else None),
)

bst.save_model("xgb_stream_hashsplit.json")
print("[DONE] model saved -> xgb_stream_hashsplit.json")


