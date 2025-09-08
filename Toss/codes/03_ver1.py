# /home/pc/Study/Project/Toss/codes/run_once_xgb_stream.py
# 분할 Parquet(20만행 단위)에서 배치 스트리밍 학습/예측 + tqdm 진행률 + 에러 무시 옵션
import os, gc, csv, time, math, warnings, json, datetime
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import xgboost as xgb
from tqdm.auto import tqdm

# ===== 설정 =====
VER = "ver1"
BASE = "/home/pc/Study/Project/Toss"
TRAIN_DIR = f"{BASE}/_split/train_200k"
TEST_DIR  = f"{BASE}/_split/test_200k"
OUT_DIR   = f"{BASE}/_out/{VER}_submission"
os.makedirs(OUT_DIR, exist_ok=True)

seed_file = f"{OUT_DIR}/SEED_COUNTS.json"
if not os.path.exists(seed_file):
    seed_state = {"seed": 770}
else:
    with open(seed_file, "r") as f:
        seed_state = json.load(f)

SEED = seed_state["seed"]
print(f"[Current Run SEED]: {SEED}")
seed_state["seed"] += 1
with open(seed_file, "w") as f:
    json.dump(seed_state, f)

save_path = f'{OUT_DIR}/{SEED}_submission_{VER}/'
os.makedirs(save_path , exist_ok=True )

# ---- dtype helpers ----
def to_cat(df, cols):
    for c in cols:
        if c in df:
            df[c] = df[c].astype("category")
    return df

def to_int16_safe(df, cols):
    for c in cols:
        if c in df:
            s = pd.to_numeric(df[c], errors="coerce").fillna(-1).astype(np.int16)
            df[c] = s
    return df

def to_f32(df):
    float_cols = df.select_dtypes(include=["float", "float32", "float64"]).columns
    if len(float_cols) > 0:
        df[float_cols] = df[float_cols].astype("float32")
    return df

def enforce_xgb_dtypes(df, cats=None, ints=None):
    """pd.concat 등으로 object로 승격된 컬럼을 XGBoost 호환 dtype으로 강제 변환."""
    if cats is None: cats = CATS
    if ints is None: ints = INTS
    df = df.copy()
    for c in cats:
        if c in df.columns:
            df[c] = df[c].astype("string").astype("category")
    for c in ints:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(-1).astype(np.int16)
    for c in df.columns:
        if c in cats or c in ints:
            continue
        s = df[c]
        if pd.api.types.is_object_dtype(s):
            s2 = pd.to_numeric(s, errors="ignore")
            if pd.api.types.is_numeric_dtype(s2):
                df[c] = s2
            else:
                df[c] = s.astype("string").astype("category")
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].astype("float32")
        elif pd.api.types.is_integer_dtype(df[c]):
            if "Int" in str(df[c].dtype):
                df[c] = df[c].fillna(-1).astype(np.int32)
    bad = [c for c in df.columns if df[c].dtype.name == "object"]
    for c in bad:
        df[c] = df[c].astype("string").astype("category")
    return df

# ---- GPU 가용성 체크 ----
def gpu_ok():
    try:
        _ = xgb.train(
            {"tree_method": "gpu_hist", "predictor": "gpu_predictor"},
            xgb.DMatrix(np.array([[0.0],[1.0]], dtype=np.float32), label=np.array([0,1], dtype=np.int8)),
            num_boost_round=1
        )
        return True
    except xgb.core.XGBoostError:
        return False

# 메모리/속도 튜닝
TRAIN_BATCH = 120_000
TEST_BATCH  = 150_000
NUM_BOOST_ROUND = 800

# === Early Stopping용 검증 샘플 설정 ===
EARLY_STOP_ROUNDS = 100
VALID_SAMPLE_FRAC = 0.10
VALID_MAX_ROWS    = 400_000
VALID_BATCH       = 150_000

USE_GPU = gpu_ok()
if USE_GPU:
    TREE_METHOD = "gpu_hist"
    PREDICTOR   = "gpu_predictor"
else:
    TREE_METHOD = "hist"
    PREDICTOR   = None
    print("[GPU] 사용 불가. CPU(hist)로 전환합니다.")

SILENCE_ERRORS = True

CATS = ("gender","age_group","inventory_id")
INTS = ("day_of_week","hour")
EXCLUDE = {"clicked","ID","seq"}
DROP_VIRTUAL = {"__fragment_index","__batch_index","__last_in_fragment","__filename"}

# 다운샘플 파라미터(언더샘플링 + 가중치)
NEG_KEEP_RATIO = 0.2
USE_DS = True

# ===== util =====
def dataset_and_cols(path: str):
    dset = ds.dataset(path, format="parquet")
    cols = [c for c in dset.schema.names if c not in DROP_VIRTUAL]
    return dset, cols

def count_rows(dset: ds.Dataset) -> int:
    return dset.count_rows()

def n_batches(total_rows: int, batch_size: int) -> int:
    return max(1, math.ceil(total_rows / batch_size))

# ===== 데이터셋/피처 =====
train_ds, train_cols = dataset_and_cols(TRAIN_DIR)
test_ds,  test_cols  = dataset_and_cols(TEST_DIR)
features = [c for c in train_cols if (c in test_cols) and (c not in EXCLUDE)]
print(f"[INFO] #features = {len(features)}")
missing_in_test = [c for c in train_cols if c not in test_cols and c not in EXCLUDE]
if missing_in_test:
    print(f"[WARN] train-only cols ignored: {missing_in_test[:8]}{' ...' if len(missing_in_test)>8 else ''}")

# ===== 라벨 카운트(진행바) =====
def label_counts(dset: ds.Dataset, batch=150_000):
    total = count_rows(dset)
    sc = dset.scanner(columns=["clicked"], batch_size=batch)
    pos = 0; tot = 0
    with tqdm(total=total, unit="rows", desc="[COUNT] labels") as pbar:
        for b in sc.to_batches():
            arr = b.column(0).to_numpy(zero_copy_only=False)
            n = arr.size
            tot += n; pos += int(arr.sum())
            pbar.update(n)
    return pos, tot - pos

print("[INFO] counting labels …")
pos, neg = label_counts(train_ds)
den = max(1, pos + neg)
print(f"[INFO] pos={pos:,}, neg={neg:,}, pi={pos/den:.6f}")

# ===== EarlyStopping용 dvalid =====
def build_validation_dmatrix(dset: ds.Dataset, feature_cols, label_col="clicked",
                             sample_frac=VALID_SAMPLE_FRAC, max_rows=VALID_MAX_ROWS,
                             batch_size=VALID_BATCH, seed=SEED):
    rs = np.random.RandomState(seed)
    X_list, y_list = [], []
    total = count_rows(dset)
    used = 0
    sc = dset.scanner(columns=feature_cols + [label_col], batch_size=batch_size)
    with tqdm(total=total, unit="rows", desc="[VALID] build") as pbar:
        for b in sc.to_batches():
            tbl = pa.Table.from_batches([b])
            df  = tbl.to_pandas()
            pbar.update(b.num_rows)

            m = rs.rand(len(df)) < sample_frac
            if not m.any():
                del tbl, df, b
                continue

            sdf = df.loc[m].copy()
            y = sdf.pop(label_col).astype(np.int8).values
            sdf = to_cat(sdf, CATS)
            sdf = to_int16_safe(sdf, INTS)
            sdf = to_f32(sdf)

            X_list.append(sdf)
            y_list.append(y)

            used += len(y)
            del tbl, df, sdf, b
            gc.collect()
            if used >= max_rows:
                break

    if used == 0:
        return None, None, 0
    X = pd.concat(X_list, axis=0, ignore_index=True)
    y = np.concatenate(y_list)

    # ★ concat으로 object 승격 방지
    X = enforce_xgb_dtypes(X, CATS, INTS)

    dvalid = xgb.DMatrix(X, label=y, enable_categorical=True)
    return dvalid, y, used

# ===== DataIter 구현 (tqdm + 에러 무시) =====
class ArrowIter(xgb.core.DataIter):
    """PyArrow Dataset → XGBoost 배치 주입(DataIter) with tqdm + silent skip"""
    def __init__(self, dset: ds.Dataset, feature_cols, label_col=None, batch_size=120_000,
                 pbar_desc: str = "", silence_errors: bool = True):
        super().__init__()
        self.dset = dset
        self.feats = list(feature_cols)
        self.label = label_col
        self.batch_size = batch_size
        self.silence_errors = silence_errors

        self.skipped_batches = 0
        self.skipped_rows = 0

        self.total_rows = count_rows(dset)
        self.pbar = tqdm(total=self.total_rows, unit="rows", desc=pbar_desc)

        cols = self.feats + ([label_col] if label_col else [])
        self.scanner = dset.scanner(columns=cols, batch_size=batch_size)
        self._it = iter(self.scanner.to_batches())

    def reset(self):
        cols = self.feats + ([self.label] if self.label else [])
        self.scanner = self.dset.scanner(columns=cols, batch_size=self.batch_size)
        self._it = iter(self.scanner.to_batches())

    def _process_batch(self, batch):
        tbl = pa.Table.from_batches([batch])
        df  = tbl.to_pandas()

        y = None
        w = None
        if self.label and self.label in df:
            y = df.pop(self.label).astype("int8").values

            if USE_DS:
                neg_mask = (y == 0)
                keep = np.ones_like(y, dtype=bool)
                if neg_mask.any():
                    rng = np.random.default_rng(seed=SEED)
                    keep_neg = rng.random(neg_mask.sum()) < NEG_KEEP_RATIO
                    keep[neg_mask] = keep_neg
                df = df.loc[keep].reset_index(drop=True)
                y  = y[keep]

                n_pos = (y == 1).sum()
                n_neg = (y == 0).sum()
                if n_pos > 0 and n_neg > 0:
                    w_pos = 0.5 / n_pos
                    w_neg = 0.5 / n_neg
                    w = np.where(y == 1, w_pos, w_neg).astype(np.float32)
                else:
                    w = np.ones_like(y, dtype=np.float32)
            else:
                w = np.ones_like(y, dtype=np.float32)

        # dtype 경량화 + XGB 호환
        df = to_cat(df, CATS)
        df = to_int16_safe(df, INTS)
        df = to_f32(df)
        df = enforce_xgb_dtypes(df, CATS, INTS)
        return df, y, w

    def next(self, input_data):
        while True:
            try:
                batch = next(self._it)
            except StopIteration:
                self.pbar.close()
                return 0

            n = batch.num_rows
            self.pbar.update(n)

            try:
                df, y, w = self._process_batch(batch)
                input_data(data=df, label=y, weight=w)
                del df, y, batch
                gc.collect()
                return 1
            except Exception:
                if self.silence_errors:
                    self.skipped_batches += 1
                    self.skipped_rows += n
                    continue
                else:
                    raise

# ===== 학습 진행바 콜백 =====
class TQDMCallback(xgb.callback.TrainingCallback):
    def __init__(self, total_rounds: int, desc: str = "[TRAIN] boosting",
                 eval_name: str = "train", metrics=("logloss", "auc")):
        self.pbar = tqdm(total=total_rounds, desc=desc, unit="round")
        self.eval_name = eval_name
        self.metrics = metrics

    def after_iteration(self, model, epoch: int, evals_log):
        if self.eval_name in evals_log:
            latest = {}
            for m in self.metrics:
                if m in evals_log[self.eval_name]:
                    v = evals_log[self.eval_name][m][-1]
                    latest[m] = f"{v:.5f}"
            if latest:
                self.pbar.set_postfix(latest, refresh=False)
        self.pbar.update(1)
        return False

    def after_training(self, model):
        self.pbar.close()
        return model

# ======= 리더보드 지표(AP, WLL) 계산 유틸 =======
def average_precision_simple(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int8)
    if y_true.sum() == 0:
        return 0.0
    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]
    cum_tp = np.cumsum(y_sorted)
    precision = cum_tp / (np.arange(y_sorted.size) + 1)
    ap = precision[y_sorted == 1].mean()
    return float(ap)

def weighted_logloss_5050(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-15) -> float:
    y_true = y_true.astype(np.int8)
    pos_mask = (y_true == 1)
    neg_mask = ~pos_mask
    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()
    if n_pos == 0 or n_neg == 0:
        p = np.clip(y_prob, eps, 1 - eps)
        return float(-np.mean(y_true*np.log(p) + (1-y_true)*np.log(1-p)))
    p_pos = np.clip(y_prob[pos_mask], eps, 1 - eps)
    p_neg = np.clip(y_prob[neg_mask], eps, 1 - eps)
    wll = -0.5*(np.log(p_pos).mean()) - 0.5*(np.log(1 - p_neg).mean())
    return float(wll)

# --- 온도 스케일링 (선택: 기본 ON) ---
USE_TEMP_SCALING = True
def calibrate_temperature(y_true, p, iters=200, lr=0.05):
    """WLL(50:50) 최소화하는 T를 간단 수치미분으로 최적화."""
    eps=1e-15
    logit = np.log(np.clip(p,eps,1-eps)) - np.log(1-np.clip(p,eps,1-eps))
    T = 1.0
    for _ in range(iters):
        pT = 1/(1+np.exp(-logit/max(T,1e-3)))
        wll = weighted_logloss_5050(y_true, pT)
        dT = 1e-3
        pT2 = 1/(1+np.exp(-logit/max(T+dT,1e-3)))
        wll2 = weighted_logloss_5050(y_true, pT2)
        g = (wll2 - wll)/dT
        T -= lr*g
        if T < 1e-3: T = 1e-3
        if T > 100.0: T = 100.0
    return T

def apply_temperature(p, T):
    eps=1e-15
    logit = np.log(np.clip(p,eps,1-eps)) - np.log(1-np.clip(p,eps,1-eps))
    return 1/(1+np.exp(-logit/max(T,1e-3)))

def eval_on_validation(bst, dset: ds.Dataset, feature_cols, label_col="clicked",
                       sample_frac=VALID_SAMPLE_FRAC, max_rows=VALID_MAX_ROWS,
                       batch_size=VALID_BATCH, seed=SEED, iteration_range=None, T=None):
    rs = np.random.RandomState(seed)
    y_list = []; p_list = []
    total = count_rows(dset); used = 0
    sc = dset.scanner(columns=feature_cols + [label_col], batch_size=batch_size)
    with tqdm(total=total, unit="rows", desc="[VALID] scan") as pbar:
        for b in sc.to_batches():
            tbl = pa.Table.from_batches([b]); df  = tbl.to_pandas()
            pbar.update(b.num_rows)
            m = rs.rand(len(df)) < sample_frac
            if not m.any():
                del tbl, df, b; continue
            sdf = df.loc[m].copy()
            y = sdf.pop(label_col).astype(np.int8).values
            sdf = to_cat(sdf, CATS); sdf = to_int16_safe(sdf, INTS); sdf = to_f32(sdf)
            sdf = enforce_xgb_dtypes(sdf, CATS, INTS)
            dmx = xgb.DMatrix(sdf, enable_categorical=True)
            p = bst.predict(dmx, iteration_range=iteration_range) if iteration_range else bst.predict(dmx)
            if T is not None:
                p = apply_temperature(p, T)
            y_list.append(y); p_list.append(p)
            used += len(y)
            del tbl, df, sdf, dmx, b, y, p; gc.collect()
            if used >= max_rows: break

    if used == 0:
        return {"AP": None, "WLL": None, "Score": None, "Used": 0}
    y_true = np.concatenate(y_list); y_prob = np.concatenate(p_list)
    ap  = average_precision_simple(y_true, y_prob)
    wll = weighted_logloss_5050(y_true, y_prob)
    score = 0.5*ap + 0.5*(1.0/(1.0 + wll))
    return {"AP": ap, "WLL": wll, "Score": score, "Used": used}

# ===== 메인 =====
def main():
    t0 = time.time()

    try:
        xgb.train({'tree_method': TREE_METHOD},
                  xgb.DMatrix([[0.0],[1.0]], label=[0,1]), num_boost_round=1)
        print(f"[GPU/CPU] tree_method={TREE_METHOD} OK")
    except xgb.core.XGBoostError as e:
        print(f"[GPU] '{TREE_METHOD}' 불가 → CPU 'hist'로 자동 전환 권장. 에러: {e}")

    # === QuantileDMatrix 빌드(스트리밍) ===
    train_iter = ArrowIter(train_ds, features, label_col="clicked",
                           batch_size=TRAIN_BATCH, pbar_desc="[QDM] bins",
                           silence_errors=SILENCE_ERRORS)

    dtrain = xgb.QuantileDMatrix(train_iter, enable_categorical=True)

    # === dvalid 생성 ===
    dvalid, y_valid, used_valid = build_validation_dmatrix(
        train_ds, features, label_col="clicked",
        sample_frac=VALID_SAMPLE_FRAC, max_rows=VALID_MAX_ROWS,
        batch_size=VALID_BATCH, seed=SEED
    )
    if dvalid is None:
        print("[VALID] 샘플 생성 실패 → early_stopping 비활성")
        evals_list = [(dtrain, "train")]
        es_rounds = None
        cb = TQDMCallback(NUM_BOOST_ROUND, eval_name="train", metrics=("logloss","auc"))
    else:
        print(f"[VALID] dvalid rows = {used_valid:,}")
        evals_list = [(dtrain, "train"), (dvalid, "valid")]
        es_rounds = EARLY_STOP_ROUNDS
        cb = TQDMCallback(NUM_BOOST_ROUND, eval_name="valid", metrics=("logloss","auc"))

    # === 파라미터 ===
    params = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss","auc"],
        "tree_method": TREE_METHOD,
        "max_depth": 8,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "enable_categorical": True,
        "seed": SEED,
        "verbosity": 1,
    }
    if PREDICTOR:
        params["predictor"] = PREDICTOR

    # === 학습 ===
    print("[TRAIN] start training …")
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        evals=evals_list,
        early_stopping_rounds=es_rounds,   # ★ 여기! params가 아니라 train 인자로 넣어야 동작
        callbacks=[cb],
        verbose_eval=False,
    )
    best_iter = getattr(bst, "best_iteration", None)
    print(f"[TRAIN] best_iteration = {best_iter if best_iter is not None else 'N/A'}")
    print(f"[TRAIN] skipped batches={train_iter.skipped_batches}, rows={train_iter.skipped_rows}")

    # === 온도 스케일링(검증셋 기준) ===
    T = None
    iter_range = (0, best_iter + 1) if best_iter is not None else None
    if dvalid is not None and USE_TEMP_SCALING:
        p_valid = bst.predict(dvalid, iteration_range=iter_range) if iter_range else bst.predict(dvalid)
        T = calibrate_temperature(y_valid, p_valid)
        print(f"[CALIB] Temperature T = {T:.4f}")

    # === 내부 검증(리더보드 지표 근사; 온도 보정 적용하여 리포트) ===
    print("[VALID] computing AP/WLL/Score on train-sample …")
    valid_metrics = eval_on_validation(
        bst, train_ds, features, label_col="clicked",
        sample_frac=VALID_SAMPLE_FRAC, max_rows=VALID_MAX_ROWS,
        batch_size=VALID_BATCH, seed=SEED, iteration_range=iter_range, T=T
    )
    ap, wll, score, used = valid_metrics["AP"], valid_metrics["WLL"], valid_metrics["Score"], valid_metrics["Used"]
    if ap is not None:
        print(f"[VALID] Used={used:,} rows | AP={ap:.6f} | WLL={wll:.6f} | Score=0.5*AP + 0.5*(1/(1+WLL)) = {score:.6f}")
    else:
        print("[VALID] Not enough data to compute metrics.")

    model_path = os.path.join(save_path, "xgb_stream_once.json")
    bst.save_model(model_path)
    print("[TRAIN] saved:", model_path)

    # === 예측 ===
    print("[PRED] streaming predict …")
    total_test = count_rows(test_ds)
    today = datetime.datetime.now().strftime('%Y%m%d')
    score_str = ("nan" if (valid_metrics["Score"] is None or np.isnan(valid_metrics["Score"])) else f"{valid_metrics['Score']:.4f}").replace('.', '_')
    out_csv = os.path.join(save_path, f"{VER}_score_{score_str}_submission_{today}.csv")

    skipped_pred_batches = 0
    skipped_pred_rows = 0

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ID","clicked"])
        sc = test_ds.scanner(columns=["ID"] + features, batch_size=TEST_BATCH)
        with tqdm(total=total_test, unit="rows", desc="[PRED] rows") as pbar:
            for b in sc.to_batches():
                n = b.num_rows
                try:
                    tbl = pa.Table.from_batches([b]); df  = tbl.to_pandas()
                    ids = df.pop("ID").astype(str)
                    df = to_cat(df, CATS); df = to_int16_safe(df, INTS); df = to_f32(df)
                    df = enforce_xgb_dtypes(df, CATS, INTS)

                    dtest = xgb.DMatrix(df, enable_categorical=True)
                    p = bst.predict(dtest, iteration_range=iter_range) if iter_range else bst.predict(dtest)
                    if T is not None:
                        p = apply_temperature(p, T)

                    w.writerows(zip(ids.tolist(), p.tolist()))
                except Exception:
                    if not SILENCE_ERRORS:
                        raise
                    try:
                        id_idx = b.schema.get_field_index("ID")
                        ids_arr = b.column(id_idx).to_numpy(zero_copy_only=False)
                        id_strs = [str(x) for x in ids_arr]
                    except Exception:
                        id_strs = [""] * n
                    w.writerows(zip(id_strs, [0.5]*n))
                    skipped_pred_batches += 1
                    skipped_pred_rows += n
                finally:
                    pbar.update(n)
                    try:
                        del tbl, df
                    except Exception:
                        pass
                    del b
                    gc.collect()

    print(f"[PRED] skipped batches={skipped_pred_batches}, rows={skipped_pred_rows}")
    print(f"[DONE] wrote {out_csv}")
    print(f"[TIME] total {(time.time()-t0):.1f}s")

    return {"metrics": valid_metrics, "out_csv": out_csv, "model_path": model_path, "seed": SEED, "T": T}

if __name__ == "__main__":
    result = main()

# ===== 실행 로그 저장 =====
with open(os.path.join(save_path, f"(LOG)model_{VER}.txt"), "a") as f:
    f.write(f"{VER}\n<SEED :{result['seed']}>\n")
    if result["metrics"]["AP"] is not None:
        f.write(f"AP={result['metrics']['AP']:.6f}, WLL={result['metrics']['WLL']:.6f}, Score={result['metrics']['Score']:.6f}\n")
    if result["T"] is not None:
        f.write(f"Temperature T={result['T']:.6f}\n")
    f.write("="*40 + "\n")

with open(f"./Toss/_out/(LOG)model.txt", "a") as f:
    f.write(f"{VER}\n<SEED :{result['seed']}>\n")
    if result["metrics"]["AP"] is not None:
        f.write(f"AP={result['metrics']['AP']:.6f}, WLL={result['metrics']['WLL']:.6f}, Score={result['metrics']['Score']:.6f}\n")
    if result["T"] is not None:
        f.write(f"Temperature T={result['T']:.6f}\n")
    f.write("="*40 + "\n")
