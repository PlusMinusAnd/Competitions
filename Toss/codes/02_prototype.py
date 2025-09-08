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
VER = "ver_proto"
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
            s = pd.to_numeric(df[c], errors="coerce").fillna(-1).astype(np.int16)  # np.int16 (nullable X)
            df[c] = s
    return df

def to_f32(df):
    float_cols = df.select_dtypes(include=["float", "float32", "float64"]).columns
    if len(float_cols) > 0:
        df[float_cols] = df[float_cols].astype("float32")
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

# === 검증 샘플링 설정 (리더보드 지표용) ===
VALID_SAMPLE_FRAC = 0.10     # train에서 10% 랜덤 샘플
VALID_MAX_ROWS    = 400_000  # 최대 40만 행까지만 사용 (메모리 안전)
VALID_BATCH       = 150_000  # 스캔 배치 크기

USE_GPU = gpu_ok()
if USE_GPU:
    TREE_METHOD = "gpu_hist"
    PREDICTOR   = "gpu_predictor"
else:
    TREE_METHOD = "hist"
    PREDICTOR   = None  # CPU
    print("[GPU] 사용 불가. CPU(hist)로 전환합니다.")

SILENCE_ERRORS = True           # 배치 변환 오류 발생 시 조용히 건너뛰기

CATS = ("gender","age_group","inventory_id")
INTS = ("day_of_week","hour")
EXCLUDE = {"clicked","ID","seq"}
DROP_VIRTUAL = {"__fragment_index","__batch_index","__last_in_fragment","__filename"}

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
    print(f"[WARN] columns present in train but missing in test (ignored): {missing_in_test[:8]}{' ...' if len(missing_in_test)>8 else ''}")

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

        # 초기 스캐너/이터레이터 구성
        cols = self.feats + ([label_col] if label_col else [])
        self.scanner = dset.scanner(columns=cols, batch_size=batch_size)
        self._it = iter(self.scanner.to_batches())

    def reset(self):
        # Scanner 재생성(안정성↑)
        cols = self.feats + ([self.label] if self.label else [])
        self.scanner = self.dset.scanner(columns=cols, batch_size=self.batch_size)
        self._it = iter(self.scanner.to_batches())

    def _process_batch(self, batch):
        tbl = pa.Table.from_batches([batch])
        df  = tbl.to_pandas()

        y = None
        if self.label and self.label in df:
            y = df.pop(self.label).astype("int8")

        df = to_cat(df, CATS)
        df = to_int16_safe(df, INTS)  # np.int16 + fillna(-1)
        df = to_f32(df)
        return df, y

    def next(self, input_data):
        while True:
            try:
                batch = next(self._it)
            except StopIteration:
                self.pbar.close()
                return 0

        # ... (중간 생략 없음)
            n = batch.num_rows
            self.pbar.update(n)

            try:
                df, y = self._process_batch(batch)
                input_data(data=df, label=y)
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
        # 최신 metric을 진행바 postfix로만 업데이트
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
    """Sklearn 정의와 동일한 AP: 양성 샘플 위치의 precision 평균."""
    y_true = y_true.astype(np.int8)
    if y_true.sum() == 0:
        return 0.0
    order = np.argsort(-y_score, kind="mergesort")  # 안정 정렬
    y_sorted = y_true[order]
    cum_tp = np.cumsum(y_sorted)
    precision = cum_tp / (np.arange(y_sorted.size) + 1)
    ap = precision[y_sorted == 1].mean()
    return float(ap)

def weighted_logloss_5050(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-15) -> float:
    """WLL: 0/1 클래스 기여를 50:50로 맞춘 가중 LogLoss.
       -0.5*E_pos[log(p)] -0.5*E_neg[log(1-p)]"""
    y_true = y_true.astype(np.int8)
    pos_mask = (y_true == 1)
    neg_mask = ~pos_mask
    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()
    if n_pos == 0 or n_neg == 0:
        # 극단 케이스 방어: 일반 logloss로 대체
        p = np.clip(y_prob, eps, 1 - eps)
        return float(-np.mean(y_true*np.log(p) + (1-y_true)*np.log(1-p)))
    p_pos = np.clip(y_prob[pos_mask], eps, 1 - eps)
    p_neg = np.clip(y_prob[neg_mask], eps, 1 - eps)
    wll = -0.5*(np.log(p_pos).mean()) - 0.5*(np.log(1 - p_neg).mean())
    return float(wll)

def eval_on_validation(bst, dset: ds.Dataset, feature_cols, label_col="clicked",
                       sample_frac=VALID_SAMPLE_FRAC, max_rows=VALID_MAX_ROWS,
                       batch_size=VALID_BATCH, seed=SEED):
    """train에서 랜덤 샘플을 뽑아 지표 계산"""
    rs = np.random.RandomState(seed)
    y_list = []
    p_list = []
    total = count_rows(dset)
    used = 0
    sc = dset.scanner(columns=feature_cols + [label_col], batch_size=batch_size)
    with tqdm(total=total, unit="rows", desc="[VALID] scan") as pbar:
        for b in sc.to_batches():
            tbl = pa.Table.from_batches([b])
            df  = tbl.to_pandas()
            pbar.update(b.num_rows)

            # 샘플링
            m = rs.rand(len(df)) < sample_frac
            if not m.any():
                del tbl, df, b
                continue
            sdf = df.loc[m].copy()

            y = sdf.pop(label_col).astype(np.int8).values
            sdf = to_cat(sdf, CATS)
            sdf = to_int16_safe(sdf, INTS)
            sdf = to_f32(sdf)

            dmx = xgb.DMatrix(sdf, enable_categorical=True)
            p = bst.predict(dmx)

            y_list.append(y)
            p_list.append(p)

            used += len(y)
            del tbl, df, sdf, dmx, b, y, p
            gc.collect()
            if used >= max_rows:
                break

    if used == 0:
        return {"AP": None, "WLL": None, "Score": None, "Used": 0}

    y_true = np.concatenate(y_list)
    y_prob = np.concatenate(p_list)
    ap  = average_precision_simple(y_true, y_prob)
    wll = weighted_logloss_5050(y_true, y_prob)
    score = 0.5*ap + 0.5*(1.0/(1.0 + wll))
    return {"AP": ap, "WLL": wll, "Score": score, "Used": used}

# ===== 메인 =====
def main():
    t0 = time.time()

    # (선택) 현재 TREE_METHOD 간단 테스트
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

    dtrain = xgb.QuantileDMatrix(
        train_iter,
        enable_categorical=True
    )

    # === 파라미터 ===
    params = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss","auc"],
        "tree_method": TREE_METHOD,          # "gpu_hist" / "hist"
        "max_depth": 8,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "enable_categorical": True,
        "scale_pos_weight": max(1.0, neg / max(1, pos)),
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
        evals=[(dtrain, "train")],
        callbacks=[TQDMCallback(NUM_BOOST_ROUND, eval_name="train", metrics=("logloss","auc"))],
        verbose_eval=False,
    )
    print(f"[TRAIN] skipped batches={train_iter.skipped_batches}, rows={train_iter.skipped_rows}")

    # === 내부 검증(리더보드 지표 근사) ===
    print("[VALID] computing AP/WLL/Score on train-sample …")
    valid_metrics = eval_on_validation(bst, train_ds, features, label_col="clicked",
                                       sample_frac=VALID_SAMPLE_FRAC, max_rows=VALID_MAX_ROWS,
                                       batch_size=VALID_BATCH, seed=SEED)
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
    out_csv = os.path.join(save_path, f"{VER}_{today}submission.csv")

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
                    tbl = pa.Table.from_batches([b])
                    df  = tbl.to_pandas()

                    ids = df.pop("ID").astype(str)
                    df = to_cat(df, CATS)
                    df = to_int16_safe(df, INTS)
                    df = to_f32(df)

                    dtest = xgb.DMatrix(df, enable_categorical=True)
                    p = bst.predict(dtest)

                    w.writerows(zip(ids.tolist(), p.tolist()))
                except Exception:
                    if not SILENCE_ERRORS:
                        raise
                    # fallback: 해당 배치 전체 0.5로 채워 행 보존
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

    return {"metrics": valid_metrics, "out_csv": out_csv, "model_path": model_path, "seed": SEED}

if __name__ == "__main__":
    result = main()

# ===== 실행 로그 저장 =====
with open(os.path.join(save_path, f"(LOG)model_{VER}.txt"), "a") as f:
    f.write(f"{VER}\n<SEED :{result['seed']}>\n")
    if result["metrics"]["AP"] is not None:
        f.write(f"AP={result['metrics']['AP']:.6f}, WLL={result['metrics']['WLL']:.6f}, Score={result['metrics']['Score']:.6f}\n")
    f.write("="*40 + "\n")
