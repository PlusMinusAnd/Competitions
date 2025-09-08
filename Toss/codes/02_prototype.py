# /home/pc/Study/Project/Toss/codes/run_once_xgb_stream.py
# 분할 Parquet(20만행 단위)에서 배치 스트리밍 학습/예측 + tqdm 진행률 + 에러 무시 옵션

import os, gc, csv, time, math, warnings
warnings.filterwarnings("ignore")

import json
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

#endregion IMPORT

SEED = seed_state["seed"]
print(f"[Current Run SEED]: {SEED}")

#region BASIC OPTIONS
seed_state["seed"] += 1
with open(seed_file, "w") as f:
    json.dump(seed_state, f)
import datetime
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
    def __init__(self, total_rounds: int, desc: str = "[TRAIN] boosting"):
        self.pbar = tqdm(total=total_rounds, desc=desc, unit="round")

    def after_iteration(self, model, epoch: int, evals_log):
        self.pbar.update(1)
        return False  # 학습 계속

    def after_training(self, model):
        self.pbar.close()
        return model

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
        "seed": 73,
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
        evals=[(dtrain, "train")],  # metric 로그 활성화
        callbacks=[TQDMCallback(NUM_BOOST_ROUND)]
    )
    print(f"[TRAIN] skipped batches={train_iter.skipped_batches}, rows={train_iter.skipped_rows}")

    model_path = os.path.join(save_path, "xgb_stream_once.json")
    bst.save_model(model_path)
    print("[TRAIN] saved:", model_path)

    # === 예측 ===
    print("[PRED] streaming predict …")
    
    
    today = datetime.datetime.now().strftime('%Y%m%d')
    
    total_test = count_rows(test_ds)
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
                    # 메모리 정리
                    try:
                        del tbl, df
                    except Exception:
                        pass
                    del b
                    gc.collect()

    print(f"[PRED] skipped batches={skipped_pred_batches}, rows={skipped_pred_rows}")
    print(f"[DONE] wrote {out_csv}")
    print(f"[TIME] total {(time.time()-t0):.1f}s")

if __name__ == "__main__":
    main()

with open(save_path + f"(LOG)model_{VER}.txt", "a") as f:
    f.write(f"{VER}\n")
    f.write(f"<SEED :{SEED}>\n")

    f.write("="*40 + "\n")