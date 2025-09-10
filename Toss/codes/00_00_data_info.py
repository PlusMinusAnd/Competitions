# =========================
# 대용량 Parquet 통계 생성기 (Top-to-Bottom, fixed)
# =========================
import os, math, gc, json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
from collections import Counter

# ---------- 설정 ----------
DATA_PATH     = "./Project/Toss/test.parquet"   # train.parquet
OUT_DIR       = "./Project/Toss/feature_stats"   # 결과 저장 디렉토리
BATCH_ROWS    = 200_000                          # 배치 크기
SAMPLE_FRAC   = 0.002                            # 분포/백분위 추정용 샘플 비율(0.2%~0.5% 권장)
HIST_BINS     = 50                               # 히스토그램 구간 수
TARGET_COL    = "clicked"
ID_COL        = "ID"                             # train에는 없음
TRACK_CATS    = ["gender","age_group","day_of_week","hour","inventory_id","l_feat_14"]
TOPK_CAT      = 50

os.makedirs(OUT_DIR, exist_ok=True)
np.random.seed(42)

# ---------- 준비 ----------
pf = pq.ParquetFile(DATA_PATH)

# 1배치 읽어 타입/컬럼 파악
first_rb  = next(pf.iter_batches(batch_size=1))
first_df  = first_rb.to_pandas()

# (A) 첫 배치에서 '일관된 유니크 컬럼명' 생성(위치 기반 매핑)
seen = {}
unique_cols = []
for col in list(first_df.columns):
    if col not in seen:
        seen[col] = 0
        unique_cols.append(col)
    else:
        seen[col] += 1
        unique_cols.append(f"{col}__dup{seen[col]}")

# 위치→유니크명 매핑 (이후 모든 배치에 동일 적용)
col_map_by_pos = {i: unique_cols[i] for i in range(len(unique_cols))}
first_df.columns = unique_cols

# 타입/리스트 확정
all_cols  = list(first_df.columns)
num_cols  = list(first_df.select_dtypes(include=[np.number]).columns)
obj_cols  = [c for c in all_cols if c not in num_cols]
cat_track = [c for c in TRACK_CATS if c in all_cols]

# 타깃/ID는 피처 목록에서 제거
if TARGET_COL in num_cols:
    num_cols.remove(TARGET_COL)
if ID_COL in num_cols:
    num_cols.remove(ID_COL)
if TARGET_COL in obj_cols:
    obj_cols.remove(TARGET_COL)
if ID_COL in obj_cols:
    obj_cols.remove(ID_COL)

print("[INFO] 총 컬럼:", len(all_cols), " / 수치형:", len(num_cols), " / 기타(object 등):", len(obj_cols))
print("[INFO] Top-N 카테고리 추적:", cat_track)

# 수치형 통계 누적 버퍼(정확치)
stats_cnt   = {c:0 for c in num_cols}
stats_sum   = {c:0.0 for c in num_cols}
stats_sumsq = {c:0.0 for c in num_cols}
stats_min   = {c: np.inf for c in num_cols}
stats_max   = {c:-np.inf for c in num_cols}
stats_nulls = {c:0 for c in num_cols}

# 샘플 버퍼(분포/백분위 근사용)
sample_chunks = []

# 범주형(추적 대상만) TopK 카운터
cat_counters = {c: Counter() for c in cat_track}

# ---------- 스캔 ----------
total_rows = 0
pbar = tqdm(pf.iter_batches(batch_size=BATCH_ROWS), desc="Scanning", unit="batch")
for rb in pbar:
    df = rb.to_pandas()  # 필요시 columns=... 로 부분 열만 읽기 가능

    # (B) 모든 배치에 '같은' 유니크 컬럼명 적용 (첫 배치와 동일한 위치 매핑)
    if df.shape[1] == len(col_map_by_pos):
        df.columns = [col_map_by_pos[i] for i in range(df.shape[1])]
    else:
        # 비정상 케이스 안전망: 배치 내에서 자체 유니크화
        seen_b = {}
        uniq_b = []
        for col in list(df.columns):
            if col not in seen_b:
                seen_b[col] = 0
                uniq_b.append(col)
            else:
                seen_b[col] += 1
                uniq_b.append(f"{col}__dup{seen_b[col]}")
        df.columns = uniq_b

    n = len(df)
    total_rows += n

    # ---- 수치형 정확 통계 누적 ----
    if num_cols:
        num_df = df[num_cols]
        null_counts = num_df.isna().sum().to_dict()
        for c, v in null_counts.items():
            stats_nulls[c] += int(v)

        num_np = num_df.to_numpy(dtype="float64", copy=False)
        col_sum   = np.nansum(num_np, axis=0)
        col_sumsq = np.nansum(num_np*num_np, axis=0)
        col_cnt   = (~np.isnan(num_np)).sum(axis=0)
        col_min   = np.nanmin(num_np, axis=0)
        col_max   = np.nanmax(num_np, axis=0)

        for idx, c in enumerate(num_cols):
            stats_cnt[c]   += int(col_cnt[idx])
            stats_sum[c]   += float(col_sum[idx])
            stats_sumsq[c] += float(col_sumsq[idx])
            if not np.isnan(col_min[idx]):
                stats_min[c] = min(stats_min[c], float(col_min[idx]))
            if not np.isnan(col_max[idx]):
                stats_max[c] = max(stats_max[c], float(col_max[idx]))

    # ---- 샘플 추출(분포/백분위 근사) ----
    if SAMPLE_FRAC > 0 and n > 0:
        k = max(1, int(n * SAMPLE_FRAC))
        sample_idx = np.random.choice(n, size=k, replace=False)

        # 중복 방지: num_cols만 담기 (TARGET_COL은 통계표에서 제외, 따로 CTR 계산)
        take_cols = [c for c in num_cols if c in df.columns]
        sample_chunk = df.iloc[sample_idx][take_cols]
        sample_chunks.append(sample_chunk)

    # ---- 범주형 TopN 카운터 업데이트(추적 대상만) ----
    for c in cat_track:
        vc = df[c].astype("object").value_counts(dropna=False)
        for key, cnt in vc.iloc[:TOPK_CAT].items():
            cat_counters[c][key] += int(cnt)

    # 메모리 정리
    del df, rb
    gc.collect()

pbar.close()
print(f"[INFO] 총 행 수: {total_rows:,}")

# ---------- 샘플 합치기 ----------
if sample_chunks:
    sample_df = pd.concat(sample_chunks, axis=0, ignore_index=True)
    # 혹시 모를 중복 컬럼 제거(첫 배치 매핑을 쓰므로 보통 필요 없음)
    sample_df = sample_df.loc[:, ~sample_df.columns.duplicated()]
    del sample_chunks
    gc.collect()
else:
    sample_df = pd.DataFrame(columns=num_cols)

print("[INFO] 샘플 크기:", sample_df.shape)

# ---------- 수치형 요약 테이블 만들기 ----------
rows = []
quantiles = [0.005, 0.01, 0.25, 0.50, 0.75, 0.99, 0.995]

for c in num_cols:  # LABEL 제외
    cnt   = stats_cnt[c]
    nulls = stats_nulls[c]
    total = cnt + nulls
    if cnt > 0:
        mean = stats_sum[c] / cnt
        var  = max(0.0, stats_sumsq[c] / cnt - mean*mean)   # 모집단 분산
        std  = math.sqrt(var)
    else:
        mean = np.nan
        std  = np.nan

    # 샘플 기반 백분위/중앙값 (넘파이로 일관 계산 → KeyError 방지)
    qs = {q: np.nan for q in quantiles}
    if c in sample_df.columns:
        vals = pd.to_numeric(sample_df[c], errors="coerce").to_numpy()
        vals = vals[~np.isnan(vals)]
        if vals.size > 0:
            q_vals = np.nanquantile(vals, quantiles, method="linear")
            qs = dict(zip(quantiles, q_vals))

    row = {
        "column": c,
        "count_nonnull": cnt,
        "count_null": nulls,
        "null_ratio": nulls / total if total > 0 else np.nan,
        "mean": mean,
        "std": std,
        "min": stats_min[c] if stats_min[c] != np.inf else np.nan,
        "q_0.5%":  qs[0.005],
        "q_1%":    qs[0.01],
        "median":  qs[0.50],
        "q_25%":   qs[0.25],
        "q_75%":   qs[0.75],
        "q_99%":   qs[0.99],
        "q_99.5%": qs[0.995],
        "max": stats_max[c] if stats_max[c] != -np.inf else np.nan,
        "IQR": (qs[0.75] - qs[0.25]) if (pd.notna(qs[0.75]) and pd.notna(qs[0.25])) else np.nan
    }
    rows.append(row)

stats_numeric = pd.DataFrame(rows).sort_values("column").reset_index(drop=True)
stats_numeric.to_csv(os.path.join(OUT_DIR, "stats_numeric.csv"), index=False)
print("[SAVE] 수치형 요약 ->", os.path.join(OUT_DIR, "stats_numeric.csv"))

# ---------- 수치형 분포(히스토그램) 파일 저장 ----------
HIST_DIR = os.path.join(OUT_DIR, "histograms")
os.makedirs(HIST_DIR, exist_ok=True)

for c in num_cols:
    if c not in sample_df.columns:
        continue
    vals = pd.to_numeric(sample_df[c], errors="coerce").dropna().to_numpy()
    if vals.size == 0:
        continue
    counts, edges = np.histogram(vals, bins=HIST_BINS)
    hist_df = pd.DataFrame({
        "column": [c]*(len(edges)-1),
        "bin_left": edges[:-1],
        "bin_right": edges[1:],
        "count": counts
    })
    hist_df.to_csv(os.path.join(HIST_DIR, f"hist_{c}.csv"), index=False)

print("[SAVE] 히스토그램 CSVs ->", HIST_DIR)

# ---------- 범주형(추적 대상) Top-N 저장 ----------
CAT_DIR = os.path.join(OUT_DIR, "categoricals")
os.makedirs(CAT_DIR, exist_ok=True)

for c in cat_counters:
    cnts = cat_counters[c]
    top_items = cnts.most_common(TOPK_CAT)
    if len(top_items) == 0:
        continue
    tot = sum(cnts.values())
    cat_df = pd.DataFrame(top_items, columns=[c, "count"])
    cat_df["ratio"] = cat_df["count"] / tot if tot > 0 else np.nan
    cat_df.to_csv(os.path.join(CAT_DIR, f"{c}_top{TOPK_CAT}.csv"), index=False)

print("[SAVE] 범주형 TopN ->", CAT_DIR)

# ---------- 타깃(클릭) 요약(정확 CTR) ----------
if TARGET_COL in all_cols:
    print("[INFO] 클릭률 계산 중(정확) ...")
    click_sum = 0.0
    total_lab = 0
    # TARGET_COL만 재스캔 (메모리 매우 적게 사용)
    for rb in tqdm(pf.iter_batches(batch_size=BATCH_ROWS, columns=[TARGET_COL]), desc="CTR", unit="batch"):
        y = rb.to_pandas()
        # 위치 매핑 적용
        if y.shape[1] == len(col_map_by_pos):
            y.columns = [col_map_by_pos[i] for i in range(y.shape[1])]
        else:
            # 단일 컬럼만 읽었으므로 일반적으로 아래 분기
            pass
        if TARGET_COL in y.columns:
            valid = pd.to_numeric(y[TARGET_COL], errors="coerce").dropna().to_numpy()
            click_sum += float(valid.sum())
            total_lab += int(valid.size)
        elif f"{TARGET_COL}__dup1" in y.columns:
            # 혹시나 타깃이 중복 컬럼으로 존재할 때(비현실적이지만 안전망)
            valid = pd.to_numeric(y[f"{TARGET_COL}__dup1"], errors="coerce").dropna().to_numpy()
            click_sum += float(valid.sum())
            total_lab += int(valid.size)

    ctr = click_sum / total_lab if total_lab > 0 else np.nan
    with open(os.path.join(OUT_DIR, "target_summary.json"), "w") as f:
        json.dump({"clicked_positives": click_sum, "label_rows": total_lab, "CTR": ctr},
                  f, ensure_ascii=False, indent=2)
    print("[SAVE] target_summary.json ->", os.path.join(OUT_DIR, "target_summary.json"))

print("[DONE]")
