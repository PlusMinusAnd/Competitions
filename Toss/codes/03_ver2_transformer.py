# /home/pc/Study/Project/Toss/codes/run_once_xgb_transformer.py
# 분할 Parquet(20만행 단위)에서 스트리밍 학습/예측 + tqdm + EarlyStopping + AP/WLL
# Model: FT-Transformer style for tabular (categorical + numeric tokens)

import os, gc, csv, time, math, warnings, json, datetime
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

# ===== 설정 =====
VER = "ver2_transformer"  # transformer 버전
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

SEED = seed_state["seed"]; print(f"[Current Run SEED]: {SEED}")
seed_state["seed"] += 1
with open(seed_file, "w") as f:
    json.dump(seed_state, f)

save_path = f'{OUT_DIR}/{SEED}_submission_{VER}/'
os.makedirs(save_path, exist_ok=True)

# ===== 데이터/피처 정의 =====
CATS = ("gender","age_group","inventory_id")
INTS = ("day_of_week","hour")   # 정수형 수치 피처
EXCLUDE = {"clicked","ID","seq"}
DROP_VIRTUAL = {"__fragment_index","__batch_index","__last_in_fragment","__filename"}

# ===== 학습/검증/예측 배치 =====
SCAN_TRAIN_BATCH = 120_000   # Arrow 스캔 배치(큰 배치)
SCAN_TEST_BATCH  = 150_000
GPU_BATCH        = 2048    # GPU 미니배치(이 값으로 쪼개서 학습)
EPOCHS           = 20
LR               = 2e-4
WEIGHT_DECAY     = 1e-4
EARLY_STOP_ROUNDS = 3        # valid score 미개선 epoch 횟수
GRAD_ACCUM_STEPS = 1         # 필요시 늘리면 됨

# ===== 모델 하이퍼 =====
D_MODEL   = 256
N_HEAD    = 8
N_LAYERS  = 4
FFN_DIM   = 1024
DROPOUT   = 0.1

# ===== 검증 샘플링 =====
VALID_SAMPLE_FRAC = 0.20
VALID_MAX_ROWS    = 400_000
VALID_BATCH       = 150_000

# ===== 다운샘플링(음성 언더샘플) =====
USE_DS = False
NEG_KEEP_RATIO = 0.4

# ===== 레어 카테고리 버킷팅 =====
RARE_MIN_COUNT = {"inventory_id": 3, "gender": 1, "age_group": 3}          # 이보다 적게 등장하면 "__rare__"
MAX_TOKENS_PER_CAT = {"inventory_id": 2048}    # (선택) 상위 k개만 쓰고 나머지 rare로

# ===== 온도 스케일링(예측 보정) =====
USE_TEMP_SCALING = True

# ===== 장치 =====
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED); np.random.seed(SEED)
torch.cuda.manual_seed(SEED)

# ===== 공통 유틸 =====
def dataset_and_cols(path: str):
    dset = ds.dataset(path, format="parquet")
    cols = [c for c in dset.schema.names if c not in DROP_VIRTUAL]
    return dset, cols

def count_rows(dset: ds.Dataset) -> int:
    return dset.count_rows()

def label_counts(dset: ds.Dataset, batch=150_000):
    total = count_rows(dset)
    sc = dset.scanner(columns=["clicked"], batch_size=batch)
    pos = 0; tot = 0
    with tqdm(total=total, unit="rows", desc="[COUNT] labels") as pbar:
        for b in sc.to_batches():
            arr = b.column(0).to_numpy(zero_copy_only=False)
            n = arr.size
            tot += n; pos += int(arr.sum()); pbar.update(n)
    return pos, tot - pos

def average_precision_simple(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int8)
    if y_true.sum() == 0: return 0.0
    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]
    cum_tp = np.cumsum(y_sorted)
    precision = cum_tp / (np.arange(y_sorted.size) + 1)
    ap = precision[y_sorted == 1].mean()
    return float(ap)

def weighted_logloss_5050(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-15) -> float:
    y_true = y_true.astype(np.int8)
    pos_mask = (y_true == 1); neg_mask = ~pos_mask
    n_pos = pos_mask.sum(); n_neg = neg_mask.sum()
    if n_pos == 0 or n_neg == 0:
        p = np.clip(y_prob, eps, 1 - eps)
        return float(-np.mean(y_true*np.log(p) + (1-y_true)*np.log(1-p)))
    p_pos = np.clip(y_prob[pos_mask], eps, 1 - eps)
    p_neg = np.clip(y_prob[neg_mask], eps, 1 - eps)
    wll = -0.5*(np.log(p_pos).mean()) - 0.5*(np.log(1 - p_neg).mean())
    return float(wll)

def sigmoid_np(x): return 1.0/(1.0 + np.exp(-x))

# ===== 1) 학습/예측 준비: 스키마, 피처, 통계 =====
train_ds, train_cols = dataset_and_cols(TRAIN_DIR)
test_ds,  test_cols  = dataset_and_cols(TEST_DIR)
features = [c for c in train_cols if (c in test_cols) and (c not in EXCLUDE)]
print(f"[INFO] #features = {len(features)}")
missing_in_test = [c for c in train_cols if c not in test_cols and c not in EXCLUDE]
if missing_in_test:
    print(f"[WARN] train-only cols ignored: {missing_in_test[:8]}{' ...' if len(missing_in_test)>8 else ''}")

print("[INFO] counting labels …")
pos, neg = label_counts(train_ds)
pi = pos / max(1, pos + neg)
print(f"[INFO] pos={pos:,}, neg={neg:,}, pi={pi:.6f}")

# ===== 2) 카테고리 vocab 빌드 + 레어 버킷팅 =====
from collections import Counter, defaultdict
def build_categorical_vocab(
    dset,
    cat_cols,
    min_count=RARE_MIN_COUNT,
    batch=300_000,
    max_tokens_per_col=MAX_TOKENS_PER_CAT,
):
    """
    min_count, max_tokens_per_col 둘 다
    - int 또는
    - {"col_name": int} dict
    형태 모두 지원.
    """
    vocab = {}
    for col in cat_cols:
        # 1) 컬럼별 min_count 결정
        if isinstance(min_count, dict):
            mc = int(min_count.get(col, 20))   # 디폴트 20
        else:
            mc = int(min_count)

        # 2) 전체 빈도 카운트
        cnt = Counter()
        sc = dset.scanner(columns=[col], batch_size=batch)
        for b in sc.to_batches():
            s = pa.Table.from_batches([b]).to_pandas()[col].astype(str)
            cnt.update(s.values.tolist())

        # 3) 빈도 필터링 + 정렬
        items = [(k, v) for k, v in cnt.items() if v >= mc]
        items.sort(key=lambda x: (-x[1], x[0]))

        # 4) 컬럼별 top-k 제한 (옵션)
        if isinstance(max_tokens_per_col, dict):
            mt = max_tokens_per_col.get(col, None)
        else:
            mt = max_tokens_per_col

        if mt is not None:
            mt = int(mt)
            if mt <= 0:
                items = []
            else:
                items = items[:mt]

        # 5) vocab 구성: 0="__unk__", 1="__rare__", 그 뒤로 상위 토큰
        itos = ["__unk__", "__rare__"] + [k for k, _ in items]
        stoi = {s: i for i, s in enumerate(itos)}
        vocab[col] = {"itos": itos, "stoi": stoi, "min_count": mc, "topk": mt}

        print(f"[VOCAB] {col}: {len(itos)} tokens (>= {mc}"
              f"{', topk='+str(mt) if mt is not None else ''})")
    return vocab

print("[INFO] building categorical vocabs …")
VOCAB = build_categorical_vocab(train_ds, CATS)

def map_cats(df: pd.DataFrame, vocab):
    out = []
    for col in CATS:
        if col in df:
            s = df[col].astype(str)
        else:
            s = pd.Series([""], index=df.index)
        stoi = vocab[col]["stoi"]
        # rare 처리: stoi에 없으면 "__rare__"로
        idx = s.map(lambda x: stoi.get(x, stoi["__rare__"])).astype(np.int32).values
        out.append(idx)
    return np.stack(out, axis=1)  # [B, n_cat]

# ===== 3) 수치 통계(표준화) =====
def estimate_numeric_stats(dset, num_cols, sample_frac=0.10, max_rows=600_000, batch=200_000, seed=SEED):
    rs = np.random.RandomState(seed)
    sums = defaultdict(float); sqs = defaultdict(float); cnts = defaultdict(int)
    sc = dset.scanner(columns=num_cols, batch_size=batch)
    used = 0
    with tqdm(total=count_rows(dset), unit="rows", desc="[NUM] stats") as pbar:
        for b in sc.to_batches():
            df = pa.Table.from_batches([b]).to_pandas()
            pbar.update(b.num_rows)
            m = rs.rand(len(df)) < sample_frac
            if not m.any(): 
                del df,b; continue
            sdf = df.loc[m]
            for c in num_cols:
                x = pd.to_numeric(sdf[c], errors="coerce").fillna(0.0).astype(np.float32).values
                sums[c] += float(x.sum()); sqs[c] += float((x*x).sum()); cnts[c] += x.size
            used += len(sdf); del sdf, df, b
            if used >= max_rows: break
    means = {c: (sums[c]/max(1,cnts[c])) for c in num_cols}
    stds  = {c: float(np.sqrt(max(1e-6, sqs[c]/max(1,cnts[c]) - means[c]**2))) for c in num_cols}
    print("[NUM] stats ready.")
    return means, stds

NUM_COLS = [c for c in features if (c in INTS) or (c not in CATS)]
MEANS, STDS = estimate_numeric_stats(train_ds, NUM_COLS)

def map_nums(df: pd.DataFrame, means, stds):
    arrs = []
    for c in NUM_COLS:
        if c in df:
            x = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(np.float32).values
        else:
            x = np.zeros(len(df), dtype=np.float32)
        mu, sd = means.get(c, 0.0), stds.get(c, 1.0)
        sd = sd if sd > 0 else 1.0
        x = (x - mu) / sd
        arrs.append(x.astype(np.float32))
    return np.stack(arrs, axis=1) if arrs else np.zeros((len(df),0), dtype=np.float32)  # [B, n_num]

# ===== 4) 모델 정의: FT-Transformer =====
class FTTransformer(nn.Module):
    def __init__(self, cat_cardinals, n_num, d_model=D_MODEL, n_head=N_HEAD, n_layers=N_LAYERS, ffn_dim=FFN_DIM, dropout=DROPOUT):
        super().__init__()
        self.n_cat = len(cat_cardinals); self.n_num = n_num
        # 각 카테고리 피처별 embedding
        self.cat_embeds = nn.ModuleList([nn.Embedding(card, d_model) for card in cat_cardinals])
        # 각 수치 피처를 token으로 투영
        self.num_linear = nn.Linear(1, d_model)
        # [CLS] 토큰
        self.cls = nn.Parameter(torch.randn(1,1,d_model)*0.02)
        # 트랜스포머 인코더
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=ffn_dim,
                                                   dropout=dropout, batch_first=True, activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)
        # 분류 헤드
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x_cat: torch.Tensor, x_num: torch.Tensor):
        # x_cat: [B, n_cat] (long), x_num: [B, n_num] (float32)
        B = x_cat.size(0)
        tokens = []

        # 1) 카테고리: 각 열 임베딩(B, d) -> stack해서 (B, n_cat, d)
        if self.n_cat > 0:
            cat_tok_list = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeds)]  # each (B, d)
            # (B, d) 리스트를 새 축(dim=1)으로 쌓아 (B, n_cat, d)
            cat_tokens = torch.stack(cat_tok_list, dim=1)
            tokens.append(cat_tokens)

        # 2) 수치: 스칼라 하나당 1개 토큰 → (B, n_num, d)
        if self.n_num > 0:
            num_tokens = self.num_linear(x_num.unsqueeze(-1))  # (B, n_num, 1) -> (B, n_num, d)
            tokens.append(num_tokens)

        if len(tokens) == 0:
            raise RuntimeError("No tokens to encode.")

        # 3) 토큰 합치기: (B, n_tokens, d)
        x = torch.cat(tokens, dim=1)

        # 4) [CLS] 붙이기: (B, 1+n_tokens, d)
        cls = self.cls.expand(B, 1, -1)
        x = torch.cat([cls, x], dim=1)

        # 5) 인코더 & 헤드
        x = self.encoder(x)
        cls_out = x[:, 0, :]
        logit = self.head(self.dropout(cls_out)).squeeze(-1)
        return logit

# ===== 5) 학습 루프 =====
def make_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

def compute_score(y_true, y_prob):
    ap  = average_precision_simple(y_true, y_prob)
    wll = weighted_logloss_5050(y_true, y_prob)
    score = 0.5*ap + 0.5*(1.0/(1.0 + wll))
    return ap, wll, score

def train_epoch(model, dset, optimizer):
    model.train()
    sc = dset.scanner(columns=["clicked"] + list(features), batch_size=SCAN_TRAIN_BATCH)
    loss_meter = 0.0; n_steps = 0; seen = 0
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE=="cuda"))
    with tqdm(total=count_rows(dset), unit="rows", desc="[TRAIN] rows") as pbar:
        for b in sc.to_batches():
            df = pa.Table.from_batches([b]).to_pandas()
            y = df.pop("clicked").astype("int8").values
            # 다운샘플링
            if USE_DS:
                neg_mask = (y==0)
                keep = np.ones_like(y, dtype=bool)
                if neg_mask.any():
                    rng = np.random.default_rng(SEED)
                    keep_neg = rng.random(neg_mask.sum()) < NEG_KEEP_RATIO
                    keep[neg_mask] = keep_neg
                df = df.loc[keep].reset_index(drop=True); y = y[keep]
            # 매핑
            x_cat = torch.from_numpy(map_cats(df, VOCAB)).long()
            x_num = torch.from_numpy(map_nums(df, MEANS, STDS)).float()
            y_t   = torch.from_numpy(y.astype(np.float32))
            # 미니배치로 쪼개서 학습
            for s in range(0, len(df), GPU_BATCH):
                e = min(len(df), s+GPU_BATCH)
                xb_cat = x_cat[s:e].to(DEVICE, non_blocking=True)
                xb_num = x_num[s:e].to(DEVICE, non_blocking=True)
                yb     = y_t[s:e].to(DEVICE, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=(DEVICE=="cuda")):
                    logit = model(xb_cat, xb_num)
                    loss = F.binary_cross_entropy_with_logits(logit, yb)
                scaler.scale(loss/GRAD_ACCUM_STEPS).backward()
                if ((n_steps+1) % GRAD_ACCUM_STEPS)==0:
                    scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                loss_meter += loss.item()* (e - s); n_steps += 1; seen += (e - s)
            pbar.update(b.num_rows)
            del df, b, x_cat, x_num, y_t; gc.collect()
    return loss_meter/max(1,seen)

@torch.no_grad()
def evaluate_sample(model, dset, sample_frac=VALID_SAMPLE_FRAC, max_rows=VALID_MAX_ROWS, batch=VALID_BATCH):
    model.eval()
    rs = np.random.RandomState(SEED)
    y_list=[]; p_list=[]; used=0
    sc = dset.scanner(columns=["clicked"] + list(features), batch_size=batch)
    with tqdm(total=count_rows(dset), unit="rows", desc="[VALID] scan") as pbar:
        for b in sc.to_batches():
            df = pa.Table.from_batches([b]).to_pandas()
            pbar.update(b.num_rows)
            m = rs.rand(len(df)) < sample_frac
            if not m.any(): del df,b; continue
            sdf = df.loc[m].copy()
            y = sdf.pop("clicked").astype("int8").values
            x_cat = torch.from_numpy(map_cats(sdf, VOCAB)).long().to(DEVICE)
            x_num = torch.from_numpy(map_nums(sdf, MEANS, STDS)).float().to(DEVICE)
            # 미니배치 추론
            preds=[]
            for s in range(0, len(sdf), GPU_BATCH*2):
                e=min(len(sdf), s+GPU_BATCH*2)
                logit = model(x_cat[s:e], x_num[s:e])
                preds.append(torch.sigmoid(logit).detach().cpu().numpy())
            p = np.concatenate(preds) if preds else np.zeros(0)
            y_list.append(y); p_list.append(p)
            used += len(y)
            del df, sdf, x_cat, x_num, y, p, preds, b; gc.collect()
            if used >= max_rows: break
    if used==0: return {"AP":None,"WLL":None,"Score":None,"Used":0,"p_valid":None,"y_valid":None}
    y_true = np.concatenate(y_list); y_prob = np.concatenate(p_list)
    ap,wll,score = compute_score(y_true, y_prob)
    return {"AP":ap,"WLL":wll,"Score":score,"Used":used,"p_valid":y_prob,"y_valid":y_true}

def calibrate_temperature(y_true, p, iters=200, lr=0.05):
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
        T -= lr*g; T = float(np.clip(T, 1e-3, 100.0))
    return T

def apply_temperature_np(p, T):
    eps=1e-15
    logit = np.log(np.clip(p,eps,1-eps)) - np.log(1-np.clip(p,eps,1-eps))
    return 1/(1+np.exp(-logit/max(T,1e-3)))

# ===== 6) 메인 루틴 =====
def main():
    t0 = time.time()
    # 카테고리 카디널리티
    cat_cardinals = [len(VOCAB[c]["itos"]) for c in CATS]
    n_num = len(NUM_COLS)
    print(f"[MODEL] cats={cat_cardinals}, n_num={n_num}, device={DEVICE}")

    model = FTTransformer(cat_cardinals, n_num).to(DEVICE)
    optim = make_optimizer(model)

    # ===== 학습 + EarlyStopping =====
    best_score = -1e9; best_state = None; no_improve = 0
    for epoch in range(1, EPOCHS+1):
        print(f"\n===== EPOCH {epoch}/{EPOCHS} =====")
        train_loss = train_epoch(model, train_ds, optim)
        print(f"[TRAIN] epoch {epoch} loss={train_loss:.6f}")

        valid = evaluate_sample(model, train_ds, sample_frac=VALID_SAMPLE_FRAC, max_rows=VALID_MAX_ROWS, batch=VALID_BATCH)
        ap,wll,score,used = valid["AP"], valid["WLL"], valid["Score"], valid["Used"]
        if ap is not None:
            print(f"[VALID] Used={used:,} | AP={ap:.6f} | WLL={wll:.6f} | Score={score:.6f}")
        else:
            print("[VALID] Not enough data.")

        if score is not None and score > best_score:
            best_score = score; best_state = {k:v.cpu() for k,v in model.state_dict().items()}
            no_improve = 0
            print(f"[EARLY] New best score: {best_score:.6f}")
        else:
            no_improve += 1
            print(f"[EARLY] no_improve={no_improve}/{EARLY_STOP_ROUNDS}")
            if no_improve >= EARLY_STOP_ROUNDS:
                print("[EARLY] Stopping.")
                break

    # best state 복원
    if best_state is not None:
        model.load_state_dict(best_state)

    # ===== 온도 스케일링 (선택) =====
    T = None
    if USE_TEMP_SCALING:
        valid = evaluate_sample(model, train_ds, sample_frac=min(0.12, VALID_SAMPLE_FRAC+0.02),
                                max_rows=VALID_MAX_ROWS, batch=VALID_BATCH)
        if valid["AP"] is not None:
            T = calibrate_temperature(valid["y_valid"], valid["p_valid"])
            print(f"[CALIB] Temperature T={T:.4f}")

    # ===== 모델 저장 =====
    model_path = os.path.join(save_path, "transformer_tab.pt")
    torch.save({"state_dict": model.state_dict(),
                "cat_cardinals": cat_cardinals,
                "n_num": n_num,
                "means": MEANS, "stds": STDS,
                "vocab": VOCAB,
                "T": T}, model_path)
    print("[TRAIN] saved:", model_path)

    # ===== 예측 =====
    print("[PRED] streaming predict …")
    total_test = count_rows(test_ds)
    today = datetime.datetime.now().strftime('%Y%m%d')
    score_str = ("nan" if (best_score is None or np.isnan(best_score)) else f"{best_score:.4f}").replace('.', '_')
    out_csv = os.path.join(save_path, f"{VER}_tr_score_{score_str}_submission_{today}.csv")

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["ID","clicked"])
        sc = test_ds.scanner(columns=["ID"] + list(features), batch_size=SCAN_TEST_BATCH)
        with tqdm(total=total_test, unit="rows", desc="[PRED] rows") as pbar:
            for b in sc.to_batches():
                df = pa.Table.from_batches([b]).to_pandas()
                ids = df.pop("ID").astype(str).values
                x_cat = map_cats(df, VOCAB)
                x_num = map_nums(df, MEANS, STDS)

                # 미니배치 추론
                preds=[]
                model.eval()
                with torch.no_grad():
                    for s in range(0, len(df), GPU_BATCH*2):
                        e=min(len(df), s+GPU_BATCH*2)
                        xb_cat = torch.from_numpy(x_cat[s:e]).long().to(DEVICE)
                        xb_num = torch.from_numpy(x_num[s:e]).float().to(DEVICE)
                        logit = model(xb_cat, xb_num)
                        p = torch.sigmoid(logit).detach().cpu().numpy()
                        preds.append(p)
                p_row = np.concatenate(preds) if preds else np.zeros(0)
                if T is not None:
                    p_row = apply_temperature_np(p_row, T)

                w.writerows(zip(ids.tolist(), p_row.tolist()))
                pbar.update(b.num_rows)
                del df, b, x_cat, x_num, ids, preds, p_row; gc.collect()

    print(f"[DONE] wrote {out_csv}")
    print(f"[TIME] total {(time.time()-t0):.1f}s")

    # ===== 로그 파일 =====
    with open(os.path.join(save_path, f"(LOG)model_{VER}.txt"), "a") as lf:
        lf.write(f"{VER}\n<SEED :{SEED}>\n")
        lf.write(f"best_Score={best_score:.6f}\n")
        if T is not None: lf.write(f"Temperature T={T:.6f}\n")
        lf.write("="*40 + "\n")

    return {"out_csv": out_csv, "model_path": model_path, "best_score": best_score, "T": T}

if __name__ == "__main__":
    _ = main()
