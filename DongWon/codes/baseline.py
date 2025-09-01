# dacon_persona_openai_single.py
# -*- coding: utf-8 -*-
"""
VSCode 한 파일 베이스라인:
1) 제품별 '싱글턴' 프롬프트 자동 생성
2) OpenAI API로 페르소나(JSON) 생성(Structured Outputs)
3) population_weight 가중 purchase_prob 집계 → 제출 CSV(UTF-8)

설치:
  pip install --upgrade openai pandas numpy tenacity
환경변수:
  export OPENAI_API_KEY=sk-...

사용 예:
  python dacon_persona_openai_single.py \
    --product_csv product_info.csv \
    --sample_csv sample_submission.csv \
    --personas_dir personas_out \
    --out_csv submission_openai.csv \
    --model gpt-4o-mini \
    --limit 20
"""

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from openai import OpenAI
from openai import APIConnectionError, RateLimitError, APIError

# -----------------------------
# 설정
# -----------------------------
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_SEED = 73

PERSONA_SCHEMA_KEYS = [
    "persona_id",
    "age_range",
    "gender_identity",
    "location",
    "income_level",
    "education_level",
    "occupation_role",
    "tech_savviness",
    "values_and_motivations",
    "pain_points",
    "goals_outcomes",
    "brand_loyalty",
    "price_sensitivity",
    "purchase_channels",
    "content_tone_preference",
    "privacy_risk_tolerance",
    "population_weight",
    "purchase_prob"
]

ID_LIKE = {"id", "ID", "row_id", "index", "product_id", "ProductId", "prod_id", "PID"}

# -----------------------------
# 유틸
# -----------------------------
def guess_id_column(df: pd.DataFrame) -> str:
    for c in ["product_id", "prod_id", "id", "ID"]:
        if c in df.columns:
            return c
    return df.columns[0]

def guess_join_key(prod_cols: List[str], sub_cols: List[str]) -> Optional[str]:
    cand = [c for c in prod_cols if c in sub_cols and c.lower() in {"id", "product_id", "prod_id"}]
    if cand:
        return cand[0]
    inter = [c for c in prod_cols if c in sub_cols and "id" in c.lower()]
    return inter[0] if inter else None

def target_columns(sub: pd.DataFrame) -> List[str]:
    t = [c for c in sub.columns if c not in ID_LIKE and c.lower() not in {"id"}]
    return t or [sub.columns[-1]]

def normalize_weights(ws: np.ndarray) -> np.ndarray:
    ws = np.asarray(ws, dtype=float)
    ws = np.clip(ws, 0.0, None)
    s = ws.sum()
    return (np.ones_like(ws) / len(ws)) if s <= 0 else (ws / s)

def seed_from_product(global_seed: int, product_id: Any) -> int:
    h = hashlib.sha256(f"{global_seed}:{product_id}".encode("utf-8")).hexdigest()
    return int(h[:8], 16)

# -----------------------------
# 프롬프트 & 스키마
# -----------------------------
def build_single_turn_prompt(product_row: Dict[str, Any], seed_str: str) -> str:
    schema_list = json.dumps(PERSONA_SCHEMA_KEYS, ensure_ascii=False)
    prod_json = json.dumps(product_row, ensure_ascii=False)
    prompt = f"""
당신은 상업용 제품 페르소나 모델러입니다. 아래 규칙을 *모두* 지키세요.

# 목표
- 주어진 '제품 1개'에 대해, 서로 다른 4~8명 페르소나를 생성합니다.
- 각 페르소나는 아래 키({len(PERSONA_SCHEMA_KEYS)}개)를 모두 포함해야 합니다:
{schema_list}
- 각 페르소나는 이 제품에 대한 **purchase_prob**(0~1)와 **population_weight**(0~1)을 반드시 포함하세요.
  - population_weight는 세그먼트 상대 규모(합계=1 권장)입니다.

# 형식
- **JSON 배열**만 출력합니다. 코드블록/설명/주석 금지.
- 모든 값은 간결하고 현실적인 범위로 작성하세요.

# 제약
- **단일 턴**(Single-turn)만 허용됩니다(추가 질문/수정 없음).
- 제품 입력 외 사실을 임의로 꾸미지 마세요(허위 금지).
- 다양성: 연령/소득/동기/채널/톤이 서로 다르게.

# 랜덤성 제어
- seed="{seed_str}"

# 제품 입력(JSON)
{prod_json}
""".strip()
    return prompt

def persona_json_schema() -> Dict[str, Any]:
    string_field = {"type": "string", "minLength": 1}
    number_0_1 = {"type": "number", "minimum": 0.0, "maximum": 1.0}
    array_str = {"type": "array", "items": {"type": "string"}, "minItems": 1}

    persona_obj = {
        "type": "object",
        "properties": {
            "persona_id": string_field,
            "age_range": string_field,
            "gender_identity": string_field,
            "location": string_field,
            "income_level": string_field,
            "education_level": string_field,
            "occupation_role": string_field,
            "tech_savviness": string_field,
            "values_and_motivations": array_str,
            "pain_points": array_str,
            "goals_outcomes": array_str,
            "brand_loyalty": string_field,
            "price_sensitivity": string_field,
            "purchase_channels": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "content_tone_preference": string_field,
            "privacy_risk_tolerance": string_field,
            "population_weight": number_0_1,
            "purchase_prob": number_0_1
        },
        "required": PERSONA_SCHEMA_KEYS,
        "additionalProperties": False
    }

    return {
        "type": "array",
        "minItems": 4,
        "maxItems": 8,
        "items": persona_obj
    }

# -----------------------------
# OpenAI 호출
# -----------------------------
@dataclass
class LLMConfig:
    model: str
    temperature: float
    seed: int

class LLMClient:
    def __init__(self, model: str, temperature: float):
        self.client = OpenAI()
        self.model = model
        self.temperature = temperature

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        retry=retry_if_exception_type((APIConnectionError, RateLimitError, APIError))
    )
    def generate_personas(self, prompt: str, seed: int) -> List[Dict[str, Any]]:
        schema = persona_json_schema()
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            seed=seed,  # best-effort reproducibility
            messages=[
                {"role": "system", "content": "You are a strict persona generator that ONLY outputs data following the provided schema."},
                {"role": "user", "content": prompt}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "persona_array",
                    "schema": schema,
                    "strict": True
                }
            }
        )
        content = resp.choices[0].message.content
        try:
            data = json.loads(content)
            if not isinstance(data, list):
                raise ValueError("Model did not return a JSON array.")
            return data
        except Exception as e:
            raise ValueError(f"Failed to parse JSON from model: {e}\nRaw content: {content}")

# -----------------------------
# 집계
# -----------------------------
def aggregate_product_score(personas: List[Dict[str, Any]]) -> float:
    probs, weights = [], []
    for p in personas:
        prob = float(p.get("purchase_prob", np.nan))
        w = p.get("population_weight", None)
        if w is None:
            weights.append(1.0)
        else:
            try:
                weights.append(float(w))
            except Exception:
                weights.append(1.0)
        probs.append(prob)

    probs = np.asarray(probs, dtype=float)
    mask = np.isfinite(probs) & (probs >= 0) & (probs <= 1)
    if not mask.any():
        return 0.0

    probs = probs[mask]
    weights = normalize_weights(np.asarray(weights, dtype=float)[mask])
    return float(np.clip((probs * weights).sum(), 0.0, 1.0))

# -----------------------------
# 메인
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="DACON Persona Baseline (OpenAI API, single file)")
    ap.add_argument("--product_csv", type=Path, required=True, help="product_info.csv 경로")
    ap.add_argument("--sample_csv", type=Path, required=True, help="sample_submission.csv 경로")
    ap.add_argument("--out_csv", type=Path, default=Path("submission_openai.csv"), help="출력 제출 CSV 경로")
    ap.add_argument("--personas_dir", type=Path, default=Path("personas_out"), help="생성된 페르소나 JSON 저장 폴더")
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL, help="OpenAI 모델명 (예: gpt-4o-mini)")
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="샘플링 온도")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="글로벌 시드(제품별 해시로 파생)")
    ap.add_argument("--limit", type=int, default=0, help="생성할 제품 개수 제한 (0=전체)")
    ap.add_argument("--join_key", type=str, default=None, help="product_info와 sample_submission 연결 키 (없으면 추정)")
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY 환경변수를 설정하세요.")

    prod = pd.read_csv(args.product_csv)
    sub = pd.read_csv(args.sample_csv)

    key = args.join_key or guess_join_key(prod.columns.tolist(), sub.columns.tolist())
    prod_id_col = guess_id_column(prod)

    llm = LLMClient(model=args.model, temperature=args.temperature)
    args.personas_dir.mkdir(parents=True, exist_ok=True)

    product_rows = prod.to_dict(orient="records")
    if args.limit and args.limit > 0:
        product_rows = product_rows[: args.limit]

    product_scores: Dict[Any, float] = {}
    for idx, row in enumerate(product_rows, start=1):
        pid = row.get(prod_id_col)
        seed_val = seed_from_product(args.seed, pid)
        seed_str = f"seed-{args.seed}-prod-{pid}"
        out_file = args.personas_dir / f"personas_{pid}.json"

        # 캐시 사용
        personas = None
        if out_file.exists():
            try:
                personas = json.loads(out_file.read_text(encoding="utf-8"))
            except Exception:
                personas = None

        # 생성
        if personas is None:
            try:
                prompt = build_single_turn_prompt(row, seed_str)
                personas = llm.generate_personas(prompt, seed=seed_val)
                out_file.write_text(json.dumps(personas, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception as e:
                print(f"[ERROR] product_id={pid} persona gen failed: {e}")
                personas = None

        if personas:
            score = aggregate_product_score(personas)
            product_scores[pid] = score
            print(f"[{idx}/{len(product_rows)}] product_id={pid} score={score:.4f}")
        else:
            print(f"[{idx}/{len(product_rows)}] product_id={pid} score=0.0 (no personas)")

        time.sleep(0.1)  # rate-limit 완화

    # 제출 CSV 생성
    out = sub.copy()
    targets = target_columns(out)

    if key and (key in out.columns) and (key in prod.columns):
        out["_score_tmp"] = out[key].map(product_scores)
        fallback = float(np.nanmean(list(product_scores.values()))) if product_scores else 0.0
        out["_score_tmp"] = out["_score_tmp"].fillna(fallback).clip(0, 1)
        for t in targets:
            out[t] = out["_score_tmp"]
        out.drop(columns=["_score_tmp"], inplace=True)
    else:
        val = float(np.nanmean(list(product_scores.values()))) if product_scores else 0.0
        for t in targets:
            out[t] = float(np.clip(val, 0.0, 1.0))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False, encoding="utf-8")
    print(f"[OK] Wrote submission → {args.out_csv}")
    print(f"    targets={targets}")
    print(f"    personas_dir={args.personas_dir} generated={len(product_scores)}")

if __name__ == "__main__":
    main()
