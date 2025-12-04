# src/search_naive.py
"""
base_minutes_rag.parquet (chunk + 임베딩)과
test_queries.csv (질의 100개)를 사용해

각 질의별로 임베딩 기반 나이브 RAG 검색을 수행하고,
top1~top5 단락을 CSV로 저장하는 스크립트.

출력: naive_rag_results_top5.csv
"""

from typing import List, Dict, Any
import numpy as np
import pandas as pd
import faiss
from dotenv import load_dotenv
from openai import OpenAI

from pathlib import Path
from paths import DATA_DIR, RESULTS_DIR, NAIVE_RESULTS_DIR

# --------------------------------------------------
# 경로 설정 (paths.py 기반)
# --------------------------------------------------

CHUNKS_PARQUET = DATA_DIR / "base_minutes_rag.parquet"    # data/
QUERY_CSV      = RESULTS_DIR / "test_queries.csv"          # results/ 내부
OUT_CSV        = NAIVE_RESULTS_DIR / "naive_rag_results_top5.csv"

TEXT_COL_CANDIDATES = ["chunk_text", "content", "text"]


# --------------------------------------------------
# OpenAI 설정 (임베딩용)
# --------------------------------------------------
load_dotenv()
import os
OPENAI_KEY  = os.getenv("OPENAI_API_KEY")
OPENAI_BASE = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

client = OpenAI(api_key=OPENAI_KEY, base_url=OPENAI_BASE)
EMB_MODEL = "text-embedding-3-large"


def embed_query(text: str) -> np.ndarray:
    """질의문 임베딩"""
    res = client.embeddings.create(
        model=EMB_MODEL,
        input=[text],
    )
    v = np.array(res.data[0].embedding, dtype="float32")
    v /= np.linalg.norm(v) + 1e-12     # L2 normalize
    return v.reshape(1, -1)


def main():

    print(f"=== chunk + 임베딩 로드: {CHUNKS_PARQUET} ===")
    df_chunks = pd.read_parquet(CHUNKS_PARQUET)

    if "embedding" not in df_chunks.columns:
        raise ValueError("parquet에 'embedding' 컬럼이 없습니다.")

    # 텍스트 컬럼 자동 탐지
    text_col = None
    for c in TEXT_COL_CANDIDATES:
        if c in df_chunks.columns:
            text_col = c
            break

    if text_col is None:
        raise ValueError(f"텍스트 컬럼을 찾을 수 없습니다: {TEXT_COL_CANDIDATES}")

    print(f"사용 텍스트 컬럼: {text_col}")

    # ---- 임베딩 행렬 ----
    emb = np.vstack(df_chunks["embedding"].values).astype("float32")
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    print("embedding matrix shape:", emb.shape)

    # FAISS index
    d = emb.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(emb)
    print("FAISS index 준비 완료.")

    # ---- test_queries.csv ----
    df_q = pd.read_csv(QUERY_CSV)
    print("질의 수:", len(df_q))

    has_id = "id" in df_q.columns
    result_rows: List[Dict[str, Any]] = []

    # ---- 질의별 top5 검색 ----
    for i, row in df_q.iterrows():
        query_id = int(row["id"]) if has_id else (i + 1)
        qtext = row["query"]

        print(f"\n[{i+1}/{len(df_q)}] query_id={query_id}  질의:", qtext)

        q_vec = embed_query(qtext)
        scores, idxs = index.search(q_vec, 5)
        scores = scores[0]
        idxs = idxs[0]

        base: Dict[str, Any] = {
            "query_id": query_id,
            "query": qtext,
        }

        for rank, (score, ridx) in enumerate(zip(scores, idxs), start=1):
            if ridx < 0:
                base[f"top{rank}_id"] = None
                base[f"top{rank}_score"] = None
                base[f"top{rank}_content"] = None
                continue

            row_seg = df_chunks.iloc[ridx]
            seg_id = row_seg.get("id")
            content = str(row_seg[text_col])

            base[f"top{rank}_id"] = int(seg_id) if seg_id is not None else None
            base[f"top{rank}_score"] = float(score)
            base[f"top{rank}_content"] = content

        result_rows.append(base)

    # ---- 저장 ----
    NAIVE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    out_df = pd.DataFrame(result_rows)
    out_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\n✔ 나이브 RAG top5 CSV 저장 완료: {OUT_CSV}")


if __name__ == "__main__":
    main()
