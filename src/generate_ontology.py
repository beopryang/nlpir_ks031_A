"""
ontology_rag_results_tasks.csv (온톨로지 RAG 검색 결과)를 사용해
질의(query)별 기사문을 생성하는 스크립트.
"""

import json
from typing import List, Dict, Any

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from paths import (
    RESULTS_DIR,
    ONTOLOGY_RESULTS_DIR,
)

# --------------------------------------------------
# 경로 설정
# --------------------------------------------------

RAG_RESULT_CSV     = ONTOLOGY_RESULTS_DIR / "ontology_rag_results_tasks.csv"
OUT_ARTICLE_CSV    = ONTOLOGY_RESULTS_DIR / "ontology_rag_articles.csv"
OUT_ARTICLE_JSONL  = ONTOLOGY_RESULTS_DIR / "ontology_rag_articles.jsonl"

# --------------------------------------------------
# OpenAI 설정
# --------------------------------------------------
load_dotenv()
import os

OPENAI_KEY  = os.getenv("OPENAI_API_KEY")
OPENAI_BASE = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

client = OpenAI(api_key=OPENAI_KEY, base_url=OPENAI_BASE)
GEN_MODEL = "gpt-4.1-mini"


# --------------------------------------------------
# 1. 세그먼트 수집
# --------------------------------------------------
def collect_segments_for_query(df_q: pd.DataFrame, max_segments: int = 10) -> List[Dict[str, Any]]:
    seg_rows: List[Dict[str, Any]] = []

    for _, row in df_q.iterrows():
        for k in range(1, 6):  # top1 ~ top5
            seg_id = row.get(f"top{k}_id")
            seg_score = row.get(f"top{k}_score")
            seg_content = row.get(f"top{k}_content")

            if pd.isna(seg_id) or seg_id is None:
                continue
            if isinstance(seg_content, float) and pd.isna(seg_content):
                continue

            seg_rows.append(
                {
                    "id": int(seg_id),
                    "score": float(seg_score) if seg_score is not None else 0.0,
                    "content": str(seg_content),
                }
            )

    # 중복 ID → 최고 score만 유지
    best_by_id = {}
    for s in seg_rows:
        sid = s["id"]
        if sid not in best_by_id or s["score"] > best_by_id[sid]["score"]:
            best_by_id[sid] = s

    unique_list = list(best_by_id.values())
    unique_list.sort(key=lambda x: x["score"], reverse=True)
    return unique_list[:max_segments]


# --------------------------------------------------
# 2. 기사 생성
# --------------------------------------------------
def build_context_block(segments: List[Dict[str, Any]]) -> str:
    lines = []
    for i, seg in enumerate(segments, start=1):
        header = f"[단락 {i} | id={seg['id']} | score={seg['score']:.4f}]"
        body = seg["content"].strip()
        lines.append(header + "\n" + body)
    return "\n\n".join(lines)


def generate_article(query_text: str, segments: List[Dict[str, Any]]) -> str:
    if not segments:
        return ""

    context_block = build_context_block(segments)

    user_prompt = f"""
아래는 어떤 질의에 대해 검색된 지방의회 회의록 단락들이다.
이 단락들만을 근거로 한국어 뉴스 기사 1편을 작성하라.

[질의]
{query_text}

[참고 회의록 단락들]
{context_block}

요구 사항:
- 기사 제목 1개 포함
- 본문은 3~5개 단락
- 제공된 단락 내용을 기반으로만 작성 (허위 내용 금지)
- 회의 쟁점과 발언 요지를 중심으로 균형 있게 구성
"""

    res = client.chat.completions.create(
        model=GEN_MODEL,
        messages=[
            {
                "role": "system",
                "content": "당신은 지방의회 회의록을 분석해 기사문을 작성하는 전문 기자입니다."
            },
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
    )

    return res.choices[0].message.content.strip()


# --------------------------------------------------
# 3. 전체 실행
# --------------------------------------------------
def main():
    # 입력 결과 로드
    df = pd.read_csv(RAG_RESULT_CSV)
    print("입력 CSV:", RAG_RESULT_CSV)
    print("행 수:", len(df))

    if "query_id" not in df.columns:
        raise ValueError("ontology_rag_results_tasks.csv에 query_id 컬럼이 없습니다.")

    article_rows = []

    grouped = df.groupby("query_id", sort=True)

    for qid, df_q in grouped:
        original_query = df_q["original_query"].iloc[0]
        category = df_q["category"].iloc[0] if "category" in df_q.columns else None

        print(f"\n=== query_id={qid} ===")
        print("질의:", original_query)

        segments = collect_segments_for_query(df_q, max_segments=10)
        print("  사용 세그먼트:", len(segments))

        article_text = generate_article(original_query, segments)
        used_ids = [s["id"] for s in segments]

        article_rows.append(
            {
                "query_id": int(qid),
                "category": category,
                "original_query": original_query,
                "used_segment_ids": json.dumps(used_ids, ensure_ascii=False),
                "article": article_text,
            }
        )

    # 출력 폴더 생성
    ONTOLOGY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # CSV 저장
    out_df = pd.DataFrame(article_rows)
    out_df.to_csv(OUT_ARTICLE_CSV, index=False, encoding="utf-8-sig")
    print("\n✔ 기사 CSV 저장 완료:", OUT_ARTICLE_CSV)

    # JSONL 저장
    with open(OUT_ARTICLE_JSONL, "w", encoding="utf-8") as f:
        for row in article_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("✔ 기사 JSONL 저장 완료:", OUT_ARTICLE_JSONL)


if __name__ == "__main__":
    main()
