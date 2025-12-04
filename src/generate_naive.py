# src/generate_articles_naive.py
"""
naive_rag_results_top5.csv (나이브 RAG 검색 결과)를 사용해
질의(query)별로 기사문을 생성하는 스크립트.

입력:
  - naive_rag_results_top5.csv
    (search_naive.py에서 생성한, query 단위 top1~top5 결과)

출력:
  - naive_rag_articles.csv
    각 query_id별로 기사 1편
"""

import json
from typing import List, Dict, Any

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from paths import NAIVE_RESULTS_DIR  # 🔹 paths.py 사용

# --------------------------------------------------
# 경로 설정
# --------------------------------------------------
RAG_RESULT_CSV     = NAIVE_RESULTS_DIR / "naive_rag_results_top5.csv"
OUT_ARTICLE_CSV    = NAIVE_RESULTS_DIR / "naive_rag_articles.csv"
OUT_ARTICLE_JSONL  = NAIVE_RESULTS_DIR / "naive_rag_articles.jsonl"  # 평가용

# --- OpenAI 설정 ---
load_dotenv()
import os

OPENAI_KEY  = os.getenv("OPENAI_API_KEY")
OPENAI_BASE = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

client = OpenAI(api_key=OPENAI_KEY, base_url=OPENAI_BASE)
GEN_MODEL = "gpt-4.1-mini"  # 필요하면 gpt-4.1 등으로 교체


# --------------------------------------------------
# 1. 한 질의에 사용할 컨텍스트 단락 구성 (id 없어도 OK)
# --------------------------------------------------
def collect_segments_for_query(row: pd.Series, max_segments: int = 10) -> List[Dict[str, Any]]:
    """
    naive_rag_results_top5.csv에서
    한 행(row)을 받아, top1~top5의 (id, score, content)를 모아서
    점수 기준 상위 max_segments개를 반환.

    반환 형식:
      [{"id": (int 또는 None), "score": float, "content": str}, ...]
    """
    seg_rows: List[Dict[str, Any]] = []

    for k in range(1, 6):  # top1 ~ top5
        seg_id = row.get(f"top{k}_id")
        seg_score = row.get(f"top{k}_score")
        seg_content = row.get(f"top{k}_content")

        # content가 없으면 의미가 없으니 스킵
        if seg_content is None or (isinstance(seg_content, float) and pd.isna(seg_content)):
            continue

        # id는 없어도 baseline에는 큰 문제 없음 → None 허용
        sid = None
        if seg_id is not None and not (isinstance(seg_id, float) and pd.isna(seg_id)):
            try:
                sid = int(seg_id)
            except Exception:
                sid = None

        score = 0.0
        if seg_score is not None and not (isinstance(seg_score, float) and pd.isna(seg_score)):
            score = float(seg_score)

        seg_rows.append(
            {
                "id": sid,
                "score": score,
                "content": str(seg_content),
            }
        )

    # 나이브 RAG는 각 질의당 최대 5개지만, 혹시를 위해 점수 정렬
    seg_rows.sort(key=lambda x: x["score"], reverse=True)
    return seg_rows[:max_segments]


# --------------------------------------------------
# 2. 기사 생성 LLM 호출
# --------------------------------------------------
def build_context_block(segments: List[Dict[str, Any]]) -> str:
    """세그먼트들을 사람이 보기 좋게 블록 문자열로 변환"""
    lines = []
    for i, seg in enumerate(segments, start=1):
        sid = seg["id"]
        sid_str = "None" if sid is None else str(sid)
        header = f"[단락 {i} | id={sid_str} | score={seg['score']:.4f}]"
        body = seg["content"].strip()
        lines.append(header + "\n" + body)
    return "\n\n".join(lines)


def generate_article(query_text: str, segments: List[Dict[str, Any]]) -> str:
    """
    주어진 질의문과 관련 단락들을 바탕으로 한국어 기사 1편 생성.
    (프롬프트는 온톨로지 기반 RAG와 동일)
    """
    if not segments:
        return ""

    context_block = build_context_block(segments)

    user_prompt = f"""
당신은 지방의회 회의록을 바탕으로 기사를 작성하는 공공정책 전문 기자입니다.

아래는 어떤 질의에 대해 검색된 회의록 단락들입니다.
이 단락들만을 근거로 하여, 해당 질의에 답하는 한국어 뉴스 기사 1편을 작성하십시오.

[질의]
{query_text}

[참고 회의록 단락들]
{context_block}

요구 사항:
- 기사 형식: 제목 1개 + 본문 3~5단락
- 첫 줄에 기사 제목을 쓰고, 그 다음 줄부터 본문을 단락 구분이 되도록 작성
- 회의에서 오간 주요 쟁점, 의원의 문제 제기, 집행부 입장, 향후 과제 등을 중심으로 서술
- 제공된 단락들에 근거하지 않은 사실을 새로 만들어내지 말 것
- 단락 간 논리적 연결이 매끄럽게 이어지도록 구성할 것
- 특정 정당이나 인물을 과도하게 비난하거나 옹호하는 표현은 피하고,
  회의록에 나타난 발언 내용 중심으로 균형 있게 서술할 것
"""

    res = client.chat.completions.create(
        model=GEN_MODEL,
        messages=[
            {
                "role": "system",
                "content": "당신은 지방의회 의정 활동을 전문적으로 다루는 신뢰할 수 있는 기자입니다.",
            },
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
    )

    return res.choices[0].message.content.strip()


# --------------------------------------------------
# 3. 전체 배치 실행
# --------------------------------------------------
def main():
    df = pd.read_csv(RAG_RESULT_CSV)
    print("입력 CSV:", RAG_RESULT_CSV)
    print("행 수:", len(df))

    if "query_id" not in df.columns:
        raise ValueError("naive_rag_results_top5.csv에 query_id 컬럼이 없습니다.")

    article_rows: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        qid = int(row["query_id"])
        original_query = row["query"]

        print(f"\n=== query_id={qid} ===")
        print("질의:", original_query)

        # 1) 이 질의에 대한 세그먼트 수집
        segments = collect_segments_for_query(row, max_segments=10)
        print("  사용 세그먼트 개수:", len(segments))

        # 2) 기사 생성
        article_text = generate_article(original_query, segments)

        # 3) 저장용 row 구성
        used_ids = [s["id"] for s in segments]

        article_rows.append(
            {
                "query_id": qid,
                "original_query": original_query,
                "used_segment_ids": json.dumps(used_ids, ensure_ascii=False),
                "article": article_text,
            }
        )

    # 결과 폴더 보장
    NAIVE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 4) CSV로 저장
    out_df = pd.DataFrame(article_rows)
    out_df.to_csv(OUT_ARTICLE_CSV, index=False, encoding="utf-8-sig")
    print("\n✔ 기사 CSV 저장 완료:", OUT_ARTICLE_CSV)

    # 5) JSONL로도 저장
    with open(OUT_ARTICLE_JSONL, "w", encoding="utf-8") as f:
        for row in article_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("✔ 기사 JSONL 저장 완료:", OUT_ARTICLE_JSONL)


if __name__ == "__main__":
    main()
