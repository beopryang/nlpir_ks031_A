#!/usr/bin/env python3
# src/evaluate_absolute.py
# -------------------------------------------------------
# python evaluate_absolute.py
#   → naive + ontology 두 결과 모두 평가
#   - fact_score: 0 또는 1 (정당/의원/의회 등 fact 오류 유무)
#   - topic_score: 1~10 (주제 관련성 절대/상대 평가)
#   - overall_score:
#       · fact_score == 0 → 0
#       · fact_score == 1 → topic_score / 10
#   - 단락이 없으면 fact/topic/overall 모두 None (EMPTY_SEGMENT)
# -------------------------------------------------------

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from paths import (
    RESULTS_DIR,
    NAIVE_RESULTS_DIR,
    ONTOLOGY_RESULTS_DIR,
)

# =============== 0. 환경 설정 / 경로 / 파일명 ===============

# .env 로드
load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

if OPENAI_KEY is None:
    raise RuntimeError("OPENAI_API_KEY가 .env에서 로드되지 않았습니다.")

client = OpenAI(api_key=OPENAI_KEY, base_url=OPENAI_BASE)

PLANNER_MODEL  = "gpt-4.1-mini"
EXECUTOR_MODEL = "gpt-4.1-mini"

# --- 입력 파일 ---
#   results/naive/naive_rag_results_top5.csv
#   results/ontology/ontology_rag_results_tasks.csv
NAIVE_INPUT_FILE = NAIVE_RESULTS_DIR / "naive_rag_results_top5.csv"
ONTO_INPUT_FILE  = ONTOLOGY_RESULTS_DIR / "ontology_rag_results_tasks.csv"

# --- 출력 디렉터리 & 파일 ---
#   다른 평가 스크립트와 통일: results/eval/
EVAL_RESULTS_DIR = RESULTS_DIR / "eval"
EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUT_PLAN_NAIVE = EVAL_RESULTS_DIR / "eval_plans_naive_top5_absolute.json"
OUT_PLAN_ONTO  = EVAL_RESULTS_DIR / "eval_plans_onto_top5_absolute.json"
OUT_EVAL_NAIVE = EVAL_RESULTS_DIR / "eval_top5_truth_naive_top5_absolute.csv"
OUT_EVAL_ONTO  = EVAL_RESULTS_DIR / "eval_top5_truth_onto_top5_absolute.csv"


# =============== 공통 유틸 ===============

def find_query_column(columns):
    """
    질의 컬럼 추론: query > query_text > original_query > category
    """
    for cand in ["query", "query_text", "original_query", "category"]:
        if cand in columns:
            return cand
    return None


def to_float(x):
    try:
        return float(x)
    except Exception:
        return None


def extract_segment_text(row: pd.Series, content_col: str) -> str:
    """
    topk content에서 실제 텍스트를 안전하게 추출.
    - NaN, None, "nan", "null", "None" 등은 모두 빈 세그먼트로 처리.
    """
    if content_col not in row.index:
        return ""

    raw = row[content_col]

    # 진짜 NaN / None
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return ""

    s = str(raw).strip()
    if not s:
        return ""
    if s.lower() in ["nan", "none", "null"]:
        return ""

    return s


# =============== 1. 평가 기획 에이전트 (PLANNER) ===============

EVAL_PLANNER_SYSTEM_PROMPT = """
You are an evaluation planning agent for a Korean local council RAG system.

Your role:
- Read the Korean query Q.
- Decide which factual dimensions must be checked (persons, parties, councils).
- Decide whether the query requires topic relevance evaluation.
- Extract target names (persons, parties, councils) and topic keywords.

Output ONLY one JSON object with the following structure:

{
  "check_person": true or false,
  "check_party": true or false,
  "check_council": true or false,
  "need_topic": true or false,

  "target_persons": [list of person names or []],
  "target_parties": [list of party names or []],
  "target_councils": [list of council names or []],

  "topic_keywords": [list of Korean keywords for main topic or []],

  "allow_other_persons": true or false,
  "allow_single_party_segment": true or false
}

Guidelines:
- check_person: true if the query explicitly focuses on specific council members.
- check_party: true if the query is about specific parties or compares parties.
- check_council: true if the query is about specific councils (e.g., 광주광역시의회, 서울특별시의회).
- need_topic: usually true, unless the query is purely about meta info (rare).

- allow_other_persons:
  - true if the main speaker/person can appear with other names in the same segment.
  - false only if the segment must be strictly focused on the target person(s).

- allow_single_party_segment:
  - For comparison queries (e.g., 민주당 vs 국민의힘),
    set this to true so that a segment with only one party (e.g., 민주당만) is still factually valid.
  - For non-comparison queries, usually true.

- topic_keywords:
  - Choose 2~5 short Korean keywords that summarize the main policy/topic
    (e.g., "복지 예산", "교통 인프라", "직업계고", "교육", "도시계획").
"""

def call_eval_planner(query: str) -> Dict[str, Any]:
    user_prompt = f"[질의문 Q]\n{query}\n\n위 질의를 평가하기 위한 계획을 JSON으로만 출력하라."
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=PLANNER_MODEL,
                messages=[
                    {"role": "system", "content": EVAL_PLANNER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            print(f"[WARN] planner error: {e}")
            time.sleep(1.2)
    return {"_error": "planner_failed"}


# =============== 2. 평가 실행 에이전트 (EXECUTOR) ===============

EVAL_EXECUTOR_SYSTEM_PROMPT = """
You are an evaluation executor for a Korean local council RAG system.

Given:
- Query Q
- Evaluation plan P (contains what to check and targets)
- Segment S

You must:

1) Judge factual correctness of S with respect to P:
   - Check persons, parties, councils only if P.check_person / check_party / check_council is true.
   - If any required dimension (person/party/council) clearly contradicts P,
     the factual correctness is 0.
   - If all required dimensions are consistent with P (or not violated), factual correctness is 1.

2) Judge topic relevance of S to Q:
   - Score from 1 to 10 (integer):
     10: very strongly and directly related to the main topic
      7-9: generally related, but with some missing or extra content
      4-6: loosely related; partially about the topic but not central
      1-3: weak or almost irrelevant
   - If P.need_topic is false, you may set topic_score = 5 by default.

Return ONLY one JSON object:

{
  "fact_score": 0 or 1,
  "person_score": 0 or 1 or null,
  "party_score": 0 or 1 or null,
  "council_score": 0 or 1 or null,
  "topic_score": 1~10,
  "error_type": "NONE" | "WRONG_PERSON" | "WRONG_PARTY" | "WRONG_COUNCIL" |
                "IRRELEVANT" | "MIXED" | "OTHER",
  "comment": "Korean explanation"
}

Guidelines:

- person_score:
  - If P.check_person is false, set person_score = null.
  - If true:
    - 1 if S is consistent with the target persons in P (main speaker or main actor matches),
      even if other persons are mentioned, as long as P.allow_other_persons is true.
    - 0 if S is clearly about a different main person and not about the target.

- party_score:
  - If P.check_party is false, set party_score = null.
  - If true:
    - 1 if S is consistent with target_parties in P.
      For comparison queries, a segment with only one party is allowed
      when P.allow_single_party_segment is true.
    - 0 if S mainly concerns unrelated parties or clearly wrong parties.

- council_score:
  - If P.check_council is false, set council_score = null.
  - If true:
    - 1 if S belongs to or clearly refers to the correct target councils.
    - 0 if it clearly refers to different councils.

- fact_score:
  - If any enabled dimension (person/party/council) has score 0,
    then fact_score must be 0.
  - If all enabled dimensions have score 1 (or null), fact_score must be 1.

- topic_score:
  - Must be an integer 1~10.
  - Even if fact_score is 0 (wrong person/party/council),
    still judge topic relevance based on the content.

DO NOT output anything other than this JSON object.
"""

def call_eval_executor(query: str, plan: Dict[str, Any], segment: str) -> Dict[str, Any]:
    plan_str = json.dumps(plan, ensure_ascii=False)
    user_prompt = f"""
[질의문 Q]
{query}

[평가 계획 P]
{plan_str}

[세그먼트 S]
{segment}

위 P를 기준으로 S의 사실성(fact)과 주제 관련성(topic)을 평가하라.
fact_score는 0 또는 1, topic_score는 1~10 정수로만 설정하라.
JSON 객체만 출력하라.
    """
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=EXECUTOR_MODEL,
                messages=[
                    {"role": "system", "content": EVAL_EXECUTOR_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            print(f"[WARN] executor error: {e}")
            time.sleep(1.2)
    return {"_error": "executor_failed"}


# =============== 3. top5 평가 공통 함수 ===============

def evaluate_file(input_path: Path, out_plan_json: Path, out_eval_csv: Path):

    if not input_path.exists():
        raise FileNotFoundError(f"파일이 없음: {input_path}")

    df = pd.read_csv(input_path)
    q_col = find_query_column(df.columns)
    if q_col is None:
        raise ValueError(f"질의 컬럼을 찾지 못함: {df.columns}")

    if "query_id" not in df.columns:
        df["query_id"] = range(1, len(df) + 1)

    # ------- 1) query별 평가 계획 생성 -------
    plans: Dict[Any, Dict[str, Any]] = {}
    for qid, sub in df.groupby("query_id"):
        qtext = str(sub.iloc[0][q_col])
        print(f"[PLAN] {input_path.name} query_id={qid}")
        plan = call_eval_planner(qtext)
        plans[qid] = plan

    with out_plan_json.open("w", encoding="utf-8") as f:
        json.dump(plans, f, ensure_ascii=False, indent=2)

    # ------- 2) top5 평가 -------
    rows = []
    for idx, row in df.iterrows():
        qid   = row["query_id"]
        query = str(row[q_col])
        plan  = plans.get(qid, {})

        for rank in range(1, 6):

            content_col = f"top{rank}_content"
            score_col   = f"top{rank}_score"

            if content_col not in df.columns:
                continue

            seg = extract_segment_text(row, content_col)
            retr_score = row.get(score_col, None)

            if not seg:
                # LLM 호출하지 않고, "EMPTY_SEGMENT"로만 기록
                rows.append({
                    "query_id": qid,
                    "query": query,
                    "rank": rank,
                    "retrieval_score": retr_score,
                    "segment": "",
                    "fact_score": None,
                    "person_score": None,
                    "party_score": None,
                    "council_score": None,
                    "topic_score": None,
                    "overall_score": None,
                    "error_type": "EMPTY_SEGMENT",
                    "comment": "segment 없음",
                })
                continue

            print(f"[EXEC] {input_path.name} qid={qid} top{rank}")
            j = call_eval_executor(query, plan, seg)

            if "_error" in j:
                rows.append({
                    "query_id": qid,
                    "query": query,
                    "rank": rank,
                    "retrieval_score": retr_score,
                    "segment": seg,
                    "fact_score": None,
                    "person_score": None,
                    "party_score": None,
                    "council_score": None,
                    "topic_score": None,
                    "overall_score": None,
                    "error_type": "LLM_ERROR",
                    "comment": j["_error"],
                })
                continue

            fact_score    = to_float(j.get("fact_score"))
            person_score  = to_float(j.get("person_score")) if j.get("person_score") is not None else None
            party_score   = to_float(j.get("party_score")) if j.get("party_score") is not None else None
            council_score = to_float(j.get("council_score")) if j.get("council_score") is not None else None
            topic_score   = to_float(j.get("topic_score"))

            # overall_score 규칙:
            # - fact_score == 0 → 0
            # - fact_score == 1 → topic_score / 10
            if fact_score is None or topic_score is None:
                overall = None
            else:
                if int(round(fact_score)) == 0:
                    overall = 0.0
                else:
                    overall = max(0.0, min(1.0, topic_score / 10.0))

            rows.append({
                "query_id": qid,
                "query": query,
                "rank": rank,
                "retrieval_score": retr_score,
                "segment": seg,
                "fact_score": fact_score,
                "person_score": person_score,
                "party_score": party_score,
                "council_score": council_score,
                "topic_score": topic_score,
                "overall_score": overall,
                "error_type": j.get("error_type", ""),
                "comment": j.get("comment", ""),
            })

    out_df = pd.DataFrame(rows)

    # query별 평균 추가 (EMPTY_SEGMENT/LLM_ERROR는 NaN으로 남음)
    grouped = out_df.groupby("query_id")["overall_score"].mean().reset_index()
    grouped = grouped.rename(columns={"overall_score": "overall_mean_top5"})
    out_df = out_df.merge(grouped, on="query_id", how="left")

    out_df.to_csv(out_eval_csv, index=False, encoding="utf-8-sig")
    print(f"[DONE] {out_eval_csv} 저장됨")


# =============== 4. MAIN: 엔트리포인트들 ===============

def main():
    """naive + ontology 둘 다 평가"""
    print("\n=== 1) Naive RAG top5 평가 시작 ===")
    evaluate_file(NAIVE_INPUT_FILE, OUT_PLAN_NAIVE, OUT_EVAL_NAIVE)

    print("\n=== 2) Ontology RAG top5 평가 시작 ===")
    evaluate_file(ONTO_INPUT_FILE, OUT_PLAN_ONTO, OUT_EVAL_ONTO)

    print("\n=== 모든 top5 평가 완료 ===")


def main_only_ontology():
    """온톨로지 RAG top5 평가만 수행"""
    print("\n=== [ONLY ONTO] Ontology RAG top5 평가 시작 ===")
    evaluate_file(ONTO_INPUT_FILE, OUT_PLAN_ONTO, OUT_EVAL_ONTO)
    print("\n=== [ONLY ONTO] Ontology RAG top5 평가 완료 ===")


if __name__ == "__main__":
    # 인자가 없으면 naive + ontology 둘 다 평가
    # 인자로 --ontology-only 가 들어오면 온톨로지만 평가
    if len(sys.argv) > 1 and sys.argv[1] == "--ontology-only":
        main_only_ontology()
    else:
        main()
