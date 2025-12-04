# src/search.py
"""
test_queries.csv에 들어 있는 질의들에 대해
- (1) 플래너 LLM: 검색용 질의문(task_query) 여러 개 생성
- (2) 분석 LLM: 각 task_query에 대해 메타데이터(cities, parties 등) 추출
- (3) 온톨로지 + importance 우선순위(3 → 2) + FAISS 검색

을 수행하고, 아래 파일들로 저장한다.

1) query_plan.json
   - 각 원소: {
       query_id, category, original_query,
       tasks: [{task_id, name, task_query}, ...]
     }

2) task_analysis.json
   - 각 원소: {
       query_id, task_id, task_name, task_query,
       analysis: {...}
     }

3) ontology_rag_results_tasks.csv
   - 각 행: (query_id, task_id, original_query, task_name, task_query,
             top1_id/score/content, ... , top5_id/score/content)
"""

import json
import pickle
from typing import Dict, Any, List
import re

import pandas as pd
import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI

from paths import (
    CONFIG_DIR,
    DATA_DIR,
    RESULTS_DIR,
    ONTOLOGY_RESULTS_DIR,
)

# --------------------------------------------------
# 경로 및 파일 설정
# --------------------------------------------------
# data/ 쪽에 있는 파일들
PARQUET_PATH = DATA_DIR / "segments_all.parquet"

# 인덱스 위치는 실제 저장 위치에 맞게 조정
# 예: data/faiss_index/all.index, data/ontology_index/all.pkl
INDEX_PATH = DATA_DIR / "faiss_index" / "all.index"
ONTO_PATH = DATA_DIR / "ontology_index" / "all.pkl"

# test_queries.csv 는 현재 results 폴더에 있으므로 이렇게 설정
# (원하면 나중에 config로 옮기고 CONFIG_DIR / "test_queries.csv" 로 바꿔도 됨)
QUERY_CSV = RESULTS_DIR / "test_queries.csv"

# 출력 파일들
OUT_RESULTS_CSV = ONTOLOGY_RESULTS_DIR / "ontology_rag_results_tasks.csv"
OUT_PLAN_JSON = CONFIG_DIR / "query_plan.json"
OUT_ANAL_JSON = CONFIG_DIR / "task_analysis.json"

# --------------------------------------------------
# OpenAI 설정
# --------------------------------------------------
load_dotenv()
import os  # dotenv 이후에만 필요

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

client = OpenAI(api_key=OPENAI_KEY, base_url=OPENAI_BASE)

PLANNER_MODEL = "gpt-4.1-mini"           # 검색용 task_query 생성
ANALYZER_MODEL = "gpt-4.1-mini"          # 메타데이터 추출
EMB_MODEL = "text-embedding-3-large"     # 검색용 임베딩

_cache: Dict[str, Any] = {}


# --------------------------------------------------
# 0. 데이터 로딩 (한 번만)
# --------------------------------------------------
def load_all():
    """parquet + faiss index + ontology pkl 로드 (캐시)"""
    if _cache:
        return _cache["df"], _cache["index"], _cache["onto"]

    df = pd.read_parquet(PARQUET_PATH)

    # faiss는 문자열 경로를 기대하므로 str() 사용
    index = faiss.read_index(str(INDEX_PATH))

    with open(ONTO_PATH, "rb") as f:
        onto = pickle.load(f)  # by_city/by_council/by_person/by_party/by_tag

    _cache["df"] = df
    _cache["index"] = index
    _cache["onto"] = onto
    return df, index, onto


# --------------------------------------------------
# 1. 공통: 응답에서 JSON 부분만 추출
# --------------------------------------------------
def _extract_json(text: str) -> str:
    """응답에서 JSON 부분만 뽑아내기 위한 보조 함수."""
    if not text:
        return "{}"
    text = text.strip()

    # ```json ... ``` 형태일 때 안쪽만 추출
    if "```" in text:
        parts = text.split("```")
        text = max(parts, key=len).strip()

    # 중괄호 범위만 잘라서 사용
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and start < end:
        return text[start : end + 1]
    return "{}"


# --------------------------------------------------
# 2. 플래너 LLM: 원 질의 → task_query 여러 개
# --------------------------------------------------
PLANNER_PROMPT = """
당신은 지방의회 회의록 기반 온톨로지 RAG 시스템의
'검색 질의 생성 에이전트'입니다.

역할:
- 입력된 사용자의 질의문을 이해하고,
- RAG 검색에서 정확도를 높이기 위한 1개 이상 여러 개의 '검색용 질의문(task_query)'로 재구성합니다.
- 각 task_query는 실제로 회의록 단락을 검색하기에 적합한 형태여야 합니다.
  (예: 특정 정당, 특정 의회, 특정 정책 분야를 명확히 드러내기)

출력 형식:
아래 JSON 구조로만 출력하세요.

{
  "tasks": [
    {
      "task_id": 1,
      "name": "한 줄 요약된 작업 이름",
      "task_query": "이 작업에서 실제 검색에 사용할 한국어 질의문"
    },
    ...
  ]
}

규칙:
- 반드시 위 구조의 단일 JSON 객체만 출력하세요.
- 설명, 말줄임표, 마크다운, 코드블록은 사용하지 마세요.
- tasks는 최소 1개 이상이어야 합니다.
- 비교 질의(정당/도시/의회/의원 간 비교)인 경우,
  각 비교 대상에 대해 별도의 task_query를 만들어야 합니다.
"""


def plan_query(query_text: str) -> Dict[str, Any]:
    """원본 질의를 검색용 task_query 여러 개로 분해"""
    res = client.chat.completions.create(
        model=PLANNER_MODEL,
        messages=[
            {"role": "system", "content": PLANNER_PROMPT},
            {"role": "user", "content": query_text},
        ],
        temperature=0.2,
    )
    raw = res.choices[0].message.content or ""
    raw_json = _extract_json(raw)

    try:
        parsed = json.loads(raw_json)
    except Exception:
        parsed = {}

    tasks = parsed.get("tasks") or []

    fixed_tasks = []
    for i, t in enumerate(tasks, start=1):
        fixed_tasks.append(
            {
                "task_id": int(t.get("task_id") or i),
                "name": t.get("name") or f"Task {i}",
                "task_query": t.get("task_query") or query_text,
            }
        )

    return {"tasks": fixed_tasks}


# --------------------------------------------------
# 3. 분석 LLM: task_query → 메타데이터 추출
# --------------------------------------------------
ANALYZER_PROMPT = """
당신은 지방의회 회의록 검색을 위한 '메타데이터 분석 에이전트'입니다.
사용자의 한국어 질의문(task_query)을 읽고, 아래 JSON 구조로만 출력하세요.

{
  "cities": ["gwangju", "seoul"] 중 포함되는 것들,
  "councils": ["광주광역시의회", "서울특별시의회"] 중 포함되는 것들,
  "persons": 질의문에 등장하는 의원 이름들 (예: "임미란", "김동욱"),
  "parties": ["더불어민주당", "국민의힘", "무소속"] 중 포함되는 것들,
  "tag_keywords": tag 컬럼에서 찾을 정책 키워드들 (예: "예산", "교통", "교육" 등),
  "is_comparative": true/false,
  "comparison_axis": "city" / "party" / "person" / null,
  "residual_query": "실제 임베딩 검색용 핵심 내용(불필요한 수식어 제거, 간결하게)"
}

규칙:
- 반드시 위 필드들만 포함한 단일 JSON 객체를 출력하세요.
- 설명, 말줄임표, 마크다운, 코드블록은 사용하지 마세요.
"""


def _clean_list_field(parsed: Dict[str, Any], key: str) -> List[str]:
    vals = parsed.get(key) or []
    if isinstance(vals, str):
        vals = [vals]
    cleaned = []
    for v in vals:
        if v is None:
            continue
        if not isinstance(v, str):
            v = str(v)
        v = v.strip()
        if not v:
            continue
        if v.lower() == "null":
            continue
        cleaned.append(v)
    parsed[key] = cleaned
    return cleaned


def analyze_task_query(task_query: str) -> Dict[str, Any]:
    """task_query를 메타데이터+residual_query로 변환"""
    res = client.chat.completions.create(
        model=ANALYZER_MODEL,
        messages=[
            {"role": "system", "content": ANALYZER_PROMPT},
            {"role": "user", "content": task_query},
        ],
        temperature=0.0,
    )
    raw = res.choices[0].message.content or ""
    raw_json = _extract_json(raw)

    try:
        parsed = json.loads(raw_json)
    except Exception:
        parsed = {}

    # 리스트 필드 정규화
    for key in ["cities", "councils", "persons", "parties", "tag_keywords"]:
        _clean_list_field(parsed, key)

    parsed.setdefault("is_comparative", False)
    parsed.setdefault("comparison_axis", None)
    parsed.setdefault("residual_query", task_query)

    return parsed


# --------------------------------------------------
# 4. 임베딩
# --------------------------------------------------
def embed(text: str) -> np.ndarray:
    res = client.embeddings.create(
        model=EMB_MODEL,
        input=[text],
    )
    v = np.array(res.data[0].embedding, dtype="float32")
    return v.reshape(1, -1)


# --------------------------------------------------
# 5. 온톨로지 + tag 필터링 (안전 버전)
# --------------------------------------------------
def _normalize_list(x):
    if not x:
        return []
    if isinstance(x, str):
        x = [x]
    out = []
    for v in x:
        if v is None:
            continue
        if not isinstance(v, str):
            v = str(v)
        v = v.strip()
        if not v or v.lower() == "null":
            continue
        out.append(v)
    return out


def _union_from_dict(index_dict: Dict[str, List[int]], keys: List[str]) -> set:
    s = set()
    for k in keys:
        s |= set(index_dict.get(k, []))
    return s


def ontology_filter(
    df,
    onto,
    cities=None,
    councils=None,
    persons=None,
    parties=None,
    tag_keywords=None,
) -> np.ndarray:
    """
    all.pkl 온톨로지(by_* 딕셔너리)와 tag 컬럼을 활용한 후보 세그먼트 선택.
    - 각 차원(cities, councils, persons, parties)은 내부적으로 OR, 차원 간에는 AND.
    - tag_keywords는 df['tag'] 부분 문자열 검색(OR) 후 다른 조건과 AND.
    - 어떤 조건도 없으면 전체 인덱스 반환.
    """

    cand = None

    # ---- cities ----
    cities = _normalize_list(cities)
    if cities:
        s = _union_from_dict(onto["by_city"], cities)
        if s:
            cand = s if cand is None else (cand & s)

    # ---- councils ----
    councils = _normalize_list(councils)
    if councils:
        s = _union_from_dict(onto["by_council"], councils)
        if s:
            cand = s if cand is None else (cand & s)

    # ---- persons ----
    persons = _normalize_list(persons)
    if persons:
        s = _union_from_dict(onto["by_person"], persons)
        if s:
            cand = s if cand is None else (cand & s)

    # ---- parties ----
    parties = _normalize_list(parties)
    if parties:
        s = _union_from_dict(onto["by_party"], parties)
        if s:
            cand = s if cand is None else (cand & s)

    # ---- TAG filtering ----
    tag_keywords = _normalize_list(tag_keywords)
    if tag_keywords:
        tag_col = df["tag"].fillna("").astype(str)
        # 키워드들에 대한 OR 검색
        pattern = "|".join(re.escape(t) for t in tag_keywords)
        mask = tag_col.str.contains(pattern, regex=True, case=False, na=False)
        s = set(df[mask].index.tolist())
        if s:
            cand = s if cand is None else (cand & s)

    # 아무 필터도 없으면 전체
    if cand is None:
        cand = set(range(len(df)))

    return np.array(sorted(cand), dtype="int64")


# --------------------------------------------------
# 6. importance 우선 검색 (3 → 2)
# --------------------------------------------------
def search_topk_with_importance(
    df,
    cand_idx: np.ndarray,
    q_vec: np.ndarray,
    top_k: int = 5,
    min_score_primary: float = 0.2,
):
    """
    1순위: importance=3만 대상으로 검색
      - top1 score >= min_score_primary 이고 결과 개수가 >= top_k 이면 그대로 사용
    2순위: (필요 시) importance=2도 포함해서 다시 검색
    최종: score 기준 내림차순으로 top_k 반환

    반환: [(score, real_idx), ...]
    """
    results: List[tuple] = []

    imp_series = df.iloc[cand_idx]["importance"].fillna(0).astype(int)

    # ----- 1) importance=3 subset -----

