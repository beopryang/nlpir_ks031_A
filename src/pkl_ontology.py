# src/index_ontology.py (예시 파일명)
import math, re, pickle
from collections import defaultdict

import pandas as pd

from paths import DATA_DIR  # ★ 추가: 프로젝트 공통 경로 사용

# --------------------------------------------------
# 경로 설정
# --------------------------------------------------
PARQUET_PATH = DATA_DIR / "segments_all.parquet"
OUT_DIR = DATA_DIR / "ontology_index"
OUT_PATH = OUT_DIR / "all.pkl"

OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(PARQUET_PATH)
print("rows:", len(df))

by_city    = defaultdict(list)
by_council = defaultdict(list)
by_person  = defaultdict(list)   # questioner/answerer/moderator 통합
by_party   = defaultdict(list)
by_tag     = defaultdict(list)

def norm_str(x):
    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None
    s = str(x).strip()
    return s if s else None

for idx, row in df.iterrows():
    city = norm_str(row.get("city"))
    if city:
        by_city[city].append(idx)

    council = norm_str(row.get("council"))
    if council:
        by_council[council].append(idx)

    for col in ["questioner", "answerer", "moderator"]:
        name = norm_str(row.get(col))
        if name:
            by_person[name].append(idx)

    party = norm_str(row.get("party"))
    if party:
        by_party[party].append(idx)

    tag_val = row.get("tag")
    tags = []
    if isinstance(tag_val, (list, tuple)):
        tags = [norm_str(t) for t in tag_val]
    else:
        t = norm_str(tag_val)
        if t:
            if any(sep in t for sep in [",", "/", "|"]):
                raw = re.split(r"[,/|]", t)
                tags = [norm_str(x) for x in raw]
            else:
                tags = [t]
    for t in tags:
        if t:
            by_tag[t].append(idx)

ontology = {
    "by_city": dict(by_city),
    "by_council": dict(by_council),
    "by_person": dict(by_person),
    "by_party": dict(by_party),
    "by_tag": dict(by_tag),
}

with open(OUT_PATH, "wb") as f:
    pickle.dump(ontology, f)

print("✔ 통합 온톨로지 저장 완료:", OUT_PATH)
print("도시:", list(ontology["by_city"].keys()))
print("정당:", list(ontology["by_party"].keys()))
