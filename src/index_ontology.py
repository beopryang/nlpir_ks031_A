# src/index_faiss.py  (예시 파일명)

import pandas as pd
import numpy as np
import faiss
from paths import DATA_DIR   # ★ 추가

# --------------------------------------------------
# 경로 설정
# --------------------------------------------------
PARQUET_PATH = DATA_DIR / "segments_all.parquet"
INDEX_DIR    = DATA_DIR / "faiss_index"
INDEX_PATH   = INDEX_DIR / "all.index"

INDEX_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 1) 데이터 로드
# --------------------------------------------------
df = pd.read_parquet(PARQUET_PATH)
print("rows:", len(df))
print("columns:", df.columns.tolist())

# --------------------------------------------------
# 2) embedding 행렬 준비
# --------------------------------------------------
emb = np.vstack(df["embedding"].values).astype("float32")
print("embedding matrix shape:", emb.shape)  # (N, d)

# --------------------------------------------------
# 3) FAISS 인덱스 생성
# --------------------------------------------------
d = emb.shape[1]
index = faiss.IndexFlatIP(d)
index.add(emb)

# --------------------------------------------------
# 4) 저장
# --------------------------------------------------
faiss.write_index(index, str(INDEX_PATH))   # ★ FAISS는 문자열 경로 요구

print("✔ 통합 FAISS 인덱스 저장 완료:", INDEX_PATH)
