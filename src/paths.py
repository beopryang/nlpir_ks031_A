# src/paths.py
from pathlib import Path

# src/paths.py 기준으로 두 단계 위가 프로젝트 루트
ROOT_DIR = Path(__file__).resolve().parent.parent

CONFIG_DIR = ROOT_DIR / "config"
DATA_DIR   = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"

# 필요하면 서브폴더도 여기서 정의
NAIVE_RESULTS_DIR    = RESULTS_DIR / "naive"
ONTOLOGY_RESULTS_DIR = RESULTS_DIR / "ontology"
