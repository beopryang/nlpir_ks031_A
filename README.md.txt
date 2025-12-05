# **한국어 | [English](#english-version) | [日本語](#japanese-version) | [中文-简体](#chinese-simplified-version) | [中文-繁體](#chinese-traditional-version) | [Deutsch](#german-version) | [ไทย](#thai-version) | [Tiếng Việt](#vietnamese-version) | [Қазақша](#kazakh-version) | [Nederlands](#dutch-version) | [Русский](#russian-version) | [Français](#french-version) | [Español](#spanish-version) | [Hrvatski](#croatian-version) | [Eesti](#estonian-version)**

---

필요하면 **라트비아어 / 리투아니아어 / 우크라이나어 / 포르투갈어 / 이탈리아어 / 아랍어** 등도 추가해 드릴까요?


---

# 🏛 RAG를 활용한 의회 회의록 기반 기사문 생성 시스템 개발

<a name="korean-version"></a>

**Development of a News Article Generation System for Deliberation Records from Korean Legislatures Using RAG**

---

## 📌 개요 (Overview)

이 저장소는 광주광역시의회와 서울특별시의회 회의록을 기반으로,
**온톨로지 기반 메타데이터 필터링 + 임베딩 기반 유사도 검색을 결합한 하이브리드 RAG 시스템**을 구현한 연구 코드를 포함한다.

본 연구는 다음 두 retrieval 전략을 비교한다:

1. **Naive RAG** – 전처리 없이 전체 세그먼트를 대상으로 한 임베딩 기반 검색
2. **Ontology RAG** – 의회·발언자·정당·위원회 등 메타데이터를 필터링한 뒤 임베딩 기반 검색 수행

이를 통해 구조적 필터링이

* 사실성(factuality)
* 관련성(topical relevance)
* 검색 안정성(stability)
* 오류 방지(error robustness)

측면에서 어떤 개선을 제공하는지 체계적으로 평가한다.

데이터 기간은 **2022년 7월 ~ 2025년 10월**,
평가 질의는 **100개의 벤치마크 질의문**으로 구성된다.

---

# 📁 저장소 구조

```
root/
├─ src/                     # RAG 검색, 온톨로지, 평가 코드
├─ config/                  # LLM 플래너 & 평가 설정
├─ results/                 # 결과 및 평가 파일 (GitHub 포함)
└─ data/                    # 대용량 원본 데이터 (Google Drive 제공)
```

---

# 📊 1. 데이터 설명

## 1.1 results/ (GitHub 포함)

이 폴더에는 실험 과정에서 생성된 최종 결과물들이 포함된다.

### 🔍 Retrieval 결과

* `naive_rag_results_top5.csv`
* `ontology_rag_results_tasks.csv`

### 📰 생성된 기사문

* `naive_rag_articles.csv`
* `naive_rag_articles.jsonl`
* `ontology_rag_articles.csv`
* `ontology_rag_articles.jsonl`

### ✔ 사실성 평가 결과

* `eval_top5_truth_naive_top5_absolute.csv`
* `eval_top5_truth_onto_top5_absolute.csv`

### 🧪 평가 플랜

* `eval_plans_onto_top5.json`

### 🧾 평가용 질의문

* `test_queries.csv` (100개)

모든 파일은 재현성을 위해 저장소에 포함되어 있다.

---

## 1.2 data/ (Google Drive — 대용량 데이터)

원본 데이터는 용량 문제로 GitHub에 포함되지 않았으며, Google Drive로 제공된다.
Drive에는 **세 개의 파일만** 포함된다.

📥 다운로드 링크
**[https://drive.google.com/drive/folders/1_LP9o4K7Z6XR5xku7bEJc15pYAWp4hpP?usp=sharing](https://drive.google.com/drive/folders/1_LP9o4K7Z6XR5xku7bEJc15pYAWp4hpP?usp=sharing)**

### 포함된 파일

| 파일명                        | 내용                    |
| -------------------------- | --------------------- |
| `minutes.parquet`          | 전체 회의록(광주+서울) 원문      |
| `segments_all.parquet`     | 발언자 단위 세그먼트 전체 데이터    |
| `base_minutes_rag.parquet` | Naive RAG용 단순 전처리 데이터 |

### 로컬 배치 경로

```
root/data/
    minutes.parquet
    segments_all.parquet
    base_minutes_rag.parquet
```

---

# ⚙️ 2. 코드 설명 (src/)

### 🔹 `search_naive.py`

전체 세그먼트를 대상으로 임베딩 기반 검색 수행 (FAISS + cosine similarity)

### 🔹 `search_ontology.py`

온톨로지 기반 메타데이터 필터링(의회·인물·정당·위원회) 후 임베딩 검색 수행
→ 잘못된 매칭 제거로 사실성 향상

### 🔹 `generate_naive.py`

Naive RAG Top-5 검색 결과로 기사문 생성
LLM: gpt-4.1-mini

### 🔹 `generate_ontology.py`

Ontology RAG Top-5 결과 기반 기사문 생성
LLM: gpt-4.1-mini

### 🔹 `evaluate_absolute.py`

절대 평가 방식으로 사실성·관련성 점수 계산
다음 오류는 fact_ok=0 처리:

* WRONG_COUNCIL
* WRONG_PERSON
* WRONG_PARTY

### 🔹 `index_ontology.py`

온톨로지 인덱스 생성 및 캐싱

### 🔹 `pkl_ontology.py`

온톨로지 메타데이터 직렬화

### 🔹 `paths.py`

모든 파일 경로 집약 관리

---

# 📑 3. 실험 설계

### 데이터셋

* 의회: 광주광역시의회, 서울특별시의회
* 기간: 2022.07 ~ 2025.10
* 평가 질의: 100개

### 비교 대상 모델

| 모델           | 설명                   |
| ------------ | -------------------- |
| Naive RAG    | 전체 세그먼트 기반 임베딩 기반 검색 |
| Ontology RAG | 메타데이터 필터링 + 임베딩 검색   |

### 사용 모델

| 용도    | 모델                     |
| ----- | ---------------------- |
| 기사 생성 | gpt-4.1-mini           |
| 평가자   | gpt-4.1-mini           |
| 임베딩   | text-embedding-3-large |

---

# 📈 4. 평가 방법

### ✔ 1) 사실성 평가

다음 오류 유형을 **강한 사실 오류**로 간주:

* WRONG_COUNCIL
* WRONG_PERSON
* WRONG_PARTY

### ✔ 2) 관련성 평가 (topic_score)

LLM 기반 과제 분석 후
세그먼트의 주제 일치도를 1~10점으로 평가.

---

# 📊 5. 결과 요약

## 🔥 1) 사실성 오류율

| 모델           | 세그먼트 수 | 오류 수 | 오류율        |
| ------------ | ------ | ---- | ---------- |
| Naive RAG    | 500    | 161  | **32.20%** |
| Ontology RAG | 610    | 43   | **7.05%**  |

📉 **25.15%p 감소 (약 78% 상대 감소)**

---

## 🎯 2) 관련성 평균 점수

| 모델           | 평균 점수(10점 만점) |
| ------------ | ------------- |
| Naive RAG    | 5.77          |
| Ontology RAG | 6.54          |

📈 **7.66% 향상**, catastrophic failure 급의 0점 사례 거의 제거됨

---

## 📌 핵심 결론

Ontology RAG는 Naive RAG 대비

* 사실 오류 대폭 감소
* 주제 적합성 향상
* catastrophic failure 제거
* 검색 일관성·안정성 향상

등의 개선 효과를 보였다.

---

# 🚀 6. 실행 방법

### 1) 클론

```bash
git clone https://github.com/beopryang/nlpir_ks031_A.git
cd nlpir_ks031_A/root
```

### 2) 패키지 설치

```bash
pip install -r requirements.txt
```

### 3) Google Drive 데이터 다운로드

```
root/data/
    minutes.parquet
    segments_all.parquet
    base_minutes_rag.parquet
```

### 4) 검색 실행

```bash
python src/search_naive.py
python src/search_ontology.py
```

### 5) 기사문 생성

```bash
python src/generate_naive.py
python src/generate_ontology.py
```

### 6) 평가 실행

```bash
python src/evaluate_absolute.py
```

---

# 🧪 7. 재현성

* 모든 결과(`results/`) GitHub 포함
* LLM 프롬프트 및 구체적 평가 계획(`config/`) 제공
* 검색/생성/평가 코드(`src/`) 전체 공개
* 대용량 데이터(`data/`) Google Drive 제공
* 생성형 모델 특성상 문장 일부는 변동 가능하나
  **사실성/관련성 수치는 재현 가능**

---

# English Version

<a name="english-version"></a>

# 🏛 Development of a News Article Generation System for Deliberation Records from Korean Legislatures Using RAG

---

## 📌 Overview

This repository contains the research code for a **hybrid RAG (Retrieval-Augmented Generation) system** that processes deliberation records from the Gwangju Metropolitan Council and the Seoul Metropolitan Council.
The system integrates **ontology-based metadata filtering** with **embedding-based similarity search**.

The study compares two retrieval strategies:

1. **Naive RAG** – pure embedding-based retrieval without preprocessing
2. **Ontology RAG** – retrieval constrained by council, speaker, party, and committee metadata prior to embedding search

Through this comparison, the research evaluates how structural filtering improves:

* factuality
* topical relevance
* retrieval stability
* error robustness

The dataset covers **July 2022 – October 2025**, and evaluation is conducted using **100 benchmark queries**.

---

# 📁 Repository Structure

```
root/
├─ src/                     # RAG retrieval, ontology, and evaluation code
├─ config/                  # LLM planner & evaluation configuration
├─ results/                 # Retrieval/output/evaluation results (included)
└─ data/                    # Large original data (provided via Google Drive)
```

---

# 📊 1. Data Description

## 1.1 results/ (included in GitHub)

This folder contains all generated outputs from the experiments.

### 🔍 Retrieval Results

* `naive_rag_results_top5.csv`
* `ontology_rag_results_tasks.csv`

### 📰 Generated Articles

* `naive_rag_articles.csv`
* `naive_rag_articles.jsonl`
* `ontology_rag_articles.csv`
* `ontology_rag_articles.jsonl`

### ✔ Factuality Evaluation

* `eval_top5_truth_naive_top5_absolute.csv`
* `eval_top5_truth_onto_top5_absolute.csv`

### 🧪 Evaluation Plans

* `eval_plans_onto_top5.json`

### 🧾 Benchmark Queries

* `test_queries.csv` (100 queries)

All files are included to ensure full reproducibility.

---

## 1.2 data/ (Google Drive — large files)

Large original data files are not stored on GitHub and are provided via Google Drive.
Only **three files** are included.

📥 Download Link
**[https://drive.google.com/drive/folders/1_LP9o4K7Z6XR5xku7bEJc15pYAWp4hpP?usp=sharing](https://drive.google.com/drive/folders/1_LP9o4K7Z6XR5xku7bEJc15pYAWp4hpP?usp=sharing)**

### Included Files

| File Name                  | Description                             |
| -------------------------- | --------------------------------------- |
| `minutes.parquet`          | Original full minutes (Gwangju + Seoul) |
| `segments_all.parquet`     | All speaker-level segments              |
| `base_minutes_rag.parquet` | Preprocessed base data for Naive RAG    |

### Local Placement

Place the downloaded files under:

```
root/data/
    minutes.parquet
    segments_all.parquet
    base_minutes_rag.parquet
```

---

# ⚙️ 2. Code Description (src/)

### 🔹 `search_naive.py`

Performs embedding-based retrieval over all segments (FAISS + cosine similarity).

### 🔹 `search_ontology.py`

Filters segments using ontology metadata (council, speaker, party, committee),
then performs embedding-based retrieval.
→ Prevents mismatches and improves factuality.

### 🔹 `generate_naive.py`

Generates news-style articles from Naive RAG Top-5 retrieval results.
LLM: **gpt-4.1-mini**

### 🔹 `generate_ontology.py`

Generates articles using Ontology RAG Top-5 retrieval results.
LLM: **gpt-4.1-mini**

### 🔹 `evaluate_absolute.py`

Performs absolute evaluation of factuality and topical relevance.

* WRONG_COUNCIL
* WRONG_PERSON
* WRONG_PARTY
  → treated as *strong factual errors* (fact_ok = 0)

Topical relevance is scored from 1–10.

### 🔹 `index_ontology.py`

Constructs and caches ontology metadata indexes.

### 🔹 `pkl_ontology.py`

Serializes ontology metadata structures.

### 🔹 `paths.py`

Centralized configuration for directory paths.

---

# 📑 3. Experimental Setup

### Dataset

* Councils: Gwangju Metropolitan Council, Seoul Metropolitan Council
* Period: July 2022 – October 2025
* Evaluation: 100 benchmark queries

### Compared Models

| Model        | Description                                 |
| ------------ | ------------------------------------------- |
| Naive RAG    | Embedding-based retrieval over all segments |
| Ontology RAG | Metadata filtering + embedding retrieval    |

### Models Used

| Purpose            | Model                  |
| ------------------ | ---------------------- |
| Article generation | gpt-4.1-mini           |
| Evaluation         | gpt-4.1-mini           |
| Embeddings         | text-embedding-3-large |

---

# 📈 4. Evaluation Method

### ✔ 1) Factuality Evaluation (fact-level)

The following are treated as **strong factual errors**:

* WRONG_COUNCIL
* WRONG_PERSON
* WRONG_PARTY

Ontology RAG: removes EMPTY_SEGMENT queries from the dataset
Naive RAG: assigns **0-point penalty** for any retrieved segments belonging to EMPTY_SEGMENT queries

---

### ✔ 2) Topical Relevance Evaluation (topic_score)

LLM evaluates how well retrieved segments match the intended topic.
Scored on a 1–10 scale.

---

# 📊 5. Results Summary

## 🔥 1) Factual Error Rate

| Model        | #Segments | #Errors | Error Rate |
| ------------ | --------- | ------- | ---------- |
| Naive RAG    | 500       | 161     | **32.20%** |
| Ontology RAG | 610       | 43      | **7.05%**  |

**→ 25.15 percentage points reduction (≈ 78% relative reduction)**

---

## 🎯 2) Mean Topical Relevance

| Model        | Mean Score (10 max) |
| ------------ | ------------------- |
| Naive RAG    | 5.77                |
| Ontology RAG | 6.54                |

**→ +7.66% improvement**, with far fewer catastrophic low scores.

---

## 📌 Key Findings

Compared to Naive RAG, Ontology RAG provides:

* substantial reduction in factual errors
* higher topical relevance
* elimination of catastrophic 0-point retrieval failures
* more consistent and stable retrieval outcomes

These improvements stem from enforcing structural constraints before semantic retrieval.

---

# 🚀 6. How to Run

### 1) Clone the Repository

```bash
git clone https://github.com/beopryang/nlpir_ks031_A.git
cd nlpir_ks031_A/root
```

### 2) Install Dependencies

```bash
pip install -r requirements.txt
```

### 3) Download Data from Google Drive

Place files under:

```
root/data/
    minutes.parquet
    segments_all.parquet
    base_minutes_rag.parquet
```

### 4) Run Retrieval

```bash
python src/search_naive.py
python src/search_ontology.py
```

### 5) Generate Articles

```bash
python src/generate_naive.py
python src/generate_ontology.py
```

### 6) Evaluate

```bash
python src/evaluate_absolute.py
```

---

# 🧪 7. Reproducibility

* All output files in `results/` are included
* All prompts/configurations in `config/` are provided
* All retrieval/generation/evaluation code is in `src/`
* Large data files in `data/` are publicly shared via Google Drive
* While LLM-generated text may vary slightly,
  factuality and relevance metrics can be reproduced consistently

---

# 日本語版

<a name="japanese-version"></a>

# 🏛 RAG を活用した韓国の議会会議録に基づくニュース記事生成システムの開発

**Development of a News Article Generation System for Deliberation Records from Korean Legislatures Using RAG**

---

## 📌 概要 (Overview)

本リポジトリは、**光州広域市議会**および**ソウル特別市議会**の会議録を対象として、
**オンタロジーに基づくメタデータフィルタリング**と**埋め込みベースの類似度検索**を組み合わせた
ハイブリッド RAG（Retrieval-Augmented Generation）システムの研究コードを収録しています。

本研究では以下の 2 種類の検索戦略を比較します：

1. **Naive RAG** – 前処理なしで全セグメントから埋め込み検索を実行
2. **Ontology RAG** – 議会・発言者・政党・委員会などのメタデータをフィルタリングした後に埋め込み検索を実行

これにより、構造的フィルタリングが以下の点にどのような改善をもたらすかを体系的に評価します：

* 事実性（factuality）
* 主題適合性（topical relevance）
* 検索の安定性（stability）
* エラー耐性（error robustness）

対象データ期間：**2022年7月〜2025年10月**
評価用クエリ：**100件のベンチマーククエリ**

---

# 📁 リポジトリ構成

```
root/
├─ src/                     # RAG 検索・オンタロジー・評価コード
├─ config/                  # LLM プランナーおよび評価設定
├─ results/                 # 検索結果・生成記事・評価結果（GitHub に含む）
└─ data/                    # 大容量データ（Google Drive にて提供）
```

---

# 📊 1. データ説明

## 1.1 results/（GitHub に含まれるデータ）

実験により生成された最終的な成果物を収録しています。

### 🔍 検索結果

* `naive_rag_results_top5.csv`
* `ontology_rag_results_tasks.csv`

### 📰 生成されたニュース記事

* `naive_rag_articles.csv`
* `naive_rag_articles.jsonl`
* `ontology_rag_articles.csv`
* `ontology_rag_articles.jsonl`

### ✔ 事実性評価結果

* `eval_top5_truth_naive_top5_absolute.csv`
* `eval_top5_truth_onto_top5_absolute.csv`

### 🧪 評価プラン

* `eval_plans_onto_top5.json`

### 🧾 ベンチマーククエリ

* `test_queries.csv`（100件）

すべてのファイルを含めることで再現性を確保しています。

---

## 1.2 data/（Google Drive — 大容量データ）

大容量の元データは GitHub に含めず、Google Drive にて配布しています。
Drive には **次の 3 ファイルのみ**が含まれます。

📥 ダウンロードリンク
**[https://drive.google.com/drive/folders/1_LP9o4K7Z6XR5xku7bEJc15pYAWp4hpP?usp=sharing](https://drive.google.com/drive/folders/1_LP9o4K7Z6XR5xku7bEJc15pYAWp4hpP?usp=sharing)**

### 含まれるファイル

| ファイル名                      | 内容                   |
| -------------------------- | -------------------- |
| `minutes.parquet`          | 議会会議録全文（光州＋ソウル）      |
| `segments_all.parquet`     | 発言者単位の全セグメント         |
| `base_minutes_rag.parquet` | Naive RAG 用の単純前処理データ |

### ローカル配置

```
root/data/
    minutes.parquet
    segments_all.parquet
    base_minutes_rag.parquet
```

---

# ⚙️ 2. コード説明（src/）

### 🔹 `search_naive.py`

全セグメントを対象に埋め込み検索を実行（FAISS + コサイン類似度）

### 🔹 `search_ontology.py`

議会・発言者・政党・委員会などのオンタロジーメタデータを用いて
検索対象をフィルタリングした後に埋め込み検索を実行
→ 誤マッチを防ぎ、事実性を向上

### 🔹 `generate_naive.py`

Naive RAG の Top-5 検索結果からニュース記事を生成
LLM: **gpt-4.1-mini**

### 🔹 `generate_ontology.py`

Ontology RAG の Top-5 結果を用いて記事生成
LLM: **gpt-4.1-mini**

### 🔹 `evaluate_absolute.py`

事実性・主題適合性を絶対評価方式で採点

* WRONG_COUNCIL
* WRONG_PERSON
* WRONG_PARTY
  → **強い事実誤り（fact_ok = 0）**として扱う
  主題適合性は 1〜10 点で評価

### 🔹 `index_ontology.py`

オンタロジーインデックスの構築とキャッシュ

### 🔹 `pkl_ontology.py`

オンタロジーメタデータのシリアライズ

### 🔹 `paths.py`

パス設定の集中管理

---

# 📑 3. 実験設計

### データセット

* 対象議会：光州広域市議会、ソウル特別市議会
* 期間：2022年7月〜2025年10月
* 評価クエリ：100件

### 比較対象モデル

| モデル          | 説明                  |
| ------------ | ------------------- |
| Naive RAG    | 全セグメントに対する埋め込み検索    |
| Ontology RAG | メタデータフィルタリング＋埋め込み検索 |

### 使用モデル

| 用途     | モデル                    |
| ------ | ---------------------- |
| 記事生成   | gpt-4.1-mini           |
| 評価     | gpt-4.1-mini           |
| 埋め込み生成 | text-embedding-3-large |

---

# 📈 4. 評価方法

### ✔ 1) 事実性評価（fact-level）

以下の誤りは **強い事実誤り** として扱う：

* WRONG_COUNCIL
* WRONG_PERSON
* WRONG_PARTY

Ontology RAG：該当クエリにセグメントが存在しない場合（EMPTY_SEGMENT）は評価母数から除外
Naive RAG：同クエリで取得されたセグメントには **0 点ペナルティ** を付与

---

### ✔ 2) 主題適合性評価（topic_score）

LLM がクエリとセグメントの主題一致度を 1〜10 点で採点。

---

# 📊 5. 結果サマリー

## 🔥 1) 事実誤り率

| モデル          | セグメント数 | 誤り数 | 誤り率        |
| ------------ | ------ | --- | ---------- |
| Naive RAG    | 500    | 161 | **32.20%** |
| Ontology RAG | 610    | 43  | **7.05%**  |

**→ 25.15 ポイントの改善（約 78% の相対削減）**

---

## 🎯 2) 主題適合性（平均）

| モデル          | 平均点（10 点満点） |
| ------------ | ----------- |
| Naive RAG    | 5.77        |
| Ontology RAG | 6.54        |

**→ 7.66% の改善**, 最小値も大幅に改善

---

## 📌 主要知見

Ontology RAG は Naive RAG と比較して：

* 事実誤りを大幅に削減
* 主題適合性を向上
* 0 点レベルの catastrophic failure を解消
* 検索結果の安定性と一貫性を強化

といった改善効果を示した。

---

# 🚀 6. 実行方法

### 1) リポジトリのクローン

```bash
git clone https://github.com/beopryang/nlpir_ks031_A.git
cd nlpir_ks031_A/root
```

### 2) 依存関係のインストール

```bash
pip install -r requirements.txt
```

### 3) Google Drive からデータをダウンロード

```
root/data/
    minutes.parquet
    segments_all.parquet
    base_minutes_rag.parquet
```

### 4) 検索の実行

```bash
python src/search_naive.py
python src/search_ontology.py
```

### 5) 記事生成

```bash
python src/generate_naive.py
python src/generate_ontology.py
```

### 6) 評価

```bash
python src/evaluate_absolute.py
```

---

# 🧪 7. 再現性

* `results/` 内のすべての成果物を GitHub に収録
* `config/` の全プロンプト・評価設定を公開
* RAG 検索／生成／評価コードを `src/` に完全収録
* 大容量データは Google Drive にて共有
* 生成系 LLM の特性上、文章は一部変動する可能性があるが、
  **事実性・主題適合性指標は再現可能**


---

# 🇨🇳 中文版（简体）

<a name="chinese-simplified-version"></a>

---

# 🏛 基于 RAG 的韩国议会会议记录新闻生成系统开发

---

## 📌 概述（Overview）

本仓库包含一个 **混合式 RAG（Retrieval-Augmented Generation）系统**的研究代码，用于处理韩国**光州广域市议会**与**首尔特别市议会**的会议记录。
该系统结合了 **基于本体的元数据过滤** 与 **基于嵌入向量的相似度检索**。

本研究比较了两种检索策略：

1. **Naive RAG** —— 对所有段落直接进行嵌入检索，不做任何预处理
2. **Ontology RAG** —— 结合元数据（议会、发言者、政党、委员会）进行过滤后再执行嵌入检索

通过比较，本研究评估结构化过滤对以下方面的改进效果：

* 事实准确性（factuality）
* 主题相关性（topical relevance）
* 检索稳定性（stability）
* 错误鲁棒性（error robustness）

数据集覆盖 **2022 年 7 月至 2025 年 10 月**，并使用 **100 条基准查询**进行评估。

---

# 📁 仓库结构（Repository Structure）

```
root/
├─ src/                     # RAG 检索、本体、评估代码
├─ config/                  # LLM 规划器 & 评估配置
├─ results/                 # 检索/生成/评估结果（已包含）
└─ data/                    # 大型原始数据（Google Drive 提供）
```

---

# 📊 1. 数据说明（Data Description）

## 1.1 results/（已包含于 GitHub）

该文件夹包含所有实验生成的最终结果。

### 🔍 检索结果

* `naive_rag_results_top5.csv`
* `ontology_rag_results_tasks.csv`

### 📰 生成的新闻文章

* `naive_rag_articles.csv`
* `naive_rag_articles.jsonl`
* `ontology_rag_articles.csv`
* `ontology_rag_articles.jsonl`

### ✔ 事实性评估

* `eval_top5_truth_naive_top5_absolute.csv`
* `eval_top5_truth_onto_top5_absolute.csv`

### 🧪 评估计划

* `eval_plans_onto_top5.json`

### 🧾 基准查询（100条）

* `test_queries.csv`

所有文件均已包含，以确保可复现性。

---

## 1.2 data/（Google Drive — 大文件）

原始数据文件较大，因此未上传到 GitHub，而是通过 Google Drive 提供。
仅包含 **3 个文件**：

📥 下载链接：
**[https://drive.google.com/drive/folders/1_LP9o4K7Z6XR5xku7bEJc15pYAWp4hpP?usp=sharing](https://drive.google.com/drive/folders/1_LP9o4K7Z6XR5xku7bEJc15pYAWp4hpP?usp=sharing)**

### 文件列表

| 文件名                        | 内容描述                |
| -------------------------- | ------------------- |
| `minutes.parquet`          | 光州+首尔全部会议记录原文       |
| `segments_all.parquet`     | 按发言者切分的全部段落数据       |
| `base_minutes_rag.parquet` | Naive RAG 用的简单预处理数据 |

### 本地放置方式

下载后存放于：

```
root/data/
    minutes.parquet
    segments_all.parquet
    base_minutes_rag.parquet
```

---

# ⚙️ 2. 代码说明（src/）

### 🔹 `search_naive.py`

对全部段落进行嵌入检索（FAISS + 余弦相似度）。

### 🔹 `search_ontology.py`

基于元数据（议会/发言者/政党/委员会）进行过滤后再执行嵌入检索。
→ 可避免错误匹配，提高事实性。

### 🔹 `generate_naive.py`

使用 Naive RAG Top-5 检索结果生成新闻文章。
LLM：**gpt-4.1-mini**

### 🔹 `generate_ontology.py`

使用 Ontology RAG Top-5 检索结果生成新闻文章。
LLM：**gpt-4.1-mini**

### 🔹 `evaluate_absolute.py`

执行事实性与主题相关性的绝对评价。

* WRONG_COUNCIL
* WRONG_PERSON
* WRONG_PARTY
  → 视为**严重事实性错误**（fact_ok = 0）

主题相关性以 1–10 分评估。

### 🔹 `index_ontology.py`

构建并缓存本体索引。

### 🔹 `pkl_ontology.py`

序列化本体元数据。

### 🔹 `paths.py`

统一管理路径配置。

---

# 📑 3. 实验设置（Experimental Setup）

### 数据集

* 议会：光州广域市议会、首尔特别市议会
* 时间：2022.07 – 2025.10
* 基准查询：100 条

### 比较模型

| 模型           | 描述           |
| ------------ | ------------ |
| Naive RAG    | 对所有段落执行嵌入检索  |
| Ontology RAG | 元数据过滤 + 嵌入检索 |

### 使用的模型

| 用途   | 模型                     |
| ---- | ---------------------- |
| 新闻生成 | gpt-4.1-mini           |
| 评估   | gpt-4.1-mini           |
| 嵌入模型 | text-embedding-3-large |

---

# 📈 4. 评估方法（Evaluation Method）

### ✔ 1) 事实性评估（fact-level）

以下类型视为严重事实错误：

* WRONG_COUNCIL
* WRONG_PERSON
* WRONG_PARTY

Ontology RAG：删除 EMPTY_SEGMENT 查询
Naive RAG：对这些查询检索到的所有段落给予 **0 分惩罚**

---

### ✔ 2) 主题相关性（topic_score）

LLM 评估检索段落与查询主题的匹配程度。
评分范围：1–10 分。

---

# 📊 5. 结果总结（Results Summary）

## 🔥 1) 事实性错误率

| 模型           | 段落数量 | 错误数 | 错误率        |
| ------------ | ---- | --- | ---------- |
| Naive RAG    | 500  | 161 | **32.20%** |
| Ontology RAG | 610  | 43  | **7.05%**  |

**→ 降低 25.15 个百分点（相对减少约 78%）**

---

## 🎯 2) 主题相关性平均分

| 模型           | 平均分（满分 10） |
| ------------ | ---------- |
| Naive RAG    | 5.77       |
| Ontology RAG | 6.54       |

**→ 提升 7.66%，并显著减少 0 分灾难性结果**

---

## 📌 核心结论

相比 Naive RAG，Ontology RAG 提供了：

* 显著减少事实性错误
* 更高的主题相关性
* 消除灾难性的 0 分检索结果
* 更稳定、更一致的输出

其性能优势来自于在语义检索之前施加结构化约束。

---

# 🚀 6. 使用方法（How to Run）

### 1) 克隆仓库

```bash
git clone https://github.com/beopryang/nlpir_ks031_A.git
cd nlpir_ks031_A/root
```

### 2) 安装依赖

```bash
pip install -r requirements.txt
```

### 3) 下载 Google Drive 数据

存放于：

```
root/data/
    minutes.parquet
    segments_all.parquet
    base_minutes_rag.parquet
```

### 4) 执行检索

```bash
python src/search_naive.py
python src/search_ontology.py
```

### 5) 生成新闻文章

```bash
python src/generate_naive.py
python src/generate_ontology.py
```

### 6) 运行评估

```bash
python src/evaluate_absolute.py
```

---

# 🧪 7. 可复现性（Reproducibility）

* 所有结果均包含在 `results/`
* 所有 LLM 配置与提示均包含在 `config/`
* 检索/生成/评估代码均包含在 `src/`
* 大型数据通过 Google Drive 公共提供
* 虽然 LLM 文本可能略有变化，但事实性与主题相关性分数可复现

---

# 📘 **Deutsche Version** (German Version)

<a name="german-version"></a>

# 🏛 Entwicklung eines Nachrichtengenerierungssystems für parlamentarische Beratungsprotokolle in Korea unter Verwendung von RAG

---

## 📌 Überblick

Dieses Repository enthält den Forschungscode für ein **hybrides RAG-System (Retrieval-Augmented Generation)**, das Beratungsprotokolle des Stadtrats von Gwangju und des Metropolrats von Seoul verarbeitet.
Das System kombiniert **ontologiebasierte Metadatenfilterung** mit **embeddingsbasierter Ähnlichkeitssuche**.

Die Studie vergleicht zwei Retrieval-Strategien:

1. **Naive RAG** – reine embeddingsbasierte Suche ohne Vorverarbeitung
2. **Ontology RAG** – Suche nach Filterung anhand von Metadaten (Rat, Redner, Partei, Ausschuss)

Durch diesen Vergleich bewertet die Forschung, wie strukturelle Filterung folgende Aspekte verbessert:

* Faktentreue (Factuality)
* thematische Relevanz
* Stabilität des Abrufs
* Robustheit gegenüber Fehlern

Der Datensatz umfasst den Zeitraum **Juli 2022 – Oktober 2025** und wird anhand von **100 Benchmark-Abfragen** evaluiert.

---

# 📁 Repository-Struktur

```
root/
├─ src/                     # RAG-Retrieval, Ontologie und Evaluationscode
├─ config/                  # LLM-Planner & Evaluationskonfiguration
├─ results/                 # Ergebnisse (auf GitHub enthalten)
└─ data/                    # Große Originaldaten (über Google Drive)
```

---

# 📊 1. Datenbeschreibung

## 1.1 results/ (in GitHub enthalten)

Dieser Ordner umfasst alle während des Experiments erzeugten Ausgabedateien.

### 🔍 Retrieval-Ergebnisse

* `naive_rag_results_top5.csv`
* `ontology_rag_results_tasks.csv`

### 📰 Generierte Artikel

* `naive_rag_articles.csv`
* `naive_rag_articles.jsonl`
* `ontology_rag_articles.csv`
* `ontology_rag_articles.jsonl`

### ✔ Faktenbasierte Evaluation

* `eval_top5_truth_naive_top5_absolute.csv`
* `eval_top5_truth_onto_top5_absolute.csv`

### 🧪 Evaluationspläne

* `eval_plans_onto_top5.json`

### 🧾 Benchmark-Abfragen

* `test_queries.csv` (100 Abfragen)

Alle Dateien sind für vollständige Reproduzierbarkeit enthalten.

---

## 1.2 data/ (Google Drive — große Dateien)

Die Originaldaten sind zu groß für GitHub und werden daher über Google Drive bereitgestellt.
Es sind **nur drei Dateien** enthalten:

📥 **Download-Link**
[https://drive.google.com/drive/folders/1_LP9o4K7Z6XR5xku7bEJc15pYAWp4hpP?usp=sharing](https://drive.google.com/drive/folders/1_LP9o4K7Z6XR5xku7bEJc15pYAWp4hpP?usp=sharing)

### Enthaltene Dateien

| Dateiname                  | Beschreibung                                      |
| -------------------------- | ------------------------------------------------- |
| `minutes.parquet`          | Vollständige Sitzungsprotokolle (Gwangju + Seoul) |
| `segments_all.parquet`     | Alle Redebeiträge auf Segmentebene                |
| `base_minutes_rag.parquet` | Vorverarbeitete Basisdaten für Naive RAG          |

### Lokale Ablage

```text
root/data/
    minutes.parquet
    segments_all.parquet
    base_minutes_rag.parquet
```

---

# ⚙️ 2. Codebeschreibung (src/)

### 🔹 `search_naive.py`

Embeddingsbasierte Suche über alle Segmente (FAISS + Kosinusähnlichkeit).

### 🔹 `search_ontology.py`

Filtert Segmente anhand von Ontologie-Metadaten (Rat, Redner, Partei, Ausschuss)
→ verhindert Fehlzuordnungen und erhöht die Faktentreue.

### 🔹 `generate_naive.py`

Erzeugt Nachrichtenartikel aus den Top-5-Treffern von Naive RAG.
LLM: **gpt-4.1-mini**

### 🔹 `generate_ontology.py`

Erzeugt Artikel auf Grundlage der Ontology-RAG-Ergebnisse.
LLM: **gpt-4.1-mini**

### 🔹 `evaluate_absolute.py`

Absolute Evaluation von Faktentreue und thematischer Relevanz.

* WRONG_COUNCIL
* WRONG_PERSON
* WRONG_PARTY → werden als **schwere Faktenfehler** (fact_ok = 0) gewertet

Thematische Relevanz wird auf einer Skala von 1–10 bewertet.

### 🔹 `index_ontology.py`

Erstellt und cached Ontologie-Metadatenindizes.

### 🔹 `pkl_ontology.py`

Serialisiert Ontologiestrukturen.

### 🔹 `paths.py`

Zentrale Verwaltung aller Dateipfade.

---

# 📑 3. Versuchsaufbau

### Datensatz

* Räte: Gwangju Metropolitan Council, Seoul Metropolitan Council
* Zeitraum: Juli 2022 – Oktober 2025
* Evaluation: 100 Benchmark-Abfragen

### Verglichene Modelle

| Modell       | Beschreibung                         |
| ------------ | ------------------------------------ |
| Naive RAG    | Embeddingssuche über alle Segmente   |
| Ontology RAG | Metadatenfilterung + Embeddingssuche |

### Verwendete Modelle

| Zweck              | Modell                 |
| ------------------ | ---------------------- |
| Artikelgenerierung | gpt-4.1-mini           |
| Evaluation         | gpt-4.1-mini           |
| Embeddings         | text-embedding-3-large |

---

# 📈 4. Evaluationsmethode

### ✔ 1) Faktenbasierte Evaluation

Folgende Fehler gelten als **schwere Faktenfehler**:

* WRONG_COUNCIL
* WRONG_PERSON
* WRONG_PARTY

Ontology RAG: Abfragen ohne einschlägige Segmente werden entfernt
Naive RAG: für solche Abfragen erhalten alle Segmente **0 Punkte**

---

### ✔ 2) Thematische Relevanz (topic_score)

Das LLM bewertet die thematische Übereinstimmung zwischen Abfrage und Segmenten (1–10 Punkte).

---

# 📊 5. Ergebniszusammenfassung

## 🔥 1) Fehlerquote (Factual Error Rate)

| Modell       | Segmente | Fehler | Fehlerquote |
| ------------ | -------- | ------ | ----------- |
| Naive RAG    | 500      | 161    | **32,20%**  |
| Ontology RAG | 610      | 43     | **7,05%**   |

➡ **Reduktion um 25,15 Prozentpunkte (≈ 78 % relative Verbesserung)**

---

## 🎯 2) Durchschnittliche thematische Relevanz

| Modell       | Durchschnitt (max. 10) |
| ------------ | ---------------------- |
| Naive RAG    | 5,77                   |
| Ontology RAG | 6,54                   |

➡ **+7,66 % Verbesserung**, deutlich weniger Ausreißer mit 0 Punkten.

---

## 📌 Zentrale Erkenntnisse

Ontology RAG bietet im Vergleich zu Naive RAG:

* deutliche Verringerung von Faktenfehlern
* höhere thematische Relevanz
* vollständige Eliminierung katastrophaler Fehlretrievals (0 Punkte)
* stabilere und konsistentere Ergebnisse

Die Verbesserungen resultieren aus strukturellen Filtern vor der semantischen Suche.

---

# 🚀 6. Ausführung

### 1) Repository klonen

```bash
git clone https://github.com/beopryang/nlpir_ks031_A.git
cd nlpir_ks031_A/root
```

### 2) Abhängigkeiten installieren

```bash
pip install -r requirements.txt
```

### 3) Daten herunterladen

Dateien in folgenden Ordner legen:

```
root/data/
```

### 4) Retrieval starten

```bash
python src/search_naive.py
python src/search_ontology.py
```

### 5) Artikel generieren

```bash
python src/generate_naive.py
python src/generate_ontology.py
```

### 6) Evaluation durchführen

```bash
python src/evaluate_absolute.py
```

---

# 🧪 7. Reproduzierbarkeit

* Alle Ergebnisdateien liegen im Ordner `results/`
* Alle Prompts und Konfigurationen in `config/` sind offen gelegt
* Vollständiger Code für Retrieval/Generierung/Evaluation befindet sich in `src/`
* Große Datensätze sind über Google Drive verfügbar
* Trotz kleiner natürlicher Variationen in generierten Texten
  bleiben **Faktentreue und Relevanzwerte reproduzierbar**.

---

아래는 **전체 English Version을 태국어(ภาษาไทย)**로 정확하고 자연스럽게 번역한 버전입니다.
전문적·기술적 문체를 유지하면서도 태국어 문서 스타일에 맞게 표현했습니다.

---

# 🇹🇭 Thai Version

<a name="thai-version"></a>

# 🏛 การพัฒนาระบบสร้างบทความข่าวจากบันทึกการประชุมสภาของเกาหลี โดยใช้ RAG

---

## 📌 บทนำ (Overview)

รีโพสิตอรีนี้ประกอบด้วยโค้ดการวิจัยสำหรับระบบ **RAG แบบไฮบริด (Retrieval-Augmented Generation)**
ซึ่งใช้ประมวลผลบันทึกการประชุมของสภานครกวางจูและสภานครโซล ระบบนี้ผสานการทำงานระหว่าง

* **การกรองเมทาดาทาเชิงออนโทโลจี (ontology-based metadata filtering)**
* **การค้นหาความใกล้เคียงด้วยเวกเตอร์ฝังตัว (embedding-based similarity search)**

งานวิจัยนี้เปรียบเทียบกลยุทธ์การค้นหาสองแบบคือ:

1. **Naive RAG** – การค้นหาจากทุกเซกเมนต์โดยตรงโดยไม่ผ่านการกรองล่วงหน้า
2. **Ontology RAG** – การกรองด้วยเมทาดาทา (สภา ผู้พูด พรรค และคณะกรรมาธิการ) ก่อนจึงทำการค้นหาด้วยเวกเตอร์ฝังตัว

ด้วยการเปรียบเทียบนี้ งานวิจัยวิเคราะห์ว่าการกรองเชิงโครงสร้างช่วยเพิ่มประสิทธิภาพด้านใดบ้าง เช่น

* ความถูกต้องเชิงข้อเท็จจริง (factuality)
* ความสอดคล้องเชิงประเด็น (topical relevance)
* ความเสถียรของการค้นคืนข้อมูล (retrieval stability)
* ความสามารถในการลดความผิดพลาด (error robustness)

ข้อมูลครอบคลุมช่วง **กรกฎาคม 2022 – ตุลาคม 2025**
การประเมินผลทำด้วย **คำถาม 100 ข้อ (benchmark queries)**

---

# 📁 โครงสร้างของรีโพสิตอรี

```
root/
├─ src/                     # โค้ดค้นคืนข้อมูล RAG, ออนโทโลจี และการประเมินผล
├─ config/                  # การตั้งค่า LLM สำหรับ planner และ evaluator
├─ results/                 # ผลลัพธ์จากการทดลอง (รวมใน GitHub)
└─ data/                    # ข้อมูลขนาดใหญ่ (อยู่บน Google Drive)
```

---

# 📊 1. รายละเอียดข้อมูล (Data Description)

## 1.1 results/ (รวมอยู่ใน GitHub)

โฟลเดอร์นี้ประกอบด้วยผลลัพธ์ทั้งหมดที่ได้จากกระบวนการทดลอง

### 🔍 ผลการค้นคืน (Retrieval Results)

* `naive_rag_results_top5.csv`
* `ontology_rag_results_tasks.csv`

### 📰 บทความข่าวที่สร้างขึ้น (Generated Articles)

* `naive_rag_articles.csv`
* `naive_rag_articles.jsonl`
* `ontology_rag_articles.csv`
* `ontology_rag_articles.jsonl`

### ✔ ผลการประเมินด้านข้อเท็จจริง (Factuality Evaluation)

* `eval_top5_truth_naive_top5_absolute.csv`
* `eval_top5_truth_onto_top5_absolute.csv`

### 🧪 โครงร่างการประเมิน (Evaluation Plans)

* `eval_plans_onto_top5.json`

### 🧾 คำถามประเมินผล (Benchmark Queries)

* `test_queries.csv` (จำนวน 100 ข้อ)

---

## 1.2 data/ (Google Drive — ขนาดใหญ่)

ไฟล์ข้อมูลต้นฉบับไม่ถูกเก็บบน GitHub เนื่องจากขนาดใหญ่
และถูกจัดเก็บไว้ใน Google Drive เท่านั้น
มีทั้งหมด **3 ไฟล์**

📥 ลิงก์ดาวน์โหลด
**[https://drive.google.com/drive/folders/1_LP9o4K7Z6XR5xku7bEJc15pYAWp4hpP?usp=sharing](https://drive.google.com/drive/folders/1_LP9o4K7Z6XR5xku7bEJc15pYAWp4hpP?usp=sharing)**

### ไฟล์ที่รวมอยู่

| ชื่อไฟล์                   | รายละเอียด                                |
| -------------------------- | ----------------------------------------- |
| `minutes.parquet`          | บันทึกการประชุมของโซล + กวางจู (ฉบับเต็ม) |
| `segments_all.parquet`     | ข้อมูลเซกเมนต์ตามผู้พูดทั้งหมด            |
| `base_minutes_rag.parquet` | ข้อมูลพื้นฐานที่ใช้ใน Naive RAG           |

### การวางไฟล์ในเครื่อง (Local Placement)

```
root/data/
    minutes.parquet
    segments_all.parquet
    base_minutes_rag.parquet
```

---

# ⚙️ 2. คำอธิบายโค้ด (src/)

### 🔹 `search_naive.py`

ค้นคืนข้อมูลด้วยเวกเตอร์ฝังตัวจากทุกเซกเมนต์ (FAISS + cosine similarity)

### 🔹 `search_ontology.py`

กรองเซกเมนต์โดยใช้ออนโทโลจี (สภา ผู้พูด พรรค คณะกรรมาธิการ) ก่อนค้นหา
→ ลดการจับคู่ผิดพลาด เพิ่มความแม่นยำด้านข้อเท็จจริง

### 🔹 `generate_naive.py`

สร้างบทความข่าวจากผลการค้นหา Top-5 ของ Naive RAG
ใช้โมเดล LLM: **gpt-4.1-mini**

### 🔹 `generate_ontology.py`

สร้างบทความข่าวโดยใช้ผล Top-5 จาก Ontology RAG
ใช้โมเดล LLM: **gpt-4.1-mini**

### 🔹 `evaluate_absolute.py`

ประเมินผลแบบ absolute scoring สำหรับ factuality และ topical relevance

ความผิดพลาดด้านข้อเท็จจริงประเภทต่อไปนี้ถือว่า *ร้ายแรง*:

* WRONG_COUNCIL
* WRONG_PERSON
* WRONG_PARTY
  → fact_ok = 0

คะแนน topical relevance อยู่ในช่วง 1–10

### 🔹 `index_ontology.py`

สร้างและแคช index ของออนโทโลจี

### 🔹 `pkl_ontology.py`

จัดเก็บออนโทโลจีในรูปแบบ serialized

### 🔹 `paths.py`

จัดการเส้นทางไฟล์ทั้งหมดแบบรวมศูนย์

---

# 📑 3. การตั้งค่าการทดลอง (Experimental Setup)

### ข้อมูลที่ใช้

* สภาที่ใช้: โซล, กวางจู
* ช่วงเวลา: กรกฎาคม 2022 – ตุลาคม 2025
* คำถามประเมิน: 100 ข้อ

### โมเดลที่เปรียบเทียบ

| โมเดล        | รายละเอียด                                   |
| ------------ | -------------------------------------------- |
| Naive RAG    | ค้นคืนจากทุกเซกเมนต์โดยตรง                   |
| Ontology RAG | กรองด้วยออนโทโลจี + ค้นคืนด้วยเวกเตอร์ฝังตัว |

### โมเดลที่ใช้

| การใช้งาน        | โมเดล                  |
| ---------------- | ---------------------- |
| สร้างบทความข่าว  | gpt-4.1-mini           |
| ผู้ประเมินผล LLM | gpt-4.1-mini           |
| เวกเตอร์ฝังตัว   | text-embedding-3-large |

---

# 📈 4. วิธีประเมินผล (Evaluation Method)

### ✔ 1) การประเมินข้อเท็จจริง (Factuality)

ข้อผิดพลาดด้านข้อเท็จจริงที่ถือว่าร้ายแรง:

* WRONG_COUNCIL
* WRONG_PERSON
* WRONG_PARTY

Ontology RAG → ตัดคำถาม EMPTY_SEGMENT ออกจากการคิดสัดส่วน
Naive RAG → ให้คะแนน 0 สำหรับเซกเมนต์ที่คืนมาทั้งหมดในคำถามประเภทนี้

---

### ✔ 2) การประเมินความสอดคล้องเชิงประเด็น (Topical Relevance)

LLM จะให้คะแนนความสอดคล้องระหว่างเซกเมนต์กับหัวข้อคำถามในช่วง 1–10

---

# 📊 5. สรุปผลการทดลอง

## 🔥 1) อัตราความผิดพลาดด้านข้อเท็จจริง

| โมเดล        | จำนวนเซกเมนต์ | จำนวนผิดพลาด | อัตราผิดพลาด |
| ------------ | ------------- | ------------ | ------------ |
| Naive RAG    | 500           | 161          | **32.20%**   |
| Ontology RAG | 610           | 43           | **7.05%**    |

➡ ลดลง **25.15 จุดเปอร์เซ็นต์** (ลดลงประมาณ 78%)

---

## 🎯 2) คะแนนความสอดคล้องเชิงประเด็น

| โมเดล        | คะแนนเฉลี่ย (เต็ม 10) |
| ------------ | --------------------- |
| Naive RAG    | 5.77                  |
| Ontology RAG | 6.54                  |

➡ เพิ่มขึ้น **7.66%**, และมีคะแนนต่ำสุดน้อยลงมาก

---

## 📌 ข้อค้นพบสำคัญ

Ontology RAG เมื่อเทียบกับ Naive RAG:

* ลดข้อผิดพลาดด้านข้อเท็จจริงอย่างมาก
* เพิ่มความแม่นยำด้านหัวข้อ
* แก้ปัญหา catastrophic failure ที่ได้คะแนน 0
* ทำให้ผลการค้นคืนมีเสถียรภาพและสม่ำเสมอมากขึ้น

---

# 🚀 6. วิธีใช้งาน (How to Run)

### 1) โคลนรีโพสิตอรี

```bash
git clone https://github.com/beopryang/nlpir_ks031_A.git
cd nlpir_ks031_A/root
```

### 2) ติดตั้ง dependencies

```bash
pip install -r requirements.txt
```

### 3) ดาวน์โหลดข้อมูลจาก Google Drive

วางไฟล์ไว้ที่:

```
root/data/
    minutes.parquet
    segments_all.parquet
    base_minutes_rag.parquet
```

### 4) รันระบบค้นคืนข้อมูล

```bash
python src/search_naive.py
python src/search_ontology.py
```

### 5) สร้างบทความข่าว

```bash
python src/generate_naive.py
python src/generate_ontology.py
```

### 6) ประเมินผล

```bash
python src/evaluate_absolute.py
```

---

# 🧪 7. การทำซ้ำผลลัพธ์ (Reproducibility)

* ไฟล์ผลลัพธ์ทั้งหมดอยู่ใน `results/`
* ไฟล์ prompt/config ทั้งหมดอยู่ใน `config/`
* โค้ดค้นคืน สร้างบทความ และประเมินผลทั้งหมดอยู่ใน `src/`
* ไฟล์ข้อมูลขนาดใหญ่ถูกแชร์ผ่าน Google Drive
* แม้ว่าบทความที่ LLM สร้างอาจแตกต่างเล็กน้อย
  แต่ค่าการประเมินด้าน factuality และ topical relevance สามารถทำซ้ำได้

---

# 中文（繁體字）版本

<a name="chinese-traditional-version"></a>

# 🏛 基於 RAG 的韓國議會會議紀錄新聞生成系統開發

**Development of a News Article Generation System for Deliberation Records from Korean Legislatures Using RAG**

---

## 📌 概要（Overview）

此存放庫包含一套 **混合式 RAG（Retrieval-Augmented Generation）系統** 的研究程式碼，用於處理光州廣域市議會與首爾特別市議會的會議紀錄資料。
本系統結合：

* **本體（Ontology）為基礎的中繼資料過濾**
* **基於嵌入向量的語意相似度檢索**

本研究比較兩種檢索策略：

1. **Naive RAG**：無任何前處理，直接對所有段落進行嵌入檢索
2. **Ontology RAG**：依據議會、發言人、政黨、委員會等中繼資料過濾後，再進行向量檢索

比較的目的在於檢驗結構化過濾是否能提升：

* 事實正確性（factuality）
* 主題相關性（topical relevance）
* 檢索穩定性（retrieval stability）
* 錯誤魯棒性（error robustness）

資料期間涵蓋 **2022 年 7 月至 2025 年 10 月**，
評估使用 **100 條基準查詢**。

---

# 📁 存放庫結構

```
root/
├─ src/                     # RAG 檢索、本體處理、評估相關程式碼
├─ config/                  # LLM 規劃與評估設定
├─ results/                 # 檢索結果、生成文章、評估資料（已包含）
└─ data/                    # 大型原始資料（透過 Google Drive 提供）
```

---

# 📊 1. 資料說明

## 1.1 results/（已包含於 GitHub）

此資料夾包含所有實驗產生的最終成果。

### 🔍 檢索結果

* `naive_rag_results_top5.csv`
* `ontology_rag_results_tasks.csv`

### 📰 生成的新聞文章

* `naive_rag_articles.csv`
* `naive_rag_articles.jsonl`
* `ontology_rag_articles.csv`
* `ontology_rag_articles.jsonl`

### ✔ 事實正確性評估

* `eval_top5_truth_naive_top5_absolute.csv`
* `eval_top5_truth_onto_top5_absolute.csv`

### 🧪 評估規劃（LLM 評估用任務分解）

* `eval_plans_onto_top5.json`

### 🧾 100 條基準查詢

* `test_queries.csv`

所有資料皆已收錄以確保可重現性。

---

## 1.2 data/（Google Drive — 大型檔案）

由於檔案過大，未上傳至 GitHub，而是以 Google Drive 提供。
Drive 中僅包含 **三個檔案**。

📥 下載連結：

**[https://drive.google.com/drive/folders/1_LP9o4K7Z6XR5xku7bEJc15pYAWp4hpP?usp=sharing](https://drive.google.com/drive/folders/1_LP9o4K7Z6XR5xku7bEJc15pYAWp4hpP?usp=sharing)**

### 檔案內容

| 檔名                         | 說明                  |
| -------------------------- | ------------------- |
| `minutes.parquet`          | 光州 + 首爾完整會議紀錄原文     |
| `segments_all.parquet`     | 逐發言者切分後的所有段落        |
| `base_minutes_rag.parquet` | Naive RAG 用的簡易前處理資料 |

### 放置位置

將下載的檔案置於：

```
root/data/
    minutes.parquet
    segments_all.parquet
    base_minutes_rag.parquet
```

---

# ⚙️ 2. 程式碼說明（src/）

### 🔹 `search_naive.py`

對所有段落執行嵌入向量檢索（FAISS + 餘弦相似度）。

### 🔹 `search_ontology.py`

以本體中繼資料（議會、人物、政黨、委員會）進行過濾後再檢索。
→ 避免錯誤配對，大幅提升事實正確性。

### 🔹 `generate_naive.py`

以 Naive RAG Top-5 段落生成新聞文章。
LLM：**gpt-4.1-mini**

### 🔹 `generate_ontology.py`

以 Ontology RAG Top-5 段落生成文章。
LLM：**gpt-4.1-mini**

### 🔹 `evaluate_absolute.py`

以「絕對評分法」評估事實正確性與主題相關性。

以下視為嚴重事實錯誤：

* WRONG_COUNCIL
* WRONG_PERSON
* WRONG_PARTY

→ fact_ok = 0
主題相關性以 1～10 分量化。

### 🔹 `index_ontology.py`

建立與快取本體索引。

### 🔹 `pkl_ontology.py`

將本體結構序列化。

### 🔹 `paths.py`

管理所有檔案路徑設定。

---

# 📑 3. 實驗設計

### 資料集

* 議會：光州市議會、首爾市議會
* 期間：2022.07–2025.10
* 查詢：100 條基準測試查詢

### 比較模型

| 模型           | 說明           |
| ------------ | ------------ |
| Naive RAG    | 對所有段落進行嵌入檢索  |
| Ontology RAG | 本體過濾後再進行嵌入檢索 |

### 使用模型

| 用途             | 模型                     |
| -------------- | ---------------------- |
| 新聞生成           | gpt-4.1-mini           |
| 評估者（LLM judge） | gpt-4.1-mini           |
| 嵌入向量           | text-embedding-3-large |

---

# 📈 4. 評估方法

### ✔ 1）事實正確性評估（fact-level）

以下皆屬「重大事實錯誤」：

* WRONG_COUNCIL
* WRONG_PERSON
* WRONG_PARTY

Ontology RAG：排除 EMPTY_SEGMENT 查詢
Naive RAG：對同查詢下所有段落給予 **0 分懲罰**

---

### ✔ 2）主題相關性（topic_score）

LLM 分析查詢意圖後，評估段落主題是否符合查詢。
以 1～10 分量化。

---

# 📊 5. 結果摘要

## 🔥 1）事實錯誤率

| 模型           | 段落數 | 錯誤數 | 錯誤率        |
| ------------ | --- | --- | ---------- |
| Naive RAG    | 500 | 161 | **32.20%** |
| Ontology RAG | 610 | 43  | **7.05%**  |

**→ 降低 25.15 個百分點（相對減少約 78%）**

---

## 🎯 2）主題相關性平均分

| 模型           | 平均分（滿分 10） |
| ------------ | ---------- |
| Naive RAG    | 5.77       |
| Ontology RAG | 6.54       |

**→ 提升 7.66%**，且極低分情況大幅減少。

---

## 📌 核心結論

與 Naive RAG 相比，Ontology RAG 帶來：

* 顯著減少事實錯誤
* 提升主題符合度
* 消除嚴重的 0 分檢索失敗
* 檢索結果更穩定、一致

結構性的中繼資料過濾對改善檢索品質有直接效果。

---

# 🚀 6. 執行方式

### 1）複製存放庫

```bash
git clone https://github.com/beopryang/nlpir_ks031_A.git
cd nlpir_ks031_A/root
```

### 2）安裝套件

```bash
pip install -r requirements.txt
```

### 3）下載 Google Drive 資料

下載後置於：

```
root/data/
    minutes.parquet
    segments_all.parquet
    base_minutes_rag.parquet
```

### 4）執行檢索

```bash
python src/search_naive.py
python src/search_ontology.py
```

### 5）生成文章

```bash
python src/generate_naive.py
python src/generate_ontology.py
```

### 6）評估

```bash
python src/evaluate_absolute.py
```

---

# 🧪 7. 可重現性

* `results/` 中所有結果已完整收錄
* `config/` 中包含所有 LLM 提示與設定
* `src/` 中包含檢索・生成・評估的完整程式
* 大型資料透過 Google Drive 公開提供
* 雖然生成文本可能略有變動，
  **事實性與主題相關性指標可穩定重現**

---

# 🇻🇳 Phiên bản Tiếng Việt

<a name="vietnamese-version"></a>

# 🏛 Phát triển hệ thống tạo bài báo từ biên bản thảo luận của các cơ quan lập pháp Hàn Quốc bằng RAG

---

## 📌 Tổng quan

Kho lưu trữ này chứa mã nguồn nghiên cứu cho **hệ thống RAG lai (Retrieval-Augmented Generation)** dùng để xử lý biên bản thảo luận từ Hội đồng Thành phố Gwangju và Hội đồng Thủ đô Seoul.
Hệ thống kết hợp **lọc siêu dữ liệu dựa trên ontology** với **tìm kiếm tương đồng dựa trên embedding**.

Nghiên cứu so sánh hai chiến lược truy xuất:

1. **Naive RAG** – truy xuất dựa hoàn toàn trên embedding, không tiền xử lý
2. **Ontology RAG** – giới hạn truy xuất bằng metadata (cơ quan, người phát biểu, đảng phái, ủy ban) rồi mới thực hiện tìm kiếm embedding

Thông qua so sánh này, nghiên cứu đánh giá mức độ cải thiện của phương pháp lọc cấu trúc đối với:

* tính chính xác sự kiện (factuality)
* mức độ phù hợp chủ đề (topical relevance)
* tính ổn định của truy xuất
* khả năng giảm lỗi (error robustness)

Bộ dữ liệu bao gồm **tháng 7/2022 – tháng 10/2025**, và đánh giá được tiến hành bằng **100 truy vấn chuẩn**.

---

# 📁 Cấu trúc kho lưu trữ

```
root/
├─ src/                     # Mã truy xuất RAG, ontology và đánh giá
├─ config/                  # Thiết lập LLM cho lập kế hoạch & đánh giá
├─ results/                 # Kết quả truy xuất / sinh / đánh giá (được lưu kèm)
└─ data/                    # Dữ liệu gốc dung lượng lớn (qua Google Drive)
```

---

# 📊 1. Mô tả dữ liệu

## 1.1 results/ (đã bao gồm trong GitHub)

Thư mục này chứa toàn bộ kết quả do hệ thống sinh ra.

### 🔍 Kết quả truy xuất

* `naive_rag_results_top5.csv`
* `ontology_rag_results_tasks.csv`

### 📰 Bài báo được tạo

* `naive_rag_articles.csv`
* `naive_rag_articles.jsonl`
* `ontology_rag_articles.csv`
* `ontology_rag_articles.jsonl`

### ✔ Đánh giá tính chính xác sự kiện

* `eval_top5_truth_naive_top5_absolute.csv`
* `eval_top5_truth_onto_top5_absolute.csv`

### 🧪 Kế hoạch đánh giá

* `eval_plans_onto_top5.json`

### 🧾 Tập truy vấn chuẩn

* `test_queries.csv` (100 truy vấn)

Tất cả được lưu trong repo để đảm bảo khả năng tái lập.

---

## 1.2 data/ (Google Drive — tập tin lớn)

Dữ liệu gốc dung lượng lớn không lưu trực tiếp trên GitHub mà được cung cấp qua Google Drive.
Chỉ bao gồm **ba tập tin**:

📥 Liên kết tải xuống
**[https://drive.google.com/drive/folders/1_LP9o4K7Z6XR5xku7bEJc15pYAWp4hpP?usp=sharing](https://drive.google.com/drive/folders/1_LP9o4K7Z6XR5xku7bEJc15pYAWp4hpP?usp=sharing)**

### Tập tin bao gồm

| Tập tin                    | Mô tả                                   |
| -------------------------- | --------------------------------------- |
| `minutes.parquet`          | Biên bản đầy đủ (Gwangju + Seoul)       |
| `segments_all.parquet`     | Tất cả đoạn theo đơn vị người phát biểu |
| `base_minutes_rag.parquet` | Dữ liệu tiền xử lý cho Naive RAG        |

### Đặt vào thư mục local

```
root/data/
    minutes.parquet
    segments_all.parquet
    base_minutes_rag.parquet
```

---

# ⚙️ 2. Mô tả mã nguồn (src/)

### 🔹 `search_naive.py`

Truy xuất dựa trên embedding cho toàn bộ các đoạn (FAISS + cosine similarity).

### 🔹 `search_ontology.py`

Lọc đoạn bằng ontology metadata (cơ quan, người phát biểu, đảng, ủy ban) rồi mới truy xuất.
→ Giảm nhầm lẫn và nâng cao tính chính xác sự kiện.

### 🔹 `generate_naive.py`

Sinh bài báo từ kết quả truy xuất Top-5 của Naive RAG.
LLM: **gpt-4.1-mini**

### 🔹 `generate_ontology.py`

Sinh bài báo từ kết quả Top-5 của Ontology RAG.
LLM: **gpt-4.1-mini**

### 🔹 `evaluate_absolute.py`

Đánh giá tính chính xác và mức độ phù hợp chủ đề.

* WRONG_COUNCIL
* WRONG_PERSON
* WRONG_PARTY
  → Được xem là lỗi sự kiện nghiêm trọng (fact_ok = 0)

### 🔹 `index_ontology.py`

Tạo và lưu cache của chỉ mục ontology.

### 🔹 `pkl_ontology.py`

Tuần tự hóa dữ liệu ontology.

### 🔹 `paths.py`

Quản lý tập trung tất cả đường dẫn tập tin.

---

# 📑 3. Thiết lập thí nghiệm

### Bộ dữ liệu

* Cơ quan: Hội đồng Thành phố Gwangju, Hội đồng Thủ đô Seoul
* Thời gian: 07/2022 – 10/2025
* 100 truy vấn chuẩn

### Mô hình so sánh

| Mô hình      | Mô tả                         |
| ------------ | ----------------------------- |
| Naive RAG    | Truy xuất embedding không lọc |
| Ontology RAG | Lọc metadata + embedding      |

### Mô hình sử dụng

| Mục đích     | Mô hình                |
| ------------ | ---------------------- |
| Sinh bài báo | gpt-4.1-mini           |
| Đánh giá LLM | gpt-4.1-mini           |
| Embedding    | text-embedding-3-large |

---

# 📈 4. Phương pháp đánh giá

### ✔ 1) Đánh giá tính chính xác sự kiện

Các lỗi sau được xem là **lỗi nghiêm trọng**:

* WRONG_COUNCIL
* WRONG_PERSON
* WRONG_PARTY

Ontology RAG: loại bỏ các truy vấn EMPTY_SEGMENT
Naive RAG: áp dụng **điểm 0** cho tất cả đoạn được truy xuất trong truy vấn này

---

### ✔ 2) Đánh giá mức độ phù hợp chủ đề

LLM chấm điểm mức độ phù hợp giữa đoạn truy xuất và mục tiêu truy vấn.
Thang điểm **1–10**.

---

# 📊 5. Tóm tắt kết quả

## 🔥 1) Tỷ lệ lỗi sự kiện

| Mô hình      | Số đoạn | Số lỗi | Tỷ lệ lỗi  |
| ------------ | ------- | ------ | ---------- |
| Naive RAG    | 500     | 161    | **32.20%** |
| Ontology RAG | 610     | 43     | **7.05%**  |

**→ Giảm 25.15 điểm phần trăm (≈ 78% giảm tương đối)**

---

## 🎯 2) Mức độ phù hợp trung bình

| Mô hình      | Điểm trung bình (tối đa 10) |
| ------------ | --------------------------- |
| Naive RAG    | 5.77                        |
| Ontology RAG | 6.54                        |

**→ Tăng 7.66%**, giảm mạnh các trường hợp điểm thấp thảm họa.

---

## 📌 Kết luận chính

Ontology RAG mang lại:

* giảm mạnh lỗi sự kiện
* tăng độ phù hợp chủ đề
* loại bỏ lỗi truy xuất điểm 0
* truy xuất ổn định và nhất quán hơn

Lợi ích này đến từ việc áp đặt ràng buộc cấu trúc trước khi truy xuất ngữ nghĩa.

---

# 🚀 6. Cách chạy

### 1) Clone repo

```bash
git clone https://github.com/beopryang/nlpir_ks031_A.git
cd nlpir_ks031_A/root
```

### 2) Cài đặt gói

```bash
pip install -r requirements.txt
```

### 3) Tải dữ liệu từ Google Drive

Đặt vào:

```
root/data/
    minutes.parquet
    segments_all.parquet
    base_minutes_rag.parquet
```

### 4) Truy xuất

```bash
python src/search_naive.py
python src/search_ontology.py
```

### 5) Sinh bài báo

```bash
python src/generate_naive.py
python src/generate_ontology.py
```

### 6) Đánh giá

```bash
python src/evaluate_absolute.py
```

---

# 🧪 7. Khả năng tái lập

* Toàn bộ kết quả trong `results/` được lưu đầy đủ
* File cấu hình LLM trong `config/`
* Mã truy xuất / sinh / đánh giá trong `src/`
* Dữ liệu lớn trong `data/` chia sẻ qua Google Drive
* Kết quả văn bản có thể hơi khác do bản chất sinh ngôn ngữ,
  nhưng các chỉ số factuality & relevance có thể tái lập ổn định

---

# 🇰🇿 Қазақша нұсқа (Kazakh Version)

<a name="kazakh-version"></a>

# 🏛 RAG технологиясын пайдалана отырып, Корея парламенттерінің тыңдалым материалдарына негізделген жаңалық мақалаларын генерациялау жүйесін әзірлеу

---

## 📌 Шолу (Overview)

Бұл репозиторий Гванчжу қалалық кеңесі мен Сеул қалалық кеңесінің тыңдалым (deliberation) материалдарын өңдейтін **гибридті RAG (Retrieval-Augmented Generation)** жүйесінің зерттеу кодтарын қамтиды.
Жүйе **онтологияға негізделген метадеректерді сүзгілеуді** және **эмбеддинг ұқсастығына негізделген іздеуді** біріктіреді.

Бұл зерттеуде екі іздеу стратегиясы салыстырылады:

1. **Naive RAG** – алдын ала өңдеусіз таза эмбеддинг негізіндегі іздеу
2. **Ontology RAG** – іздеуге дейін кеңес/спикер/партия/комитет метадеректерімен шектелетін іздеу

Бұл салыстыру құрылымдық сүзгілердің келесі көрсеткіштерді қалай жақсартатынын бағалайды:

* факті дәлдігі
* тақырыптық сәйкестік
* іздеу нәтижелерінің тұрақтылығы
* қателерге төзімділік

Деректер жинағы **2022 жылғы шілде – 2025 жылғы қазан** аралығын қамтиды.
Бағалау **100 тест сұрағы** арқылы жүргізілді.

---

# 📁 Репозиторий құрылымы

```
root/
├─ src/                     # RAG іздеу, онтология, бағалау кодтары
├─ config/                  # LLM жоспарлау және бағалау конфигурациясы
├─ results/                 # Іздеу/шығыс/бағалау нәтижелері (бар)
└─ data/                    # Үлкен бастапқы деректер (Google Drive арқылы)
```

---

# 📊 1. Деректер сипаттамасы

## 1.1 results/ (GitHub ішінде)

Бұл қалтада барлық эксперименттерден алынған нәтижелер бар.

### 🔍 Іздеу нәтижелері

* `naive_rag_results_top5.csv`
* `ontology_rag_results_tasks.csv`

### 📰 Генерацияланған мақалалар

* `naive_rag_articles.csv`
* `naive_rag_articles.jsonl`
* `ontology_rag_articles.csv`
* `ontology_rag_articles.jsonl`

### ✔ Факті дәлдігі бағалауы

* `eval_top5_truth_naive_top5_absolute.csv`
* `eval_top5_truth_onto_top5_absolute.csv`

### 🧪 Бағалау жоспарлары

* `eval_plans_onto_top5.json`

### 🧾 Тест сұрақтары

* `test_queries.csv` (100 сұрақ)

Толық қайта өндіруге қажет барлық файлдар қосылған.

---

## 1.2 data/ (Google Drive — үлкен файлдар)

Үлкен бастапқы деректер GitHub-та сақталмайды.
Google Drive арқылы бөлек беріледі.

📥 Жүктеу сілтемесі:
**[https://drive.google.com/drive/folders/1_LP9o4K7Z6XR5xku7bEJc15pYAWp4hpP?usp=sharing](https://drive.google.com/drive/folders/1_LP9o4K7Z6XR5xku7bEJc15pYAWp4hpP?usp=sharing)**

### Қосылған файлдар

| Файл аты                   | Сипаттамасы                                  |
| -------------------------- | -------------------------------------------- |
| `minutes.parquet`          | Толық тыңдалым материалдары (Гванчжу + Сеул) |
| `segments_all.parquet`     | Барлық спикерлік сегменттер                  |
| `base_minutes_rag.parquet` | Naive RAG үшін алдын ала өңделген деректер   |

### Жергілікті орналастыру

```
root/data/
    minutes.parquet
    segments_all.parquet
    base_minutes_rag.parquet
```

---

# ⚙️ 2. Код сипаттамасы (src/)

### 🔹 `search_naive.py`

Барлық сегменттер бойынша эмбеддинг негізіндегі іздеу (FAISS + косинустық ұқсастық).

### 🔹 `search_ontology.py`

Онтология метадеректері (кеңес, спикер, партия, комитет) бойынша сүзгілеу → эмбеддинг іздеу.
Бұл сәйкессіздіктерді азайтады және факті дәлдігін арттырады.

### 🔹 `generate_naive.py`

Naive RAG Top-5 нәтижелеріне негізделген жаңалық стиліндегі мақалалар генерациялайды.
LLM: **gpt-4.1-mini**

### 🔹 `generate_ontology.py`

Ontology RAG Top-5 нәтижелеріне негізделген мақалалар генерациялайды.
LLM: **gpt-4.1-mini**

### 🔹 `evaluate_absolute.py`

Факті дәлдігі мен тақырыптық сәйкестікті абсолютті бағалау.

Келесі қателер **ауыр факт қателігі** ретінде қарастырылады:

* WRONG_COUNCIL
* WRONG_PERSON
* WRONG_PARTY

### 🔹 `index_ontology.py`

Онтология индекстерін құру және кэштеу.

### 🔹 `pkl_ontology.py`

Онтология құрылымдарын сериализациялау.

### 🔹 `paths.py`

Жоба каталогтарының орталық конфигурациясы.

---

# 📑 3. Эксперименттік орнату

### Деректер жиыны

* Кеңестер: Гванчжу, Сеул
* Кезең: 2022.07 – 2025.10
* Бағалау: 100 тест сұрағы

### Модельдер салыстыруы

| Модель       | Сипаттамасы                           |
| ------------ | ------------------------------------- |
| Naive RAG    | Эмбеддингке негізделген жалпы іздеу   |
| Ontology RAG | Метадерек сүзгілеуі + эмбеддинг іздеу |

### Қолданылған модельдер

| Мақсаты            | Модель                 |
| ------------------ | ---------------------- |
| Мақала генерациясы | gpt-4.1-mini           |
| Бағалау            | gpt-4.1-mini           |
| Эмбеддинг          | text-embedding-3-large |

---

# 📈 4. Бағалау әдісі

### ✔ 1) Факті дәлдігі

Келесі түрлер **ауыр факт қателері** деп саналады:

* WRONG_COUNCIL
* WRONG_PERSON
* WRONG_PARTY

### ✔ 2) Тақырыптық сәйкестік

LLM сегменттердің сұрақ тақырыбына қаншалықты сәйкес келетінін 1–10 баллмен бағалайды.

---

# 📊 5. Нәтижелер

## 🔥 1) Факті қатесінің жиілігі

| Модель       | #Сегмент | #Қате | Қате жиілігі |
| ------------ | -------- | ----- | ------------ |
| Naive RAG    | 500      | 161   | **32.20%**   |
| Ontology RAG | 610      | 43    | **7.05%**    |

**→ 25.15 пайыздық тармаққа төмендеу (≈ 78% салыстырмалы төмендеу)**

---

## 🎯 2) Тақырыптық орташа балл

| Модель       | Орташа балл (10 макс) |
| ------------ | --------------------- |
| Naive RAG    | 5.77                  |
| Ontology RAG | 6.54                  |

**→ +7.66% жақсару**, әрі нөлдік сорапты нәтижелер айтарлықтай азайды.

---

## 📌 Негізгі қорытындылар

Ontology RAG, Naive RAG-пен салыстырғанда:

* факт қателерін айтарлықтай азайтады
* тақырыптық сәйкестікті жақсартады
* нөлдік-құлау (catastrophic failure) жағдайларын жояды
* нәтижелердің тұрақтылығын арттырады

---

# 🚀 6. Іске қосу нұсқаулығы

### 1) Репозиторийді клондау

```bash
git clone https://github.com/beopryang/nlpir_ks031_A.git
cd nlpir_ks031_A/root
```

### 2) Тәуелділіктерді орнату

```bash
pip install -r requirements.txt
```

### 3) Деректерді жүктеу

Файлдарды келесіге орналастырыңыз:

```
root/data/
    minutes.parquet
    segments_all.parquet
    base_minutes_rag.parquet
```

### 4) Іздеуді орындау

```bash
python src/search_naive.py
python src/search_ontology.py
```

### 5) Мақала генерациясы

```bash
python src/generate_naive.py
python src/generate_ontology.py
```

### 6) Бағалау

```bash
python src/evaluate_absolute.py
```

---

# 🧪 7. Қайта өндірілуі

* `results/` ішіндегі барлық нәтижелер қосылған
* Барлық конфигурациялар `config/` ішінде
* Іздеу/генерация/бағалау кодтары `src/` ішінде
* Үлкен деректер Google Drive арқылы қолжетімді
* LLM мәтінінің шағын вариациясы болуы мүмкін, бірақ метрикалар тұрақты қайта өндіріледі

---

# 🇳🇱 **Nederlandse versie**

<a name="dutch-version"></a>

# 🏛 Ontwikkeling van een systeem voor het genereren van nieuwsartikelen op basis van beraadslagningsverslagen van Koreaanse wetgevende organen met behulp van RAG

---

## 📌 Overzicht

Deze repository bevat de onderzoeks­code voor een **hybride RAG-systeem (Retrieval-Augmented Generation)** dat beraadslagningsverslagen van de gemeenteraad van Gwangju en de gemeenteraad van Seoel verwerkt.
Het systeem integreert **ontologie-gebaseerde metadata-filtering** met **embedding-gebaseerde similariteitszoeking**.

Het onderzoek vergelijkt twee retrievalstrategieën:

1. **Naive RAG** – pure embedding-gebaseerde retrieval zonder voorfiltering
2. **Ontology RAG** – retrieval beperkt door metadata (raad, spreker, partij, commissie) vóór embedding-zoeking

Deze vergelijking onderzoekt hoe structurele filtering leidt tot verbeteringen in:

* feitelijke juistheid
* thematische relevantie
* stabiliteit van retrieval
* robuustheid tegen fouten

De dataset bestrijkt **juli 2022 – oktober 2025**, en de evaluatie is uitgevoerd op basis van **100 benchmark­vragen**.

---

# 📁 Repository-structuur

```
root/
├─ src/                     # RAG-retrieval, ontologie en evaluatiecode
├─ config/                  # LLM-planner & evaluatieconfiguratie
├─ results/                 # Retrieval-, output- en evaluatieresultaten
└─ data/                    # Grote oorspronkelijke data (Google Drive)
```

---

# 📊 1. Beschrijving van gegevens

## 1.1 results/ (opgenomen in GitHub)

Deze map bevat alle gegenereerde uitvoerbestanden van de experimenten.

### 🔍 Retrievalresultaten

* `naive_rag_results_top5.csv`
* `ontology_rag_results_tasks.csv`

### 📰 Gegenereerde artikelen

* `naive_rag_articles.csv`
* `naive_rag_articles.jsonl`
* `ontology_rag_articles.csv`
* `ontology_rag_articles.jsonl`

### ✔ Feitelijke juistheid evaluatie

* `eval_top5_truth_naive_top5_absolute.csv`
* `eval_top5_truth_onto_top5_absolute.csv`

### 🧪 Evaluatieplannen

* `eval_plans_onto_top5.json`

### 🧾 Benchmarkvragen

* `test_queries.csv` (100 vragen)

Alle bestanden zijn opgenomen om volledige reproduceerbaarheid te garanderen.

---

## 1.2 data/ (Google Drive — grote bestanden)

Grote oorspronkelijke databestanden worden niet op GitHub opgeslagen en zijn beschikbaar via Google Drive.
Er zijn slechts **drie bestanden** inbegrepen.

📥 Downloadlink
**[https://drive.google.com/drive/folders/1_LP9o4K7Z6XR5xku7bEJc15pYAWp4hpP?usp=sharing](https://drive.google.com/drive/folders/1_LP9o4K7Z6XR5xku7bEJc15pYAWp4hpP?usp=sharing)**

### Inbegrepen bestanden

| Bestandsnaam               | Beschrijving                                         |
| -------------------------- | ---------------------------------------------------- |
| `minutes.parquet`          | Volledige beraadslagningsverslagen (Gwangju + Seoel) |
| `segments_all.parquet`     | Alle segmenten op spreker­niveau                     |
| `base_minutes_rag.parquet` | Voorbewerkte basisgegevens voor Naive RAG            |

### Lokale opslag

Plaats de gedownloade bestanden in:

```
root/data/
    minutes.parquet
    segments_all.parquet
    base_minutes_rag.parquet
```

---

# ⚙️ 2. Codebeschrijving (src/)

### 🔹 `search_naive.py`

Voert embedding-gebaseerde retrieval uit op alle segmenten (FAISS + cosinus­similariteit).

### 🔹 `search_ontology.py`

Filtert segmenten op basis van ontologiemetadata (raad, spreker, partij, commissie)
en voert vervolgens embedding-gebaseerde retrieval uit.
→ Voorkomt mismatches en verhoogt de feitelijke juistheid.

### 🔹 `generate_naive.py`

Genereert nieuwsartikelen op basis van de Top-5 Naive RAG-resultaten.
LLM: **gpt-4.1-mini**

### 🔹 `generate_ontology.py`

Genereert artikelen op basis van de Top-5 Ontology RAG-resultaten.
LLM: **gpt-4.1-mini**

### 🔹 `evaluate_absolute.py`

Voert absolute evaluatie uit van feitelijke juistheid en thematische relevantie.

Sterke feitelijke fouten:

* WRONG_COUNCIL
* WRONG_PERSON
* WRONG_PARTY

→ worden beoordeeld als **fact_ok = 0**

Thematische relevantie wordt beoordeeld op een schaal van 1–10.

### 🔹 `index_ontology.py`

Bouwt en cachet ontologie-indexen.

### 🔹 `pkl_ontology.py`

Serialiseert ontologiedata­structuren.

### 🔹 `paths.py`

Bevat centrale padconfiguratie.

---

# 📑 3. Experimentele instellingen

### Dataset

* Raden: gemeenteraad Gwangju, gemeenteraad Seoel
* Periode: juli 2022 – oktober 2025
* Evaluatie: 100 benchmark­vragen

### Vergeleken modellen

| Model        | Beschrijving                                  |
| ------------ | --------------------------------------------- |
| Naive RAG    | Embedding-based retrieval over alle segmenten |
| Ontology RAG | Metadatafiltering + embedding retrieval       |

### Gebruikte modellen

| Doel             | Model                  |
| ---------------- | ---------------------- |
| Artikelgeneratie | gpt-4.1-mini           |
| Evaluatie        | gpt-4.1-mini           |
| Embeddings       | text-embedding-3-large |

---

# 📈 4. Evaluatiemethode

### ✔ 1) Feitelijke juistheid

Als **ernstige fouten** beschouwd:

* WRONG_COUNCIL
* WRONG_PERSON
* WRONG_PARTY

Ontology RAG: verwijdert vragen met EMPTY_SEGMENT
Naive RAG: geeft **0 punten** als segmenten afkomstig zijn uit EMPTY_SEGMENT-vragen

---

### ✔ 2) Thematische relevantie (topic_score)

De LLM beoordeelt de mate waarin segmenten aansluiten op het gevraagde onderwerp.
Schaal: **1–10**

---

# 📊 5. Resultaten

## 🔥 1) Foutpercentage feitelijke juistheid

| Model        | #Segmenten | #Fouten | Foutpercentage |
| ------------ | ---------- | ------- | -------------- |
| Naive RAG    | 500        | 161     | **32,20%**     |
| Ontology RAG | 610        | 43      | **7,05%**      |

**→ 25,15 procentpunt verbetering (≈ 78% relatieve reductie)**

---

## 🎯 2) Gemiddelde thematische relevantie

| Model        | Gemiddelde score (max 10) |
| ------------ | ------------------------- |
| Naive RAG    | 5,77                      |
| Ontology RAG | 6,54                      |

**→ +7,66% verbetering**, met aanzienlijk minder catastrofale lage scores.

---

## 📌 Belangrijkste bevindingen

In vergelijking met Naive RAG biedt Ontology RAG:

* aanzienlijke vermindering van feitelijke fouten
* hogere thematische relevantie
* eliminatie van retrievalresultaten met 0 punten
* veel consistenter en stabieler retrievalgedrag

Deze verbeteringen komen voort uit het toepassen van structurele beperkingen vóór semantische retrieval.

---

# 🚀 6. Uitvoering

### 1) Repository klonen

```bash
git clone https://github.com/beopryang/nlpir_ks031_A.git
cd nlpir_ks031_A/root
```

### 2) Installatie van afhankelijkheden

```bash
pip install -r requirements.txt
```

### 3) Data downloaden

```
root/data/
```

### 4) Retrieval uitvoeren

```bash
python src/search_naive.py
python src/search_ontology.py
```

### 5) Artikelen genereren

```bash
python src/generate_naive.py
python src/generate_ontology.py
```

### 6) Evaluatie

```bash
python src/evaluate_absolute.py
```

---

# 🧪 7. Reproduceerbaarheid

* Alle uitvoerbestanden in `results/` zijn inbegrepen
* Alle prompts/configuraties in `config/` zijn aanwezig
* Alle retrieval-, generatie- en evaluatiecode staat in `src/`
* Grote databestanden zijn gedeeld via Google Drive
* Hoewel LLM-uitvoer enigszins kan variëren, blijven de evaluatiemetrics reproduceerbaar

---

# 🇷🇺 Русская версия

<a name="russian-version"></a>

# 🏛 Разработка системы генерации новостных статей на основе протоколов заседаний корейских законодательных органов с использованием RAG

---

## 📌 Обзор

Данный репозиторий содержит исследовательский код для **гибридной системы RAG (Retrieval-Augmented Generation)**, предназначенной для обработки протоколов заседаний Городского совета Кванджу и Столичного совета Сеула.
Система сочетает **фильтрацию метаданных на основе онтологии** с **поиском по сходству встроенных представлений (embeddings)**.

В исследовании сравниваются две стратегии поиска:

1. **Naive RAG** – поиск, основанный только на embeddings, без предварительной фильтрации
2. **Ontology RAG** – поиск, ограниченный метаданными (совет, выступающий, партия, комитет) перед выполнением embedding-поиска

Цель исследования — оценить, как структурная фильтрация улучшает:

* фактическую точность,
* тематическую релевантность,
* стабильность поиска,
* устойчивость к ошибкам.

Датасет охватывает период **с июля 2022 года по октябрь 2025 года**, а оценивание проводится на основе **100 тестовых запросов**.

---

# 📁 Структура репозитория

```
root/
├─ src/                     # Код поиска, онтологии и оценки RAG
├─ config/                  # Настройки планировщика и оценщика LLM
├─ results/                 # Результаты поиска, генерации и оценки (включены)
└─ data/                    # Крупные исходные данные (доступны через Google Drive)
```

---

# 📊 1. Описание данных

## 1.1 results/ (включено в GitHub)

Папка содержит все результаты, полученные в ходе экспериментов.

### 🔍 Результаты поиска

* `naive_rag_results_top5.csv`
* `ontology_rag_results_tasks.csv`

### 📰 Сгенерированные статьи

* `naive_rag_articles.csv`
* `naive_rag_articles.jsonl`
* `ontology_rag_articles.csv`
* `ontology_rag_articles.jsonl`

### ✔ Оценка фактической точности

* `eval_top5_truth_naive_top5_absolute.csv`
* `eval_top5_truth_onto_top5_absolute.csv`

### 🧪 Планы оценки

* `eval_plans_onto_top5.json`

### 🧾 Тестовые запросы

* `test_queries.csv` (100 запросов)

Все файлы включены для обеспечения полной воспроизводимости.

---

## 1.2 data/ (Google Drive — большие файлы)

Из-за большого размера данные не включены в GitHub и предоставляются через Google Drive.
Всего **три файла**:

📥 Ссылка для скачивания:
**[https://drive.google.com/drive/folders/1_LP9o4K7Z6XR5xku7bEJc15pYAWp4hpP?usp=sharing](https://drive.google.com/drive/folders/1_LP9o4K7Z6XR5xku7bEJc15pYAWp4hpP?usp=sharing)**

### Включённые файлы

| Имя файла                  | Описание                                    |
| -------------------------- | ------------------------------------------- |
| `minutes.parquet`          | Полные протоколы заседаний (Кванджу + Сеул) |
| `segments_all.parquet`     | Все сегменты, разделённые по выступающим    |
| `base_minutes_rag.parquet` | Предобработанные данные для Naive RAG       |

### Локальное размещение

Сохраните файлы в:

```
root/data/
    minutes.parquet
    segments_all.parquet
    base_minutes_rag.parquet
```

---

# ⚙️ 2. Описание кода (src/)

### 🔹 `search_naive.py`

Выполняет поиск по embeddings (FAISS + косинусное сходство) по всем сегментам.

### 🔹 `search_ontology.py`

Фильтрует сегменты по онтологическим метаданным (совет, выступающий, партия, комитет),
после чего выполняет embedding-поиск.
→ Устраняет неправильные совпадения и повышает фактическую точность.

### 🔹 `generate_naive.py`

Генерирует новостные статьи на основе Top-5 результатов Naive RAG.
LLM: **gpt-4.1-mini**

### 🔹 `generate_ontology.py`

Генерация статей на основе Top-5 результатов Ontology RAG.
LLM: **gpt-4.1-mini**

### 🔹 `evaluate_absolute.py`

Абсолютная оценка фактической точности и тематической релевантности.

Ошибки:

* WRONG_COUNCIL
* WRONG_PERSON
* WRONG_PARTY

→ считаются *серьёзными фактическими ошибками* (fact_ok = 0)

Тематическая релевантность оценивается по шкале 1–10.

### 🔹 `index_ontology.py`

Создание и кэширование онтологических индексов.

### 🔹 `pkl_ontology.py`

Сериализация структур онтологических метаданных.

### 🔹 `paths.py`

Централизованная конфигурация путей.

---

# 📑 3. Экспериментальная установка

### Датасет

* Советы: Кванджу, Сеул
* Период: 2022.07 – 2025.10
* 100 тестовых запросов

### Сравниваемые модели

| Модель       | Описание                                |
| ------------ | --------------------------------------- |
| Naive RAG    | Поиск embedding по всем сегментам       |
| Ontology RAG | Фильтрация метаданных + embedding-поиск |

### Используемые модели

| Цель             | Модель                 |
| ---------------- | ---------------------- |
| Генерация статей | gpt-4.1-mini           |
| Оценка           | gpt-4.1-mini           |
| Embeddings       | text-embedding-3-large |

---

# 📈 4. Метод оценки

### ✔ 1) Оценка фактической точности

Серьёзные фактические ошибки:

* WRONG_COUNCIL
* WRONG_PERSON
* WRONG_PARTY

Ontology RAG: удаляет запросы EMPTY_SEGMENT
Naive RAG: присваивает **0 баллов**, если любой найденный сегмент относится к пустому запросу.

---

### ✔ 2) Тематическая релевантность (topic_score)

LLM оценивает, насколько сегменты соответствуют теме запроса.
Оценка: **1–10 баллов**

---

# 📊 5. Результаты

## 🔥 1) Фактическая ошибка

| Модель       | Сегментов | Ошибок | Ошибка     |
| ------------ | --------- | ------ | ---------- |
| Naive RAG    | 500       | 161    | **32.20%** |
| Ontology RAG | 610       | 43     | **7.05%**  |

➡ **Снижение на 25.15 процентных пункта (≈ 78% относительное снижение)**

---

## 🎯 2) Средняя тематическая релевантность

| Модель       | Средний балл |
| ------------ | ------------ |
| Naive RAG    | 5.77         |
| Ontology RAG | 6.54         |

➡ **Улучшение на 7.66%**, существенно меньше провальных значений.

---

## 📌 Основные выводы

Ontology RAG обеспечивает:

* значительное снижение фактических ошибок
* лучшую тематическую релевантность
* устранение провальных 0-балльных поисков
* более стабильные и согласованные результаты

Причина — структурные ограничения до выполнения семантического поиска.

---

# 🚀 6. Инструкция по запуску

### 1) Клонирование репозитория

```bash
git clone https://github.com/beopryang/nlpir_ks031_A.git
cd nlpir_ks031_A/root
```

### 2) Установка библиотек

```bash
pip install -r requirements.txt
```

### 3) Загрузка данных с Google Drive

Сохранить файлы в:

```
root/data/
    minutes.parquet
    segments_all.parquet
    base_minutes_rag.parquet
```

### 4) Выполнение поиска

```bash
python src/search_naive.py
python src/search_ontology.py
```

### 5) Генерация статей

```bash
python src/generate_naive.py
python src/generate_ontology.py
```

### 6) Оценка

```bash
python src/evaluate_absolute.py
```

---

# 🧪 7. Воспроизводимость

* Все результаты в `results/` включены
* Все настройки и промпты в `config/` доступны
* Код поиска/генерации/оценки полностью предоставлен
* Большие данные доступны через Google Drive
* Текст генерации может немного отличаться,
  но показатели фактической точности и релевантности воспроизводимы

---

# 🇫🇷 **Version Française**

<a name="french-version"></a>

# 🏛 Développement d’un système de génération d’articles de presse à partir des comptes rendus des assemblées législatives coréennes utilisant RAG

---

## 📌 Vue d’ensemble

Ce dépôt contient le code de recherche pour un **système RAG hybride (Retrieval-Augmented Generation)** conçu pour traiter les comptes rendus des délibérations du **Conseil métropolitain de Gwangju** et du **Conseil métropolitain de Séoul**.
Le système combine **un filtrage des métadonnées basé sur une ontologie** avec une **recherche par similarité via embeddings**.

L’étude compare deux stratégies de récupération :

1. **Naive RAG** – récupération basée uniquement sur les embeddings, sans prétraitement
2. **Ontology RAG** – récupération restreinte par des métadonnées (assemblée, orateur, parti, commission) avant la recherche via embeddings

Cette comparaison permet d’évaluer comment le filtrage structurel améliore :

* la factualité
* la pertinence thématique
* la stabilité de la récupération
* la robustesse aux erreurs

Le jeu de données couvre **juillet 2022 – octobre 2025**, et l’évaluation repose sur **100 requêtes de référence**.

---

# 📁 Structure du dépôt

```
root/
├─ src/                     # Code RAG, ontologie et évaluation
├─ config/                  # Configuration LLM (planification & évaluation)
├─ results/                 # Résultats de récupération et d’évaluation (inclus)
└─ data/                    # Données volumineuses (via Google Drive)
```

---

# 📊 1. Description des données

## 1.1 results/ (inclus dans GitHub)

Ce dossier contient tous les résultats générés lors des expériences.

### 🔍 Résultats de récupération

* `naive_rag_results_top5.csv`
* `ontology_rag_results_tasks.csv`

### 📰 Articles générés

* `naive_rag_articles.csv`
* `naive_rag_articles.jsonl`
* `ontology_rag_articles.csv`
* `ontology_rag_articles.jsonl`

### ✔ Évaluation de la factualité

* `eval_top5_truth_naive_top5_absolute.csv`
* `eval_top5_truth_onto_top5_absolute.csv`

### 🧪 Plans d’évaluation

* `eval_plans_onto_top5.json`

### 🧾 Requêtes de référence

* `test_queries.csv` (100 requêtes)

Tous les fichiers sont fournis afin d’assurer la reproductibilité complète.

---

## 1.2 data/ (Google Drive — fichiers volumineux)

Les données volumineuses originales ne sont pas stockées sur GitHub et sont fournies via Google Drive.
Seuls **trois fichiers** sont inclus.

📥 Lien de téléchargement
**[https://drive.google.com/drive/folders/1_LP9o4K7Z6XR5xku7bEJc15pYAWp4hpP?usp=sharing](https://drive.google.com/drive/folders/1_LP9o4K7Z6XR5xku7bEJc15pYAWp4hpP?usp=sharing)**

### Fichiers inclus

| Nom du fichier             | Description                               |
| -------------------------- | ----------------------------------------- |
| `minutes.parquet`          | Comptes rendus complets (Gwangju + Séoul) |
| `segments_all.parquet`     | Tous les segments par intervenant         |
| `base_minutes_rag.parquet` | Données prétraitées pour Naive RAG        |

### Emplacement local

Placez les fichiers téléchargés dans :

```
root/data/
    minutes.parquet
    segments_all.parquet
    base_minutes_rag.parquet
```

---

# ⚙️ 2. Description du code (src/)

### 🔹 `search_naive.py`

Effectue une récupération basée sur les embeddings sur l’ensemble des segments (FAISS + similarité cosinus).

### 🔹 `search_ontology.py`

Filtre les segments selon l’ontologie (assemblée, intervenant, parti, commission),
puis effectue la récupération via embeddings.
→ Réduit les incohérences et améliore la factualité.

### 🔹 `generate_naive.py`

Génère des articles journalistiques à partir des 5 meilleurs segments du Naive RAG.
LLM : **gpt-4.1-mini**

### 🔹 `generate_ontology.py`

Génère des articles à partir des résultats du Ontology RAG Top-5.
LLM : **gpt-4.1-mini**

### 🔹 `evaluate_absolute.py`

Évalue la factualité et la pertinence thématique.

Les erreurs suivantes sont considérées comme **fortes erreurs factuelles** :

* WRONG_COUNCIL
* WRONG_PERSON
* WRONG_PARTY

(score factuel = 0)

La pertinence thématique est notée de 1 à 10.

### 🔹 `index_ontology.py`

Construit et met en cache l’index des métadonnées de l’ontologie.

### 🔹 `pkl_ontology.py`

Sérialise les structures ontologiques.

### 🔹 `paths.py`

Centralise les chemins de configuration.

---

# 📑 3. Configuration expérimentale

### Jeu de données

* Assemblées : Gwangju et Séoul
* Période : juillet 2022 – octobre 2025
* Évaluation : 100 requêtes de référence

### Modèles comparés

| Modèle       | Description                                        |
| ------------ | -------------------------------------------------- |
| Naive RAG    | Récupération via embeddings sur tous les segments  |
| Ontology RAG | Filtrage ontologique + récupération via embeddings |

### Modèles utilisés

| Usage                 | Modèle                 |
| --------------------- | ---------------------- |
| Génération d’articles | gpt-4.1-mini           |
| Évaluation            | gpt-4.1-mini           |
| Embeddings            | text-embedding-3-large |

---

# 📈 4. Méthode d’évaluation

### ✔ 1) Évaluation factuelle (fact-level)

Erreurs considérées comme **fortes erreurs factuelles** :

* WRONG_COUNCIL
* WRONG_PERSON
* WRONG_PARTY

Ontology RAG : supprime les requêtes EMPTY_SEGMENT
Naive RAG : attribue un score **0** pour tout segment récupéré appartenant à ces requêtes

---

### ✔ 2) Pertinence thématique (topic_score)

Le LLM évalue dans quelle mesure les segments récupérés correspondent au sujet demandé.
Note de 1 à 10.

---

# 📊 5. Résultats

## 🔥 1) Taux d’erreur factuelle

| Modèle       | #Segments | #Erreurs | Taux d’erreur |
| ------------ | --------- | -------- | ------------- |
| Naive RAG    | 500       | 161      | **32,20%**    |
| Ontology RAG | 610       | 43       | **7,05%**     |

**→ Réduction de 25,15 points (≈ 78% de réduction relative)**

---

## 🎯 2) Pertinence thématique moyenne

| Modèle       | Score moyen (sur 10) |
| ------------ | -------------------- |
| Naive RAG    | 5,77                 |
| Ontology RAG | 6,54                 |

**→ Amélioration de +7,66%**, avec beaucoup moins d’échecs catastrophiques.

---

## 📌 Principales conclusions

Par rapport au Naive RAG, l’Ontology RAG apporte :

* une réduction importante des erreurs factuelles
* une meilleure pertinence thématique
* l’élimination des échecs de récupération à 0 point
* des résultats plus cohérents et stables

Ces gains proviennent de l’ajout de contraintes structurelles avant la recherche sémantique.

---

# 🚀 6. Exécution

### 1) Cloner le dépôt

```bash
git clone https://github.com/beopryang/nlpir_ks031_A.git
cd nlpir_ks031_A/root
```

### 2) Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3) Télécharger les données Google Drive

Placez les fichiers dans :

```
root/data/
    minutes.parquet
    segments_all.parquet
    base_minutes_rag.parquet
```

### 4) Lancer la récupération

```bash
python src/search_naive.py
python src/search_ontology.py
```

### 5) Générer les articles

```bash
python src/generate_naive.py
python src/generate_ontology.py
```

### 6) Évaluer

```bash
python src/evaluate_absolute.py
```

---

# 🧪 7. Reproductibilité

* Tous les fichiers dans `results/` sont fournis
* Toutes les configurations dans `config/` sont incluses
* Tout le code RAG/génération/évaluation est dans `src/`
* Les données volumineuses sont publiques via Google Drive
* Malgré des variations mineures possibles dans les textes générés,
  **les métriques de factualité et de pertinence sont reproductibles de manière stable**

---

# 🇪🇸 Spanish Version

<a name="spanish-version"></a>

# 🏛 Desarrollo de un Sistema de Generación de Noticias a partir de Actas de Deliberaciones de los Poderes Legislativos de Corea utilizando RAG

---

## 📌 Resumen

Este repositorio contiene el código de investigación para un **sistema híbrido RAG (Retrieval-Augmented Generation)** que procesa las actas de deliberaciones del Consejo Metropolitano de Gwangju y del Consejo Metropolitano de Seúl.
El sistema integra **filtrado de metadatos basado en ontologías** con una **búsqueda por similitud mediante embeddings**.

El estudio compara dos estrategias de recuperación:

1. **Naive RAG** – recuperación basada únicamente en embeddings sin preprocesamiento
2. **Ontology RAG** – recuperación restringida mediante metadatos (cámara, orador, partido, comité) antes de la búsqueda por embeddings

A través de esta comparación, la investigación evalúa cómo el filtrado estructural mejora:

* la factualidad
* la relevancia temática
* la estabilidad de la recuperación
* la robustez frente a errores

El conjunto de datos abarca **julio de 2022 – octubre de 2025**, y la evaluación se realiza mediante **100 consultas de referencia**.

---

# 📁 Estructura del Repositorio

```
root/
├─ src/                     # Código de recuperación RAG, ontología y evaluación
├─ config/                  # Configuración del planificador LLM y evaluación
├─ results/                 # Resultados de recuperación, salida y evaluación
└─ data/                    # Datos originales grandes (Google Drive)
```

---

# 📊 1. Descripción de los Datos

## 1.1 results/ (incluido en GitHub)

Esta carpeta contiene todos los resultados generados por los experimentos.

### 🔍 Resultados de Recuperación

* `naive_rag_results_top5.csv`
* `ontology_rag_results_tasks.csv`

### 📰 Artículos Generados

* `naive_rag_articles.csv`
* `naive_rag_articles.jsonl`
* `ontology_rag_articles.csv`
* `ontology_rag_articles.jsonl`

### ✔ Evaluación de Factualidad

* `eval_top5_truth_naive_top5_absolute.csv`
* `eval_top5_truth_onto_top5_absolute.csv`

### 🧪 Planes de Evaluación

* `eval_plans_onto_top5.json`

### 🧾 Consultas de Referencia

* `test_queries.csv` (100 consultas)

Todos los archivos están incluidos para garantizar reproducibilidad completa.

---

## 1.2 data/ (Google Drive — archivos grandes)

Los archivos de datos grandes no se almacenan directamente en GitHub y se proporcionan mediante Google Drive.
Solo se incluyen **tres archivos**.

📥 **Enlace de Descarga**
[https://drive.google.com/drive/folders/1_LP9o4K7Z6XR5xku7bEJc15pYAWp4hpP?usp=sharing](https://drive.google.com/drive/folders/1_LP9o4K7Z6XR5xku7bEJc15pYAWp4hpP?usp=sharing)

### Archivos Incluidos

| Nombre del archivo         | Descripción                                 |
| -------------------------- | ------------------------------------------- |
| `minutes.parquet`          | Actas completas originales (Gwangju + Seúl) |
| `segments_all.parquet`     | Todos los segmentos a nivel de orador       |
| `base_minutes_rag.parquet` | Datos preprocesados para Naive RAG          |

### Ubicación Local

Copie los archivos descargados en:

```
root/data/
    minutes.parquet
    segments_all.parquet
    base_minutes_rag.parquet
```

---

# ⚙️ 2. Descripción del Código (src/)

### 🔹 `search_naive.py`

Realiza recuperación basada en embeddings sobre todos los segmentos (FAISS + similitud coseno).

### 🔹 `search_ontology.py`

Filtra segmentos usando metadatos ontológicos (cámara, orador, partido, comité),
luego realiza recuperación basada en embeddings.
→ Reduce errores y mejora la factualidad.

### 🔹 `generate_naive.py`

Genera artículos estilo noticia a partir de los 5 principales resultados de Naive RAG.
LLM: **gpt-4.1-mini**

### 🔹 `generate_ontology.py`

Genera artículos usando los 5 principales resultados de Ontology RAG.
LLM: **gpt-4.1-mini**

### 🔹 `evaluate_absolute.py`

Evalúa factualidad y relevancia temática.

Errores considerados **fuertes errores fácticos**:

* WRONG_COUNCIL
* WRONG_PERSON
* WRONG_PARTY

→ En estos casos, la puntuación de factualidad se fija en *0*.

La relevancia temática se puntúa de 1 a 10.

### 🔹 `index_ontology.py`

Construye y almacena índices de metadatos ontológicos.

### 🔹 `pkl_ontology.py`

Serializa estructuras de metadatos de ontologías.

### 🔹 `paths.py`

Configuración centralizada de rutas de directorios.

---

# 📑 3. Configuración Experimental

### Conjunto de Datos

* Cámaras: Gwangju y Seúl
* Período: julio 2022 – octubre 2025
* Evaluación: 100 consultas de referencia

### Modelos Comparados

| Modelo       | Descripción                                    |
| ------------ | ---------------------------------------------- |
| Naive RAG    | Recuperación basada en embeddings              |
| Ontology RAG | Filtrado estructural + búsqueda por embeddings |

### Modelos Utilizados

| Propósito           | Modelo                 |
| ------------------- | ---------------------- |
| Generación de texto | gpt-4.1-mini           |
| Evaluación          | gpt-4.1-mini           |
| Embeddings          | text-embedding-3-large |

---

# 📈 4. Método de Evaluación

### ✔ 1) Evaluación de Factualidad

Los siguientes se consideran **errores fácticos severos**:

* WRONG_COUNCIL
* WRONG_PERSON
* WRONG_PARTY

Ontology RAG: elimina consultas sin segmentos (EMPTY_SEGMENT).
Naive RAG: asigna **0 puntos** si cualquier segmento recuperado pertenece a esas consultas.

---

### ✔ 2) Evaluación de Relevancia Temática (topic_score)

El LLM evalúa qué tan bien los segmentos recuperados coinciden con el tema solicitado.
Escala: 1 a 10.

---

# 📊 5. Resumen de Resultados

## 🔥 1) Tasa de Error Fáctico

| Modelo       | #Segmentos | #Errores | Tasa de Error |
| ------------ | ---------- | -------- | ------------- |
| Naive RAG    | 500        | 161      | **32.20%**    |
| Ontology RAG | 610        | 43       | **7.05%**     |

**→ Reducción de 25.15 puntos porcentuales (≈ 78% menos errores)**

---

## 🎯 2) Relevancia Temática Promedio

| Modelo       | Puntuación media (máx. 10) |
| ------------ | -------------------------- |
| Naive RAG    | 5.77                       |
| Ontology RAG | 6.54                       |

**→ Mejora del +7.66%**, con muchos menos casos catastróficos de baja puntuación.

---

## 📌 Conclusiones Principales

Comparado con Naive RAG, Ontology RAG demuestra:

* gran reducción de errores fácticos
* mejor alineación temática
* eliminación de fallos de recuperación extremos
* resultados más consistentes y estables

Esto se logra mediante la imposición de restricciones estructurales antes de la búsqueda semántica.

---

# 🚀 6. Cómo Ejecutar

### 1) Clonar el Repositorio

```bash
git clone https://github.com/beopryang/nlpir_ks031_A.git
cd nlpir_ks031_A/root
```

### 2) Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 3) Descargar Datos desde Google Drive

Coloque los archivos en:

```
root/data/
    minutes.parquet
    segments_all.parquet
    base_minutes_rag.parquet
```

### 4) Ejecutar Recuperación

```bash
python src/search_naive.py
python src/search_ontology.py
```

### 5) Generar Artículos

```bash
python src/generate_naive.py
python src/generate_ontology.py
```

### 6) Evaluar

```bash
python src/evaluate_absolute.py
```

---

# 🧪 7. Reproducibilidad

* Todos los archivos de salida están incluidos en `results/`
* Las configuraciones y prompts están en `config/`
* Todo el código de recuperación/generación/evaluación está en `src/`
* Los datos originales grandes están disponibles en Google Drive
* Aunque el texto generado por LLM puede variar,
  **las métricas de factualidad y relevancia son reproducibles de manera estable**

---

# **🇭🇷 Hrvatska Verzija**

<a name="croatian-version"></a>

# 🏛 Razvoj sustava za generiranje novinskih članaka na temelju zapisnika sjednica korejskih zakonodavnih tijela korištenjem RAG-a

---

## 📌 Pregled

Ovo spremište sadrži istraživački kod za **hibridni RAG sustav (Retrieval-Augmented Generation)** koji obrađuje zapisnike sjednica Gradskog vijeća Gwangjua i Gradskog vijeća Seula.
Sustav kombinira **ontološko filtriranje metapodataka** sa **pretraživanjem sličnosti na temelju ugradnji (embeddings)**.

Studija uspoređuje dvije strategije dohvaćanja:

1. **Naive RAG** – dohvaćanje temeljeno samo na ugradnjama, bez dodatne obrade
2. **Ontology RAG** – dohvaćanje ograničeno metapodacima (vijeće, govornik, stranka, odbor) prije pretrage ugradnjama

Ova usporedba omogućuje procjenu kako strukturalno filtriranje poboljšava:

* točnost činjenica
* tematsku relevantnost
* stabilnost dohvaćanja
* robusnost na pogreške

Skup podataka pokriva **srpanj 2022. – listopad 2025.**, a evaluacija se provodi nad **100 referentnih upita**.

---

# 📁 Struktura Spremišta

```
root/
├─ src/                     # Kod za dohvaćanje RAG-a, ontologiju i evaluaciju
├─ config/                  # Konfiguracija planera LLM-a i evaluacije
├─ results/                 # Rezultati dohvaćanja/izlaza/evaluacije (uključeni)
└─ data/                    # Veliki izvorni podaci (dostupni preko Google Drivea)
```

---

# 📊 1. Opis Podataka

## 1.1 results/ (uključeno na GitHubu)

Mapa sadrži sve izlazne datoteke generirane tijekom eksperimenata.

### 🔍 Rezultati Dohvaćanja

* `naive_rag_results_top5.csv`
* `ontology_rag_results_tasks.csv`

### 📰 Generirani Članci

* `naive_rag_articles.csv`
* `naive_rag_articles.jsonl`
* `ontology_rag_articles.csv`
* `ontology_rag_articles.jsonl`

### ✔ Evaluacija Činjenične Točnosti

* `eval_top5_truth_naive_top5_absolute.csv`
* `eval_top5_truth_onto_top5_absolute.csv`

### 🧪 Evaluacijski Planovi

* `eval_plans_onto_top5.json`

### 🧾 Referentni Upiti

* `test_queries.csv` (100 upita)

Sve datoteke su uključene kako bi se omogućila potpuna reprodukcija rezultata.

---

## 1.2 data/ (Google Drive — velike datoteke)

Veliki izvorni podaci nisu pohranjeni na GitHubu i dostupni su putem Google Drivea.
Uključene su samo **tri datoteke**.

📥 Link za preuzimanje
**[https://drive.google.com/drive/folders/1_LP9o4K7Z6XR5xku7bEJc15pYAWp4hpP?usp=sharing](https://drive.google.com/drive/folders/1_LP9o4K7Z6XR5xku7bEJc15pYAWp4hpP?usp=sharing)**

### Uključene datoteke

| Naziv datoteke             | Opis                                         |
| -------------------------- | -------------------------------------------- |
| `minutes.parquet`          | Izvorni zapisnici sjednica (Gwangju + Seoul) |
| `segments_all.parquet`     | Svi segmenti govornika                       |
| `base_minutes_rag.parquet` | Predobrađeni podaci za Naive RAG             |

### Lokalno postavljanje

Datoteke trebate smjestiti u:

```
root/data/
    minutes.parquet
    segments_all.parquet
    base_minutes_rag.parquet
```

---

# ⚙️ 2. Opis Koda (src/)

### 🔹 `search_naive.py`

Dohvaća segmente na temelju ugradnji (FAISS + kosinusna sličnost).

### 🔹 `search_ontology.py`

Filtrira segmente korištenjem ontoloških metapodataka (vijeće, govornik, stranka, odbor),
zatim pokreće pretragu ugradnjama.
→ Sprječava pogrešno podudaranje i poboljšava činjeničnu točnost.

### 🔹 `generate_naive.py`

Generira novinske članke iz Naive RAG Top-5 rezultata.
LLM: **gpt-4.1-mini**

### 🔹 `generate_ontology.py`

Generira članke iz Ontology RAG Top-5 rezultata.
LLM: **gpt-4.1-mini**

### 🔹 `evaluate_absolute.py`

Provodi apsolutnu evaluaciju činjenične točnosti i tematske relevantnosti.

* WRONG_COUNCIL
* WRONG_PERSON
* WRONG_PARTY
  → tretira se kao **velika činjenična pogreška** (fact_ok = 0)

Tematska relevantnost ocjenjuje se skalom 1–10.

### 🔹 Ostale skripte

* `index_ontology.py` — izrada indeksa ontologije
* `pkl_ontology.py` — serijalizacija ontoloških struktura
* `paths.py` — centralna konfiguracija putanja

---

# 📑 3. Eksperimentalna Postavka

### Skup Podataka

* Vijeća: Gwangju i Seoul
* Period: 7/2022 – 10/2025
* Evaluacija: 100 referentnih upita

### Uspoređeni Modeli

| Model        | Opis                                  |
| ------------ | ------------------------------------- |
| Naive RAG    | Dohvaćanje ugradnjama bez ograničenja |
| Ontology RAG | Filtriranje + dohvaćanje ugradnjama   |

### Modeli korišteni

| Namjena             | Model                  |
| ------------------- | ---------------------- |
| Generiranje članaka | gpt-4.1-mini           |
| Evaluacija          | gpt-4.1-mini           |
| Embeddings          | text-embedding-3-large |

---

# 📈 4. Metoda Evaluacije

### ✔ 1) Evaluacija Činjenične Točnosti

Sljedeće se smatra **teškim činjeničnim pogreškama**:

* WRONG_COUNCIL
* WRONG_PERSON
* WRONG_PARTY

Ontology RAG: uklanja upite bez segmenata
Naive RAG: dodjeljuje **0 bodova** ako se dohvaća segment iz EMPTY_SEGMENT grupe

---

### ✔ 2) Tematska Relevantnost

LLM ocjenjuje koliko dohvaćeni segmenti odgovaraju temi upita.
Skala: **1–10**

---

# 📊 5. Rezultati

## 🔥 1) Stopa Činjeničnih Pogrešaka

| Model        | #Segmenti | #Pogreške | Stopa Pogrešaka |
| ------------ | --------- | --------- | --------------- |
| Naive RAG    | 500       | 161       | **32.20%**      |
| Ontology RAG | 610       | 43        | **7.05%**       |

**→ Smanjenje za 25.15 postotnih bodova (≈ 78% relativnog smanjenja)**

---

## 🎯 2) Prosječna Tematska Relevantnost

| Model        | Prosječna ocjena (max 10) |
| ------------ | ------------------------- |
| Naive RAG    | 5.77                      |
| Ontology RAG | 6.54                      |

**→ Poboljšanje od +7.66%**, uz znatno manje ekstremno loših ocjena.

---

## 📌 Ključni Zaključci

Ontology RAG u odnosu na Naive RAG donosi:

* veliko smanjenje činjeničnih pogrešaka
* višu tematsku relevantnost
* uklanjanje katastrofalnih 0-bodovnih neuspjeha
* dosljednije i stabilnije dohvaćanje

Strukturalna ograničenja omogućuju čišći i točniji rad semantičke pretrage.

---

# 🚀 6. Kako Pokrenuti

### 1) Klonirajte spremište

```bash
git clone https://github.com/beopryang/nlpir_ks031_A.git
cd nlpir_ks031_A/root
```

### 2) Instalirajte ovisnosti

```bash
pip install -r requirements.txt
```

### 3) Preuzmite podatke s Google Drivea

Smjestite datoteke u:

```
root/data/
    minutes.parquet
    segments_all.parquet
    base_minutes_rag.parquet
```

### 4) Dohvaćanje

```bash
python src/search_naive.py
python src/search_ontology.py
```

### 5) Generiranje članaka

```bash
python src/generate_naive.py
python src/generate_ontology.py
```

### 6) Evaluacija

```bash
python src/evaluate_absolute.py
```

---

# 🧪 7. Reproducibilnost

* Sve izlazne datoteke u `results/` su uključene
* Sve konfiguracije u `config/` dostupne
* Kompletan kod za dohvaćanje/generiranje/evaluaciju nalazi se u `src/`
* Veliki podaci dijeljeni su putem Google Drivea
* Iako se tekst generiran LLM-om može blago razlikovati,
  **metrike točnosti i relevantnosti ostaju dosljedno reproducibilne**

---

# 🇪🇪 **Eesti versioon**

<a name="estonian-version"></a>

# 🏛 Uudisteartiklite genereerimise süsteemi arendamine Korea seadusandlike kogude aruteluprotokollide põhjal, kasutades RAG-mudelit

---

## 📌 Ülevaade

See hoidla sisaldab uurimiskoodi **hübriidse RAG-süsteemi (Retrieval-Augmented Generation)** jaoks, mis töötleb Gwangju ja Souli Metropolitan Council’i aruteluprotokolle.
Süsteem ühendab **ontoloogiapõhise metaandmete filtreerimise** ja **embeedingutel põhineva sarnasuspäringu**.

Uuring võrdleb kahte otsingustrateegiat:

1. **Naive RAG** – puhtalt embeedingutel põhinev otsing ilma eeltöötluseta
2. **Ontology RAG** – otsing, mis toimub enne embeedingupäringut volikogu, kõneleja, partei ja komisjoni metaandmetega filtreerimise kaudu

Võrdluse eesmärk on hinnata, kuidas struktuurne filtreerimine parandab:

* faktilisust
* teemapõhist asjakohasust
* otsingu stabiilsust
* vigade tõrjekindlust

Andmestik hõlmab perioodi **juuli 2022 – oktoober 2025**, ning hindamine toimub **100 kontrollpäringu** abil.

---

# 📁 Hoiu struktuur

```
root/
├─ src/                     # RAG-i otsing, ontoloogia ja hindamise kood
├─ config/                  # LLM-i plaanimise ja hindamise konfiguratsioon
├─ results/                 # Otsingu/ väljundi/ hindamise tulemused
└─ data/                    # Suured algandmed (Google Drive’is)
```

---

# 📊 1. Andmete kirjeldus

## 1.1 results/ (GitHubis kaasas)

See kaust sisaldab kõiki eksperimendi käigus loodud väljundeid.

### 🔍 Otsingutulemused

* `naive_rag_results_top5.csv`
* `ontology_rag_results_tasks.csv`

### 📰 Genereeritud artiklid

* `naive_rag_articles.csv`
* `naive_rag_articles.jsonl`
* `ontology_rag_articles.csv`
* `ontology_rag_articles.jsonl`

### ✔ Faktilisuse hindamine

* `eval_top5_truth_naive_top5_absolute.csv`
* `eval_top5_truth_onto_top5_absolute.csv`

### 🧪 Hindamisplaanid

* `eval_plans_onto_top5.json`

### 🧾 Kontrollpäringud

* `test_queries.csv` (100 päringut)

Kõik failid on lisatud täieliku reprodutseeritavuse tagamiseks.

---

## 1.2 data/ (Google Drive — suured failid)

Originaalandmed on mahukad ja seetõttu ei ole GitHubis.
Kaasa kuulub **ainult kolm faili**.

📥 Allalaadimislink:
**[https://drive.google.com/drive/folders/1_LP9o4K7Z6XR5xku7bEJc15pYAWp4hpP?usp=sharing](https://drive.google.com/drive/folders/1_LP9o4K7Z6XR5xku7bEJc15pYAWp4hpP?usp=sharing)**

### Failid

| Failinimi                  | Kirjeldus                                      |
| -------------------------- | ---------------------------------------------- |
| `minutes.parquet`          | Täismahus aruteluprotokollid (Gwangju + Seoul) |
| `segments_all.parquet`     | Kõik kõneleja tasandi lõigud                   |
| `base_minutes_rag.parquet` | Eeltöödeldud andmed Naive RAG jaoks            |

### Kohalik paigutus

Paiguta need kausta:

```
root/data/
    minutes.parquet
    segments_all.parquet
    base_minutes_rag.parquet
```

---

# ⚙️ 2. Koodi kirjeldus (src/)

### 🔹 `search_naive.py`

Teostab embeedingupõhise otsingu kõikide lõikude seast (FAISS + kosinus-sarnasus).

### 🔹 `search_ontology.py`

Filtreerib lõigud volikogu, kõneleja, partei ja komisjoni metaandmete põhjal,
seejärel rakendab embeedingupäringu.
→ Vähendab ebakõlasid ja parandab faktilisust.

### 🔹 `generate_naive.py`

Genereerib uudisartikleid Naive RAG Top-5 otsingutulemustest.
LLM: **gpt-4.1-mini**

### 🔹 `generate_ontology.py`

Genereerib artikleid Ontology RAG Top-5 tulemustest.
LLM: **gpt-4.1-mini**

### 🔹 `evaluate_absolute.py`

Hindab faktilisust ja teemapõhist asjakohasust.

* WRONG_COUNCIL
* WRONG_PERSON
* WRONG_PARTY
  → loetakse *rasketeks faktilisteks vigadeks* (fact_ok = 0)

### 🔹 `index_ontology.py`

Koostab ja salvestab ontoloogiapõhised indeksid.

### 🔹 `pkl_ontology.py`

Serialiseerib ontoloogilised metaandmed.

### 🔹 `paths.py`

Hoiab ühtset teekonna konfiguratsiooni.

---

# 📑 3. Eksperimentide seadistus

### Andmestik

* Gwangju Metropolitan Council
* Seoul Metropolitan Council
* Periood: juuli 2022 – oktoober 2025
* 100 kontrollpäringut

### Võrreldud mudelid

| Mudel        | Kirjeldus                                      |
| ------------ | ---------------------------------------------- |
| Naive RAG    | Embeedingupõhine otsing kõigi lõikude seast    |
| Ontology RAG | Ontoloogiline filtreerimine + embeedinguotsing |

### Kasutatud mudelid

| Otstarve                | Mudel                  |
| ----------------------- | ---------------------- |
| Artiklite genereerimine | gpt-4.1-mini           |
| Hindamine               | gpt-4.1-mini           |
| Embeedingud             | text-embedding-3-large |

---

# 📈 4. Hindamismeetod

### ✔ 1) Faktilisuse hindamine

Järgmisi käsitletakse **rasketena faktiliste vigadena**:

* WRONG_COUNCIL
* WRONG_PERSON
* WRONG_PARTY

Ontology RAG: eemaldab EMPTY_SEGMENT päringud
Naive RAG: annab **0 punkti**, kui mõni TOP-5 lõik kuulub nende hulka

---

### ✔ 2) Teemapõhine asjakohasus (topic_score)

Mudeli hinnang, kuivõrd lõigud vastavad päringu temaatikale.
Skaala 1–10.

---

# 📊 5. Tulemustest kokkuvõte

## 🔥 1) Faktiliste vigade määr

| Mudel        | #Lõigud | #Vead | Vea määr   |
| ------------ | ------- | ----- | ---------- |
| Naive RAG    | 500     | 161   | **32.20%** |
| Ontology RAG | 610     | 43    | **7.05%**  |

**→ 25.15 protsendipunkti paranemine (≈ 78% suhteline vähenemine)**

---

## 🎯 2) Teemapõhine asjakohasus

| Mudel        | Keskmine (10 p.) |
| ------------ | ---------------- |
| Naive RAG    | 5.77             |
| Ontology RAG | 6.54             |

**→ +7.66% paranemine**, tunduvalt vähem katastroofilisi nullpunkte.

---

## 📌 Peamised järeldused

Ontology RAG tagab:

* märkimisväärselt vähem faktilisi vigu
* kõrgema teemapõhise täpsuse
* nullpunkti juhtumite kadumise
* stabiilsema ja usaldusväärsema otsingu

Parandused tulenevad struktuursete filtrite rakendamisest enne semantilist otsingut.

---

# 🚀 6. Kuidas käivitada

### 1) Repo kloonimine

```bash
git clone https://github.com/beopryang/nlpir_ks031_A.git
cd nlpir_ks031_A/root
```

### 2) Sõltuvuste installimine

```bash
pip install -r requirements.txt
```

### 3) Andmete allalaadimine (Google Drive)

Paiguta failid kausta:

```
root/data/
    minutes.parquet
    segments_all.parquet
    base_minutes_rag.parquet
```

### 4) Otsing

```bash
python src/search_naive.py
python src/search_ontology.py
```

### 5) Artiklite genereerimine

```bash
python src/generate_naive.py
python src/generate_ontology.py
```

### 6) Hindamine

```bash
python src/evaluate_absolute.py
```

---

# 🧪 7. Reprodutseeritavus

* Kõik tulemused on lisatud `results/` kausta
* Kõik konfiguratsioonid on kaustas `config/`
* Kogu otsingu/generatsiooni/hindamise kood on `src/` kaustas
* Suured andmed on Google Drive’is
* LLM-tekst võib veidi varieeruda,
  kuid faktilisuse ja teemapõhise täpsuse näitajad on reprodutseeritavad

---


