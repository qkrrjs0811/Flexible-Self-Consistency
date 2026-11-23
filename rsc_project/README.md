# RSC Project

Self-Consistency(SC) 기반 reasoning path를 생성·평가하고, probe 신호(`η_adj` / `number_of_paths`)로 **샘플별 동적 예산(FSC / Ours)** 을 할당하는 실험 코드입니다.

## Overview

1. **Stage1 pool 생성**: 샘플당 최대 N개(기본 40) reasoning path를 vLLM으로 생성
2. **Baseline 비교**: SC, AC, ESC, DSC 등 고정/적응형 샘플링 방법 평가
3. **Ours (FSC)**: `probe/` 의 per-sample `number_of_paths`로 SC 예산을 동적으로 할당해 평가
4. **Ablation / 분석**: fixed-budget, SC↔FSC 전이, entropy–oracle 상관 등

## Directory

```
rsc_project/
├── rsc/                 # RSC 코어 (generator, aggregator, engine)
├── evaluation/          # 데이터 로더, 메트릭, eval 러너
├── dsc/                 # DSC/AC/ESC 등 비교 실험
├── probe/               # 샘플별 number_of_paths (η_adj 기반) — Google Drive 제공, git 미포함
├── experiments/results/ # Stage1 JSON, eval_result.txt — Google Drive 제공, git 미포함
├── eval/                # 반복 실행(sample_1~3) 결과 + aggregate_samples.py — 결과는 Drive 제공
├── visualization/       # 논문용 플롯
├── main.py              # 단일 inference / eval 진입점
├── run_all_experiments.sh
├── evaluate_ours_method.py
├── ablation.py
├── fixed_budget_baseline.py
├── sc_fsc_transition_analysis.py
└── fsc_entropy_oracle_analysis.py
```

## Setup

```bash
conda create -n rasc python=3.10 -y   # 또는 기존 rasc 환경 사용
conda activate rasc
pip install -r requirements.txt       # 저장소 루트 기준 (PyTorch, vLLM 0.11, transformers 등)
cd rsc_project                        # 아래 모든 명령은 rsc_project/ 에서 실행 (dsc 스크립트가 상대경로 사용)
```

평가용 입력 데이터셋은 저장소 루트 `eval/*.jsonl` 에 포함되어 있습니다.

### 실험 결과 데이터 (Google Drive)

용량 문제로 아래 폴더는 git에 포함하지 않고 Google Drive로 제공합니다: `<DRIVE_LINK>`

| 폴더 | 내용 | 필요한 단계 |
|---|---|---|
| `rsc_project/probe/` | 샘플별 `number_of_paths`, `entropy_pred` (공동 1저자 제공 probe 출력) | Ours(FSC), ablation |
| `rsc_project/experiments/results/` | Stage1 reasoning path JSON, DSC 중간 결과, `eval_result.txt` | 논문 수치 재현 (생성 생략) |
| `rsc_project/eval/{model}/` | 반복 실행(sample_1~3) Stage1 JSON 및 결과 | 평균 ± 표준편차 재현 |

다운로드 후 위 경로에 그대로 풀어두면 됩니다.

### 재현성 메모

- probe(`number_of_paths`) 생성 코드는 이 저장소에 포함되어 있지 않습니다. FSC 평가는 Drive의 `probe/` 를 입력으로 사용합니다.
- Stage1 생성은 `SEED` (기본 42)로 vLLM 시드를 고정합니다. 다만 GPU/배치 구성에 따라 완전히 동일하지 않을 수 있으므로, 논문 수치를 정확히 재현하려면 Drive의 Stage1 JSON을 사용하고 생성 단계를 건너뛰세요. (기존 결과는 시드 고정 이전에 생성됨)
- 평가 단계(SC/AC/ESC 서브샘플링)는 `BOOTSTRAP_SEED` 로 결정적입니다.

## Quick Start

실행 순서: **1 → 2 → 3 → 4** (뒤 단계가 앞 단계의 Stage1 결과를 사용).
Drive의 `experiments/results/` 를 받았다면 1단계(생성)는 건너뛸 수 있습니다.

### 1) Reasoning path 생성 (Stage1)

```bash
# 환경변수로 모델/데이터셋/스테이지 제어
MODEL_LIST="Qwen/Qwen2.5-7B-Instruct" \
DATASET_LIST="MATH500_test" \
STAGES_CONFIG=40 \
NUM_OF_PATHS=40 \
./run_all_experiments.sh
```

결과는 `experiments/results/<model>/stages-40/Stage1/<dataset>.json` 에 저장됩니다.

단일 실행:

```bash
python main.py --mode eval --config <config.yaml>
# 또는
python main.py --mode inference --config <config.yaml> --question "..."
```

### 2) DSC 난이도 랭킹 / 3) DSC 비교

```bash
bash dsc/run_all_eval.sh   # DSC 난이도 랭킹 → .../dsc_eval/
bash dsc/run_all_cmp.sh    # DSC 비교 → .../dsc_cmp/, eval_result*.txt 에 DSC 추가
```

### 4) Ours (FSC) 평가

`probe/math/<model>/p_*/` 의 JSON에서 `number_of_paths`를 읽고, Stage1 pool에서 해당 개수만큼 샘플링해 평가합니다.

```bash
python evaluate_ours_method.py
# 옵션 예: --model Qwen2.5-7B-Instruct --dataset MATH500_test
```

### 5) 반복 실행 집계 (mean ± std)

```bash
python eval/aggregate_samples.py   # eval/{model}/sample_k → eval/eval_result_summary.txt
```

### 6) Ablation / 분석

```bash
python ablation.py                      # FSC-Discrete, Fixed/Random-Budget-Matched
python fixed_budget_baseline.py         # SC@1/3/5/10 vs Fixed-Budget vs FSC
python sc_fsc_transition_analysis.py    # SC@40 ↔ FSC 답변 일치
python fsc_entropy_oracle_analysis.py   # entropy vs oracle chain 수
```

## Models & Datasets

- **Models**: Qwen2.5-3B/7B/14B-Instruct, gemma-3-4b-it 등
- **Datasets**: MATH500, AMC23, AIME 2024/2025, GPQA-Diamond, MMLU-Pro(+full)
