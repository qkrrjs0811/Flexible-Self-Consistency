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
├── probe/               # 샘플별 number_of_paths (η_adj 기반)
├── experiments/results/ # Stage1 JSON, eval_result.txt
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

저장소 루트의 `requirements.txt` 기준 (Python, PyTorch, vLLM, transformers 등).

```bash
cd rsc_project
```

평가용 데이터는 저장소 `eval/` (jsonl)을 사용합니다.

## Quick Start

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

### 2) Baseline (DSC 등) 평가

```bash
cd dsc
./run_all_cmp.sh
```

### 3) Ours (FSC) 평가

`probe/math/<model>/p_*/` 의 JSON에서 `number_of_paths`를 읽고, Stage1 pool에서 해당 개수만큼 샘플링해 평가합니다.

```bash
python evaluate_ours_method.py
# 옵션 예: --model Qwen2.5-7B-Instruct --dataset MATH500_test
```

### 4) Ablation / 분석

```bash
python ablation.py                      # FSC-Discrete, Fixed/Random-Budget-Matched
python fixed_budget_baseline.py         # SC@1/3/5/10 vs Fixed-Budget vs FSC
python sc_fsc_transition_analysis.py    # SC@40 ↔ FSC 답변 일치
python fsc_entropy_oracle_analysis.py   # entropy vs oracle chain 수
```

## Models & Datasets

- **Models**: Qwen2.5-3B/7B/14B-Instruct, gemma-3-4b-it 등
- **Datasets**: MATH500, AMC23, AIME 2024/2025, GPQA-Diamond, MMLU-Pro(+full)
