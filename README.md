# Flexible Self-Consistency (FSC)

Self-Consistency(SC)는 질문마다 고정된 개수(N)의 reasoning path를 샘플링해 다수결로 답을 정합니다.
**FSC(Ours)** 는 probe가 예측한 문항별 불확실성(`η_adj`)으로 **샘플마다 필요한 path 수(`number_of_paths`)를 동적으로 할당**해,
SC와 비슷하거나 더 높은 정확도를 더 적은 생성 비용으로 달성합니다.

이 저장소는 FSC와 베이스라인(CoT, SC, AC, ESC, DSC)의 **실험 · 평가 · 분석 코드**를 담고 있습니다.

---

## Contents

- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Data (Google Drive)](#data-google-drive)
- [Pipeline](#pipeline)
- [Analysis & Visualization](#analysis--visualization)
- [Reproducibility Notes](#reproducibility-notes)

---

## Repository Structure

```
.
├── eval/                         # 입력 데이터셋 (jsonl): MATH500, AMC23, AIME 2024/2025, GPQA-Diamond, MMLU-Pro, ...
├── requirements.txt
└── rsc_project/                  # ★ 메인 실험 코드 (모든 명령은 여기서 실행)
    ├── main.py                   # Stage1 생성 / 평가 진입점
    ├── run_all_experiments.sh    # [1] 모델 × 데이터셋 Stage1 생성 + CoT/SC/AC/ESC 평가
    ├── dsc/                      # [2][3] DSC 베이스라인 (난이도 랭킹, 비교)
    ├── evaluate_ours_method.py   # [4] FSC(Ours) 평가
    ├── eval/aggregate_samples.py # [5] 반복 실행(sample_1~3) mean ± std 집계
    ├── ablation.py               # FSC-Discrete, Fixed/Random-Budget-Matched
    ├── fixed_budget_baseline.py  # SC@k vs Fixed-Budget vs FSC
    ├── sc_fsc_transition_analysis.py
    ├── fsc_entropy_oracle_analysis.py
    ├── rsc/                      # reasoning path 생성기(vLLM), 답 집계
    ├── evaluation/               # 데이터 로더, 메트릭(CoT/SC/AC/ESC), eval 러너
    ├── visualization/            # 논문 figure 스크립트
    ├── probe/                    # (Drive) 문항별 number_of_paths
    └── experiments/results/      # (Drive) Stage1 path 및 평가 결과
```

> `src/` 는 초기 버전(confidence 기반 early stopping) 코드로, 현재 실험 파이프라인에서는 사용하지 않습니다.

**Models**: Qwen2.5-3B/7B/14B-Instruct, gemma-3-4b-it
**Datasets**: MATH500, AMC23, AIME 2024, AIME 2025, GPQA-Diamond, MMLU-Pro (+full)

---

## Setup

```bash
git clone https://github.com/qkrrjs0811/Flexible-Self-Consistency.git
cd Flexible-Self-Consistency

conda create -n rasc python=3.10 -y
conda activate rasc
pip install -r requirements.txt      # torch 2.8.0, vllm 0.11.0, transformers 등

cd rsc_project                       # 이후 모든 명령은 rsc_project/ 에서 실행
```

- CUDA 버전에 맞는 PyTorch / vLLM 설치는 각 공식 가이드를 참고하세요.
- 모델 가중치는 HuggingFace에서 자동 다운로드됩니다 (gemma는 HF 로그인 및 라이선스 동의 필요).

---

## Data (Google Drive)

용량 문제로 아래 데이터는 git에 포함하지 않고 Google Drive로 제공합니다.

**Download**: `<DRIVE_LINK>`

| 파일 | 압축 해제 위치 | 내용 | 필요한 단계 |
|---|---|---|---|
| `probe.zip` | `rsc_project/probe/` | 문항별 `number_of_paths`, `entropy_pred` (probe 출력) | FSC 평가, ablation |
| `experiments.zip` | `rsc_project/experiments/` | Stage1 reasoning path(40개/문항), DSC 중간 결과, `eval_result.txt` | 논문 수치 재현 |
| `eval.zip` | `rsc_project/eval/` | 반복 실행(sample_1~3) Stage1 JSON 및 결과 | mean ± std 재현 |

```bash
cd rsc_project
unzip probe.zip && unzip experiments.zip && unzip eval.zip
```

> probe(`number_of_paths`)를 생성하는 코드는 이 저장소에 포함되어 있지 않습니다. FSC 평가는 Drive의 `probe/` 를 입력으로 사용합니다.

---

## Pipeline

뒤 단계가 앞 단계의 Stage1 결과를 사용하므로 **[1] → [5] 순서**로 실행합니다.
Drive의 `experiments/` 를 받았다면 [1]의 생성은 건너뛰고 [2]부터 실행해도 됩니다.

### [1] Stage1 reasoning path 생성 + CoT / SC / AC / ESC

```bash
MODEL_LIST="Qwen/Qwen2.5-7B-Instruct" \
DATASET_LIST="MATH500_test AMC23" \
NUM_OF_PATHS=40 SEED=42 \
bash run_all_experiments.sh
```

| 환경변수 | 기본값 | 설명 |
|---|---|---|
| `MODEL_LIST` / `DATASET_LIST` | 스크립트 내 목록 | 공백으로 구분 |
| `NUM_OF_PATHS` | 40 | 문항당 생성할 path 수 (pool 크기) |
| `TEMPERATURE` / `TOP_P` | 0.7 / 0.95 | 샘플링 파라미터 |
| `SEED` | 42 | vLLM 생성 시드 (`none` 이면 비결정적) |
| `BOOTSTRAP_SEED` | 137 | 평가 서브샘플링 시드 |
| `RESULTS_DIR` | `experiments/results` | 출력 경로 |

출력: `experiments/results/<model>/stages-40/Stage1/<dataset>.json`, `eval_result.txt`

### [2] DSC 난이도 랭킹 → [3] DSC 비교

```bash
bash dsc/run_all_eval.sh    # → .../stages-40/dsc_eval/
bash dsc/run_all_cmp.sh     # → .../stages-40/dsc_cmp/, eval_result*.txt 에 DSC 결과 추가
```

모델/데이터셋 목록은 각 스크립트 상단의 `MODELS`, `DATASETS` 에서 수정합니다.

### [4] FSC (Ours) 평가

`probe/math/<model>/p_<p>/<dataset>.json` 의 `number_of_paths` 만큼 Stage1 pool에서 path를 샘플링해 다수결로 평가합니다.

```bash
python evaluate_ours_method.py                                  # 전체 모델 × 데이터셋, p=1.0
python evaluate_ours_method.py --model Qwen2.5-7B-Instruct --dataset MATH500_test
python evaluate_ours_method.py --all_p_values                   # 모든 p_* 설정 평가
```

결과는 `eval_result.txt` 에 `Ours_<probe>(p_x)` 라인으로 추가됩니다.

### [5] 반복 실행 집계 (mean ± std)

```bash
python eval/aggregate_samples.py                 # eval/<model>/sample_k → eval/eval_result_summary.txt
python eval/aggregate_samples.py --summary_only  # 기존 결과로 요약만 다시 생성
```

---

## Analysis & Visualization

```bash
python ablation.py                      # FSC-Discrete, Fixed/Random-Budget-Matched
python fixed_budget_baseline.py         # SC@1/3/5/10 vs Fixed-Budget vs FSC
python sc_fsc_transition_analysis.py    # SC@40 ↔ FSC 답 일치/전이 분석
python fsc_entropy_oracle_analysis.py   # entropy_pred vs oracle 필요 path 수
```

`visualization/` 의 스크립트로 논문 figure를 생성합니다.

| 스크립트 | 내용 |
|---|---|
| `visualize_chains_vs_accuracy_grid.py` | 데이터셋별 평균 chain 수 vs 정확도 (SC/ESC/AC/DSC/Ours) |
| `visualize_mmlu_pro_category_radar.py` | MMLU-Pro 카테고리별 정확도 / 토큰 radar |
| `visualize_answer_diversity_math.py` | MATH 난이도(level 1–5)별 SC 고유 답 개수 |
| `visualize_sc_consensus_chains_by_level.py` | 난이도별 SC majority 수렴에 필요한 최소 chain 수 |
| `dsc_ours_per_sample_paths.py`, `dsc_ours_math500_level.py` | DSC vs FSC 문항별 path 할당 비교 |
| `summarize_eval_results.py` | `eval_result.txt` 표 요약 |

---

## Reproducibility Notes

- **평가 단계는 결정적**입니다. 같은 Stage1 JSON과 `BOOTSTRAP_SEED` 를 쓰면 동일한 수치가 나옵니다.
  논문 수치를 정확히 재현하려면 Drive의 Stage1 결과를 사용하세요.
- **생성 단계**는 `SEED` 로 vLLM 시드를 고정하지만, GPU/배치 구성에 따라 완전히 동일하지 않을 수 있습니다.
  (Drive의 기존 결과는 시드 고정 이전에 생성되었습니다.)
- 모든 스크립트는 `rsc_project/` 를 작업 디렉토리로 가정합니다 (상대 경로 사용).
