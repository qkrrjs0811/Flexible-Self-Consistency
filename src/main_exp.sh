#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

# 실험별 기본 파라미터 (이 값을 직접 수정해서 사용)
NUM_OF_PATHS=40
NUM_OF_SAMPLES=1400
BATCH_SIZE="${BATCH_SIZE:-32}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
PROMPT_TEMPLATE="${PROMPT_TEMPLATE:-$REPO_ROOT/src/prompt_template.json}"
EVAL_DIR="${EVAL_DIR:-$REPO_ROOT/eval}"
DISABLE_FEWSHOT="${DISABLE_FEWSHOT:-true}"
# 모드 필터링 (환경변수로 제어)
RUN_THINKING="${RUN_THINKING:-true}"        # thinking 모드 실행 여부
RUN_NON_THINKING="${RUN_NON_THINKING:-true}" # non-thinking 모드 실행 여부
# Confidence 추출 설정
ENABLE_CONFIDENCE="${ENABLE_CONFIDENCE:-true}"  # BINARY_FT01 confidence 추출 활성화 여부

DEFAULT_MODELS=(
  # "Qwen/Qwen2.5-3B-Instruct"
  # "Qwen/Qwen2.5-7B-Instruct"
  # "meta-llama/Llama-3.1-8B-Instruct"
  # "google/gemma-3-4b-it"
  # "Qwen/Qwen3-4B"
  "Qwen/Qwen3-8B"
  # nvidia/NVIDIA-Nemotron-Nano-9B-v2
)

if [[ -n "${MODEL_LIST:-}" ]]; then
  read -r -a MODELS <<< "${MODEL_LIST}"
else
  MODELS=("${DEFAULT_MODELS[@]}")
fi

declare -A DATASETS=(
  # ["AIME_2024.jsonl"]="math"
  # ["AIME_2025.jsonl"]="math"
  ["MATH500_test.jsonl"]="math"
  # ["GSM8K.jsonl"]="math"
  # ["GPQA-Diamond.jsonl"]="science"
  # ["MMLU-Pro.jsonl"]="general"
)

if [[ "${#MODELS[@]}" -eq 0 ]]; then
  echo "실행할 모델 목록이 비어 있습니다. MODEL_LIST 환경변수를 설정하거나 스크립트를 수정하세요." >&2
  exit 1
fi

missing_any=false
for dataset in "${!DATASETS[@]}"; do
  if [[ ! -f "$EVAL_DIR/$dataset" ]]; then
    echo "경고: $EVAL_DIR/$dataset 파일을 찾을 수 없습니다. 해당 실험은 건너뜁니다." >&2
    missing_any=true
  fi
done

# Qwen3 모델인지 확인하는 함수
is_qwen3_model() {
  local model_name="$1"
  local model_lower="${model_name,,}"
  [[ "$model_lower" == *"qwen3"* ]] || [[ "$model_lower" == *"qwen-3"* ]]
}

# Nemotron 모델인지 확인하는 함수
is_nemotron_model() {
  local model_name="$1"
  local model_lower="${model_name,,}"
  [[ "$model_lower" == *"nemotron"* ]] || [[ "$model_lower" == *"nvidia-nemotron"* ]]
}

pushd "$REPO_ROOT/src" >/dev/null

for model_name in "${MODELS[@]}"; do
  echo "===================="
  echo "모델: $model_name"
  
  # Qwen3 또는 Nemotron 모델인지 확인
  if is_qwen3_model "$model_name"; then
    echo "[Qwen3 모델 감지] Thinking/Non-thinking 모드 선택 실행합니다."
    MODES=()
    if [[ "$RUN_NON_THINKING" == "true" ]]; then
      MODES+=("non-thinking")
    fi
    if [[ "$RUN_THINKING" == "true" ]]; then
      MODES+=("thinking")
    fi
  elif is_nemotron_model "$model_name"; then
    echo "[Nemotron 모델 감지] Thinking/Non-thinking 모드 선택 실행합니다."
    MODES=()
    if [[ "$RUN_NON_THINKING" == "true" ]]; then
      MODES+=("non-thinking")
    fi
    if [[ "$RUN_THINKING" == "true" ]]; then
      MODES+=("thinking")
    fi
  else
    MODES=("default")
  fi

  # 선택된 모드가 없는 경우 건너뜀
  if [[ "${#MODES[@]}" -eq 0 ]]; then
    echo " - $model_name: 선택된 모드가 없어 건너뜁니다. (RUN_THINKING/RUN_NON_THINKING 확인)"
    continue
  fi
  
  for mode in "${MODES[@]}"; do
    if [[ "$mode" == "thinking" ]]; then
      echo "--- Thinking 모드 ---"
      ENABLE_THINKING_FLAG="--enable_thinking"
      MODE_MAX_TOKENS=38912
    elif [[ "$mode" == "non-thinking" ]]; then
      echo "--- Non-thinking 모드 ---"
      ENABLE_THINKING_FLAG=""
      MODE_MAX_TOKENS=4096
    else
      # 기존 모델 (Qwen2.5, Llama, Gemma 등)
      ENABLE_THINKING_FLAG=""
      MODE_MAX_TOKENS="$MAX_TOKENS"
    fi
    
    for jsonl_file in "${!DATASETS[@]}"; do
      subject="${DATASETS[$jsonl_file]}"
      jsonl_path="$EVAL_DIR/$jsonl_file"

      if [[ ! -f "$jsonl_path" ]]; then
        echo " - $jsonl_file (subject=$subject) : 파일 없음 → 건너뜀"
        continue
      fi

      if [[ "$mode" == "thinking" ]] || [[ "$mode" == "non-thinking" ]]; then
        echo " - $jsonl_file (subject=$subject, mode=$mode) 실행 중..."
      else
        echo " - $jsonl_file (subject=$subject) 실행 중..."
      fi
      
      # 기본 명령어 구성
      cmd=(
        "$PYTHON_BIN" SC_generator_ours.py
        --model_name "$model_name"
        --jsonl_file "$jsonl_file"
        --subject "$subject"
        --prompt_link "$PROMPT_TEMPLATE"
        --num_of_paths "$NUM_OF_PATHS"
        --num_of_samples "$NUM_OF_SAMPLES"
        --batch_size "$BATCH_SIZE"
        --temperature "$TEMPERATURE"
        --top_p "$TOP_P"
        --max_tokens "$MODE_MAX_TOKENS"
      )
      
      # Qwen3 또는 Nemotron thinking 모드인 경우 --enable_thinking 플래그 추가
      if [[ -n "$ENABLE_THINKING_FLAG" ]]; then
        cmd+=("$ENABLE_THINKING_FLAG")
      fi
      
      # Few-shot 비활성화 플래그 추가
      if [[ "$DISABLE_FEWSHOT" == "true" ]] || [[ "$DISABLE_FEWSHOT" == "1" ]]; then
        cmd+=("--disable_fewshot")
      fi
      
      # Confidence 추출 플래그 추가
      if [[ "$ENABLE_CONFIDENCE" == "true" ]] || [[ "$ENABLE_CONFIDENCE" == "1" ]]; then
        cmd+=("--enable_confidence")
      fi
      
      "${cmd[@]}"
    done
  done
done

popd >/dev/null

if [[ "$missing_any" == true ]]; then
  echo "일부 데이터셋이 존재하지 않아 해당 실험은 건너뛰었습니다. 경고를 확인하세요." >&2
fi