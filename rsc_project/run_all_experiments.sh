#!/usr/bin/env bash
#
# RSC 전체 실험 실행 스크립트
# src/main_exp.sh와 동일한 방식으로 모든 모델 × 데이터셋 조합을 순회하며
# rsc_project/main.py --mode eval 을 호출합니다.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

# 기본 실험 파라미터 (환경변수로 덮어쓰기 가능)
NUM_OF_PATHS="${NUM_OF_PATHS:-40}" # Stage 1에서 생성할 reasoning path 개수
# MMLU-Pro_full(약 12k 문항) 전체 평가 시 NUM_OF_SAMPLES를 13000 이상으로 설정하세요.
NUM_OF_SAMPLES="${NUM_OF_SAMPLES:-13000}" # 평가할 최대 샘플 수
BATCH_SIZE="${BATCH_SIZE:-64}" # 배치 크기
TEMPERATURE="${TEMPERATURE:-0.7}"  # 기존 모델용 (Qwen3는 모드별로 다름) 기존 0.7
TOP_P="${TOP_P:-0.95}"  # 기존 모델용 (Qwen3는 모드별로 다름) 기존 0.95
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"  # GPU 메모리 사용률 (0.0 ~ 1.0)
SEED="${SEED:-42}"  # vLLM 생성 시드 (SEED=none 이면 비결정적 생성)
DEVICE="${DEVICE:-cuda}"
RESULTS_DIR="${RESULTS_DIR:-experiments/results}"
EVAL_DIR="${EVAL_DIR:-$SCRIPT_DIR/../eval}"
NUM_TRIALS="${NUM_TRIALS:-1}"  # 부트스트래핑 반복 횟수 (기본값: 100)
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-137}"  # 부트스트래핑 시드 (기본값: 42)

# 단일 실행 옵션
# - SINGLE_RUN=true 로 설정하면, 아래 환경변수들(MODEL_LIST/DATASET_LIST/STAGES_CONFIG 등)을
#   사용자가 원하는 단일 조합으로 좁혀 실행합니다. 결과는 eval_result.txt가 저장되는 경로와 동일합니다.
SINGLE_RUN="${SINGLE_RUN:-false}"
if [[ "$SINGLE_RUN" == "true" ]]; then
  export RSC_FORCE_EVAL="1"
fi

# 모드 필터링 (환경변수로 제어)
RUN_THINKING="${RUN_THINKING:-true}"        # thinking 모드 실행 여부
RUN_NON_THINKING="${RUN_NON_THINKING:-true}" # non-thinking 모드 실행 여부

# Stage 1 구성 (Stage 1만 사용)
# STAGES_CONFIG="40" 형식으로 Stage 1의 n 값만 지정
# STAGES_CONFIG가 설정되지 않으면 기본값으로 40 사용
STAGES_CONFIG="${STAGES_CONFIG:-40}"

# STAGES_CONFIG를 배열로 변환 (공백으로 구분)
if [[ -n "$STAGES_CONFIG" ]]; then
  read -r -a STAGES_CONFIG_ARRAY <<< "$STAGES_CONFIG"
else
  STAGES_CONFIG_ARRAY=("40:collective")
fi

# Stage 1의 n 값을 기준으로 내림차순 정렬 (큰 값부터 실행하여 캐시 재사용)
# 각 stages_config에서 첫 번째 값을 추출하여 정렬
declare -a STAGES_CONFIG_SORTED
declare -a STAGES_CONFIG_VALUES

for config in "${STAGES_CONFIG_ARRAY[@]}"; do
  # Stage 1의 n 값 추출 (숫자만 사용)
  first_val=$(echo "$config" | cut -d':' -f1)
  # 숫자인지 확인하고 숫자로 변환
  if [[ "$first_val" =~ ^[0-9]+$ ]]; then
    STAGES_CONFIG_VALUES+=($((first_val)))  # 숫자로 변환
  else
    STAGES_CONFIG_VALUES+=(0)  # 숫자가 아니면 0으로 처리
  fi
  STAGES_CONFIG_SORTED+=("$config")
done

# 버블 정렬 (내림차순) - 간단한 정렬 구현
for i in $(seq 0 $((${#STAGES_CONFIG_SORTED[@]} - 1))); do
  for j in $(seq $(($i + 1)) $((${#STAGES_CONFIG_SORTED[@]} - 1))); do
    if [[ ${STAGES_CONFIG_VALUES[$i]} -lt ${STAGES_CONFIG_VALUES[$j]} ]]; then
      # 값과 config 교체
      temp_val=${STAGES_CONFIG_VALUES[$i]}
      temp_config=${STAGES_CONFIG_SORTED[$i]}
      STAGES_CONFIG_VALUES[$i]=${STAGES_CONFIG_VALUES[$j]}
      STAGES_CONFIG_SORTED[$i]=${STAGES_CONFIG_SORTED[$j]}
      STAGES_CONFIG_VALUES[$j]=$temp_val
      STAGES_CONFIG_SORTED[$j]=$temp_config
    fi
  done
done

STAGES_CONFIG_ARRAY=("${STAGES_CONFIG_SORTED[@]}")

# Qwen3 및 Nemotron 모델 전용 파라미터 (SC_generator_ours.py와 동일)
# Thinking 모드: Temperature=0.6, TopP=0.95, TopK=20, MinP=0
QWEN3_THINKING_TEMP="${QWEN3_THINKING_TEMP:-0.6}"
QWEN3_THINKING_TOP_P="${QWEN3_THINKING_TOP_P:-0.95}"
QWEN3_THINKING_TOP_K="${QWEN3_THINKING_TOP_K:-20}"
QWEN3_THINKING_MIN_P="${QWEN3_THINKING_MIN_P:-0.0}"

# Non-thinking 모드: Temperature=0.7, TopP=0.8, TopK=20, MinP=0
QWEN3_NON_THINKING_TEMP="${QWEN3_NON_THINKING_TEMP:-0.7}"
QWEN3_NON_THINKING_TOP_P="${QWEN3_NON_THINKING_TOP_P:-0.8}"
QWEN3_NON_THINKING_TOP_K="${QWEN3_NON_THINKING_TOP_K:-20}"
QWEN3_NON_THINKING_MIN_P="${QWEN3_NON_THINKING_MIN_P:-0.0}"

# 모델 목록 (환경변수 MODEL_LIST로 덮어쓰기 가능, 공백 구분)
DEFAULT_MODELS=(
  "Qwen/Qwen2.5-3B-Instruct"
  # "Qwen/Qwen2.5-7B-Instruct"
  # "Qwen/Qwen2.5-14B-Instruct"
  # "meta-llama/Llama-3.2-3B-Instruct"
  # "meta-llama/Llama-3.1-8B-Instruct"
  # "google/gemma-3-4b-it"
  # "google/gemma-3-12b-it"
  # "Qwen/Qwen3-4B"
  # "Qwen/Qwen3-8B"
  # nvidia/NVIDIA-Nemotron-Nano-9B-v2
  # "kakaocorp/kanana-1.5-2.1b-instruct-2505"
  # "kakaocorp/kanana-1.5-8b-instruct-2505"
  # "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
  # "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
)
if [[ -n "${MODEL_LIST:-}" ]]; then
  read -r -a MODELS <<< "${MODEL_LIST}"
else
  MODELS=("${DEFAULT_MODELS[@]}")
fi

# 데이터셋 → subject 매핑
declare -A DATASETS=(
  # ["AIME_2024"]="math"
  # ["AIME_2025"]="math"
  # ["MATH500_test"]="math"
  ["MATH_train"]="math"
  # ["MATH2500"]="math"
  # ["AMC23"]="math"
  # ["GPQA-Diamond"]="science"
  # ["GSM8K"]="math"
  # ["DEEPMATH_2600"]="math"
  # ["MMLU-Pro"]="general"
  # ["MMLU-Pro_full"]="general"
)

# 데이터셋 순회 목록 (환경변수 DATASET_LIST로 덮어쓰기 가능)
if [[ -n "${DATASET_LIST:-}" ]]; then
  read -r -a SELECTED_DATASETS <<< "${DATASET_LIST}"
else
  SELECTED_DATASETS=("${!DATASETS[@]}")
fi

is_qwen3_model() {
  local model_lower="${1,,}"
  [[ "$model_lower" == *"qwen3"* || "$model_lower" == *"qwen-3"* ]]
}

is_nemotron_model() {
  local model_lower="${1,,}"
  [[ "$model_lower" == *"nemotron"* || "$model_lower" == *"nvidia-nemotron"* ]]
}

generate_config_multistage() {
  local model_name="$1"
  local enable_thinking="$2"
  local max_tokens="$3"
  local dataset_name="$4"
  local subject="$5"
  local stages_config_str="$6"  # "stage1_n:mode2:stage2_n:mode3:stage3_n:..." 형식
  local tmp_config
  tmp_config="$(mktemp "${SCRIPT_DIR}/experiments/tmp_config_XXXXXX.yaml")"

  # Qwen3 또는 Nemotron 모델인지 확인하고 모드별 파라미터 설정
  local temp_value top_p_value top_k_value min_p_value max_model_len_value
  if is_qwen3_model "$model_name" || is_nemotron_model "$model_name"; then
    if [[ "$enable_thinking" == "true" ]]; then
      temp_value="$QWEN3_THINKING_TEMP"
      top_p_value="$QWEN3_THINKING_TOP_P"
      top_k_value="$QWEN3_THINKING_TOP_K"
      min_p_value="$QWEN3_THINKING_MIN_P"
      max_model_len_value=32768  # Qwen3/Nemotron thinking 모드는 더 긴 입력 지원
    else
      temp_value="$QWEN3_NON_THINKING_TEMP"
      top_p_value="$QWEN3_NON_THINKING_TOP_P"
      top_k_value="$QWEN3_NON_THINKING_TOP_K"
      min_p_value="$QWEN3_NON_THINKING_MIN_P"
      max_model_len_value=16384  # Qwen3/Nemotron non-thinking 모드는 기본값
    fi
  else
    temp_value="$TEMPERATURE"
    top_p_value="$TOP_P"
    top_k_value="null"
    min_p_value="null"
    max_model_len_value=16384  # 기본 모델은 기본값
  fi

  # stages_config_str 파싱: Stage 1의 n 값만 사용
  IFS=':' read -r -a config_parts <<< "$stages_config_str"
  local stage1_n="${config_parts[0]}"
  
  # stages 리스트 구성 (Stage 1만)
  local stages_yaml=""
  stages_yaml+="    - n: $stage1_n"$'\n'
  
  # 구성 slug 생성 (최종 결과 폴더명 = eval_result.txt 저장 경로)
  local stages_config_slug="stages-${stage1_n}"

  cat <<EOF > "$tmp_config"
# 자동 생성된 설정 파일 - 삭제 가능
model:
  name: "$model_name"
  device: "$DEVICE"
  gpu_memory_utilization: $GPU_MEMORY_UTILIZATION
  max_model_len: $max_model_len_value
  enable_thinking: $enable_thinking
  seed: $( [[ "$SEED" == "none" ]] && echo null || echo "$SEED" )

rsc:
  stages:
$stages_yaml
  temperature: $temp_value
  max_tokens: $max_tokens
  top_p: $top_p_value
  top_k: $top_k_value
  min_p: $min_p_value
  final_answer_mode: "stage_k_only"

dataset:
  name: "$dataset_name"
  path: null
  max_samples: $NUM_OF_SAMPLES

subject: "$subject"
prompt_template_path: null
refinement_template_path: null
disable_fewshot: true

evaluation:
  batch_size: $BATCH_SIZE
  save_intermediate: true
  esc_window_size: 5
  ac_beta: 0.95
  num_trials: $NUM_TRIALS
  bootstrap_seed: $BOOTSTRAP_SEED

eval_dir: "$EVAL_DIR"
results_dir: "$RESULTS_DIR"
stages_config_slug: "$stages_config_slug"
cache_dir: "experiments/results_cache"
EOF

  echo "$tmp_config"
}


echo "============================"
echo "RSC 대규모 실험 실행 시작"
echo "모델 목록: ${MODELS[*]}"
echo "데이터셋 목록: ${SELECTED_DATASETS[*]}"
echo "Stage 구성 목록: ${STAGES_CONFIG_ARRAY[*]}"
echo "============================"

for model in "${MODELS[@]}"; do
  echo -e "\n>>> 모델: $model"
  declare -a MODES
  declare -a MODE_LABELS
  declare -a THINKING_FLAGS
  declare -a MAX_TOKEN_VALUES

  if is_qwen3_model "$model" || is_nemotron_model "$model"; then
    # Qwen3 또는 Nemotron 모델: 환경변수에 따라 모드 선택
    MODES=()
    THINKING_FLAGS=()
    MAX_TOKEN_VALUES=()
    
    if [[ "$RUN_NON_THINKING" == "true" ]]; then
      MODES+=("non-thinking")
      THINKING_FLAGS+=("false")
      MAX_TOKEN_VALUES+=(4096)
    fi
    
    if [[ "$RUN_THINKING" == "true" ]]; then
      MODES+=("thinking")
      THINKING_FLAGS+=("true")
      MAX_TOKEN_VALUES+=(38912)
    fi
    
    # 둘 다 비활성화된 경우 기본값으로 non-thinking만 실행
    if [[ ${#MODES[@]} -eq 0 ]]; then
      MODES=("non-thinking")
      THINKING_FLAGS=("false")
      MAX_TOKEN_VALUES=(4096)
    fi
  else
    # Qwen3/Nemotron가 아닌 모델: 기본 모드만 실행
    MODES=("default")
    THINKING_FLAGS=("false")
    MAX_TOKEN_VALUES=(4096)
  fi

  for idx in "${!MODES[@]}"; do
    mode="${MODES[$idx]}"
    thinking="${THINKING_FLAGS[$idx]}"
    max_tokens="${MAX_TOKEN_VALUES[$idx]}"

    echo -e "\n  -- 모드: $mode (enable_thinking=$thinking, max_tokens=$max_tokens)"

    for dataset in "${SELECTED_DATASETS[@]}"; do
      subject="${DATASETS[$dataset]:-math}"
      dataset_file="$EVAL_DIR/${dataset}.jsonl"
      if [[ ! -f "$dataset_file" ]]; then
        echo "    [SKIP] ${dataset}.jsonl 파일을 찾을 수 없습니다 (${dataset_file})"
        continue
      fi

      echo "    * 데이터셋: $dataset (subject=$subject)"
      
      # Stage 1 실험: STAGES_CONFIG_ARRAY의 각 구성을 순회하며 실행
      if [[ ${#STAGES_CONFIG_ARRAY[@]} -eq 0 ]]; then
        echo "      [SKIP] STAGES_CONFIG가 설정되지 않았습니다."
        continue
      fi
      
      for stages_config in "${STAGES_CONFIG_ARRAY[@]}"; do
        echo "      - RSC 평가: $stages_config"
        config_path="$(generate_config_multistage "$model" "$thinking" "$max_tokens" "$dataset" "$subject" "$stages_config")"
        
        stage1_n="${stages_config%%:*}"
        if [[ "$SINGLE_RUN" == "true" ]]; then
          if is_qwen3_model "$model" || is_nemotron_model "$model"; then
            single_run_base="$RESULTS_DIR/$model/$mode"
          else
            single_run_base="$RESULTS_DIR/$model"
          fi
          single_run_log_path="$single_run_base/stages-${stage1_n}/eval_single_${BOOTSTRAP_SEED}.txt"
          mkdir -p "$(dirname "$single_run_log_path")"
        fi
        
        set +e
        if [[ "$SINGLE_RUN" == "true" ]]; then
          export RSC_EVAL_SINGLE_OUTPUT_PATH="$single_run_log_path"
          "$PYTHON_BIN" "$SCRIPT_DIR/main.py" --mode eval --config "$config_path"
          status=$?
          unset RSC_EVAL_SINGLE_OUTPUT_PATH
        else
          "$PYTHON_BIN" "$SCRIPT_DIR/main.py" --mode eval --config "$config_path"
          status=$?
        fi
        set -e
        
        rm -f "$config_path"
        
        if [[ $status -ne 0 ]]; then
          echo "        [오류] 모델=$model, 모드=$mode, 데이터셋=$dataset, stages=$stages_config 실행 실패 (exit code $status)"
        else
          echo "        [완료] 모델=$model, 모드=$mode, 데이터셋=$dataset, stages=$stages_config"
        fi
      done
    done
  done
done

echo -e "\n모든 실험이 완료되었습니다."

