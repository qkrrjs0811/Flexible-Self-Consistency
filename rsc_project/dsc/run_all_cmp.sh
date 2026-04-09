# 단일 실험 옵션 (run_all_experiments.sh와 동일)
# SINGLE_RUN=true 이면 DSC 결과를 eval_result.txt가 아닌 eval_single_{BOOTSTRAP_SEED}.txt에만 기입 (eval_result.txt 미수정)
# 단일 조합만 실행하려면 MODELS/DATASETS를 한 요소로 설정
SINGLE_RUN="${SINGLE_RUN:-false}"

# 모델 목록 (전체 HF 모델 ID 사용)
MODELS=("Qwen/Qwen2.5-3B-Instruct" "Qwen/Qwen2.5-7B-Instruct" "Qwen/Qwen2.5-14B-Instruct" "google/gemma-3-4b-it")
# MODELS=("meta-llama/Llama-3.1-8B-Instruct")

# 데이터셋 목록 (MMLU-Pro: 1400 서브샘플, MMLU-Pro_full: HF test 전체)
DATASETS=("GSM8K" "MATH500_test" "AMC23" "AIME_2024" "AIME_2025" "MMLU-Pro" "MMLU-Pro_full" "GPQA-Diamond")

# 설정
STRATEGY="random"
RANK_PROMPT="short"
BATCH_SIZE=8
ITERATION=5
# DSC/ESC/AC/SC 평가에서 사용할 체인 수 (STAGES_CONFIG 풀에서 샘플링)
# 사용자가 원하는 규칙: STAGES_CONFIG(pool) 내에서 40개를 샘플링 (pool < 40이면 pool만큼)
N=40
JUDGE_WINDOW_SIZE=24
INTERPLOT_SIZE=16
# STAGES_CONFIG를 환경변수로 설정 가능 (기본값: 40)
# - "40" 또는 "stages-40" 둘 다 허용
STAGES_CONFIG_RAW="${STAGES_CONFIG:-40}"
RESULTS_DIR="experiments/results"

# STAGES_CONFIG 정규화
# - STAGES_CONFIG_NUM: 숫자 (예: 40)
# - STAGES_CONFIG_SLUG: "stages-40"
if [[ "$STAGES_CONFIG_RAW" =~ ^stages-[0-9]+$ ]]; then
  STAGES_CONFIG_NUM="${STAGES_CONFIG_RAW#stages-}"
  STAGES_CONFIG_SLUG="$STAGES_CONFIG_RAW"
elif [[ "$STAGES_CONFIG_RAW" =~ ^[0-9]+$ ]]; then
  STAGES_CONFIG_NUM="$STAGES_CONFIG_RAW"
  STAGES_CONFIG_SLUG="stages-${STAGES_CONFIG_RAW}"
else
  echo "ERROR: Invalid STAGES_CONFIG: '$STAGES_CONFIG_RAW' (expected e.g. 40 or stages-40)"
  exit 1
fi

# pool이 40보다 작으면 N도 그에 맞춰 줄임
if (( STAGES_CONFIG_NUM < N )); then
  N="$STAGES_CONFIG_NUM"
fi

# 부트스트래핑 설정 (run_all_experiments.sh와 동일, 단일 실험 시 eval_single_{seed}.txt 파일명에 사용)
NUM_TRIALS="${NUM_TRIALS:-1}"  # 부트스트래핑 반복 횟수 (기본값: 100)
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-137}"  # 부트스트래핑 시드 (eval_single_${BOOTSTRAP_SEED}.txt)

# 결과 디렉토리 생성
mkdir -p log

# 각 모델과 데이터셋 조합에 대해 실행
for MODEL in "${MODELS[@]}"; do
    # 로그 파일용 안전한 모델명
    MODEL_SAFE=$(echo "$MODEL" | sed 's/\//_/g')
    
    for DATASET in "${DATASETS[@]}"; do
        echo "=========================================="
        echo "Running comparison for Model: $MODEL, Dataset: $DATASET"
        echo "=========================================="
        
        # 평가 결과 파일 경로: experiments/results/{model_name}/stages-{n}/dsc_eval/
        EVAL_FILE="${RESULTS_DIR}/${MODEL}/${STAGES_CONFIG_SLUG}/dsc_eval/${DATASET}_${MODEL_SAFE}_eval_strategy_${STRATEGY}_prompt_${RANK_PROMPT}_batchsize_${BATCH_SIZE}_iter_${ITERATION}_n_1_temp_0.jsonl"
        
        # SC 샘플 파일 경로: experiments/results/{model_name}/stages-{n}/Stage1/{dataset_name}.json
        SC_SAMPLE_FILE="${RESULTS_DIR}/${MODEL}/${STAGES_CONFIG_SLUG}/Stage1/${DATASET}.json"
        
        # DSC용 greedy 샘플 파일 경로 (동일)
        DSC_SAMPLE_FILE="$SC_SAMPLE_FILE"
        
        # 파일 존재 확인
        if [ ! -f "$EVAL_FILE" ]; then
            echo "WARNING: Evaluation file not found: $EVAL_FILE"
            echo "Skipping Model: $MODEL, Dataset: $DATASET"
            echo ""
            continue
        fi
        
        if [ ! -f "$SC_SAMPLE_FILE" ]; then
            echo "WARNING: SC sample file not found: $SC_SAMPLE_FILE"
            echo "Skipping Model: $MODEL, Dataset: $DATASET"
            echo ""
            continue
        fi
        
        # 출력 디렉토리
        OUTPUT_DIR="${RESULTS_DIR}/${MODEL}/${STAGES_CONFIG_SLUG}/dsc_cmp"
        OUTPUT_FILE="${OUTPUT_DIR}/${DATASET}_${MODEL_SAFE}_n_${N}_judge_window_${JUDGE_WINDOW_SIZE}_interplot_${INTERPLOT_SIZE}.json"
        
        # 이미 결과가 있으면, 해당 JSON을 사용해 eval_result.txt만 갱신
        if [ -f "$OUTPUT_FILE" ]; then
            echo "Result file already exists for Model: $MODEL, Dataset: $DATASET"
            
            # eval_result_all.txt 파일의 존재 여부와 "해당 데이터셋 섹션" 내 DSC Trials 존재 여부 확인
            # (파일 전체 grep 으로 체크하면 GSM8K에 DSC가 1번만 있어도 나머지 데이터셋이 전부 스킵되는 문제가 있음)
            EVAL_RESULT_ALL="${RESULTS_DIR}/${MODEL}/${STAGES_CONFIG_SLUG}/eval_result_all.txt"
            if [ -f "$EVAL_RESULT_ALL" ]; then
                python3 - "$EVAL_RESULT_ALL" "$DATASET" <<'PY'
import sys

path = sys.argv[1]
dataset = sys.argv[2]

header = f"## Dataset: {dataset}"
in_section = False
found = False

with open(path, "r", encoding="utf-8") as f:
    for line in f:
        if line.startswith("## Dataset:"):
            in_section = (line.strip() == header)
            continue
        if in_section and line.startswith("DSC Trials:"):
            found = True
            break

sys.exit(0 if found else 1)
PY
                if [ $? -eq 0 ]; then
                    echo "Skipping: DSC results already exist in eval_result_all.txt for $DATASET"
                    echo ""
                    continue
                fi
            fi
            
            echo "Updating eval_result_all.txt using existing DSC result JSON..."
            
            if [[ "$SINGLE_RUN" == "true" ]]; then
                export RSC_EVAL_SINGLE_OUTPUT_PATH="${RESULTS_DIR}/${MODEL}/${STAGES_CONFIG_SLUG}/eval_single_${BOOTSTRAP_SEED}.txt"
            fi
            python3 dsc/cmp_cal.py \
                --model_name "$MODEL" \
                --dataset_name "$DATASET" \
                --results_dir "$RESULTS_DIR" \
                --stages_config "$STAGES_CONFIG_NUM" \
                --reuse_result_json "$OUTPUT_FILE" \
                > "log/cmp_${MODEL_SAFE}_${DATASET}_reuse.log" 2>&1
            
            if [ $? -ne 0 ]; then
                echo "ERROR: Failed to update eval_result.txt for Model: $MODEL, Dataset: $DATASET (reuse mode)"
                echo "Check log/cmp_${MODEL_SAFE}_${DATASET}_reuse.log for details"
            else
                echo "SUCCESS: Updated eval_result_all.txt for Model: $MODEL, Dataset: $DATASET (reuse mode)"
            fi
            [[ "$SINGLE_RUN" == "true" ]] && unset RSC_EVAL_SINGLE_OUTPUT_PATH
            
            echo ""
            continue
        fi
        
        # 실행
        echo "Input SC samples: $SC_SAMPLE_FILE"
        echo "Input DSC samples: $DSC_SAMPLE_FILE"
        echo "Input eval result: $EVAL_FILE"
        echo "Output file: $OUTPUT_FILE"
        echo ""
        
        if [[ "$SINGLE_RUN" == "true" ]]; then
            export RSC_EVAL_SINGLE_OUTPUT_PATH="${RESULTS_DIR}/${MODEL}/${STAGES_CONFIG_SLUG}/eval_single_${BOOTSTRAP_SEED}.txt"
        fi
        python3 dsc/cmp_cal.py \
            --model_name "$MODEL" \
            --dataset_name "$DATASET" \
            --n "$N" \
            --subset_n "$N" \
            --judge_window_size "$JUDGE_WINDOW_SIZE" \
            --interplot_size "$INTERPLOT_SIZE" \
            --results_dir "$RESULTS_DIR" \
            --stages_config "$STAGES_CONFIG_NUM" \
            --num_trials "$NUM_TRIALS" \
            --bootstrap_seed "$BOOTSTRAP_SEED" \
            > "log/cmp_${MODEL_SAFE}_${DATASET}.log" 2>&1
        
        if [ $? -ne 0 ]; then
            echo "ERROR: Failed to run comparison for Model: $MODEL, Dataset: $DATASET"
            echo "Check log/cmp_${MODEL_SAFE}_${DATASET}.log for details"
        else
            echo "SUCCESS: Completed comparison for Model: $MODEL, Dataset: $DATASET"
            if [ -f "$OUTPUT_FILE" ]; then
                echo "Result saved to: $OUTPUT_FILE"
            fi
        fi
        [[ "$SINGLE_RUN" == "true" ]] && unset RSC_EVAL_SINGLE_OUTPUT_PATH
        
        echo ""
    done
done

echo "=========================================="
echo "All comparisons completed!"
echo "=========================================="

