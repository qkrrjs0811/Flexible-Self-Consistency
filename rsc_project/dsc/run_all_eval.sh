# 모델 목록 (전체 HF 모델 ID 사용)
# MODELS=("meta-llama/Llama-3.1-8B-Instruct")
MODELS=("Qwen/Qwen2.5-3B-Instruct" "Qwen/Qwen2.5-7B-Instruct" "Qwen/Qwen2.5-14B-Instruct" "google/gemma-3-4b-it")


# 데이터셋 목록
# DATASETS=("AIME_2025")
# MMLU-Pro_full은 RSC Stage1·dsc_eval 산출물 경로가 MMLU-Pro와 별도(MMLU-Pro_full.json)입니다.
DATASETS=("MATH500_test" "AMC23" "AIME_2024" "AIME_2025" "MMLU-Pro" "MMLU-Pro_full" "GPQA-Diamond")

# 설정
STRATEGY="random"
RANK_PROMPT="short"
BATCH_SIZE=8
ITERATION=5
N=1
TEMP=0
TOP_P=0.9
GPU_MEMORY_UTILIZATION=0.9
# STAGES_CONFIG를 환경변수로 설정 가능 (기본값: 40)
STAGES_CONFIG_RAW="${STAGES_CONFIG:-40}"
RESULTS_DIR="experiments/results"
EVAL_DIR="../eval"

# STAGES_CONFIG 정규화 ("40" 또는 "stages-40" 둘 다 허용)
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

# 결과 디렉토리 생성
mkdir -p log

# 각 모델과 데이터셋 조합에 대해 실행
for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        echo "=========================================="
        echo "Running evaluation for Model: $MODEL, Dataset: $DATASET"
        echo "=========================================="
        
        # 로그 파일용 안전한 모델명
        MODEL_SAFE=$(echo "$MODEL" | sed 's/\//_/g')
        
        # 출력 파일 경로: experiments/results/{model_name}/stages-{n}/dsc_eval/
        OUTPUT_FILE="${RESULTS_DIR}/${MODEL}/${STAGES_CONFIG_SLUG}/dsc_eval/${DATASET}_${MODEL_SAFE}_eval_strategy_${STRATEGY}_prompt_${RANK_PROMPT}_batchsize_${BATCH_SIZE}_iter_${ITERATION}_n_${N}_temp_${TEMP}.jsonl"
        
        if [ -f "$OUTPUT_FILE" ]; then
            echo "Result file already exists: $OUTPUT_FILE"
            echo "Skipping Model: $MODEL, Dataset: $DATASET"
            echo ""
            continue
        fi
        
        # Python 스크립트 실행
        echo "Running: python3 dsc/batch_hard_eval_dataset.py --model $MODEL --dataset $DATASET ..."
        python3 dsc/batch_hard_eval_dataset.py \
            --model "$MODEL" \
            --dataset "$DATASET" \
            --strategy "$STRATEGY" \
            --rank_prompt "$RANK_PROMPT" \
            --batch_size "$BATCH_SIZE" \
            --iteration "$ITERATION" \
            --n "$N" \
            --temp "$TEMP" \
            --top_p "$TOP_P" \
            --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
            --eval_dir "$EVAL_DIR" \
            --results_dir "$RESULTS_DIR" \
            --stages_config "$STAGES_CONFIG_NUM" \
            > "log/eval_${MODEL_SAFE}_${DATASET}.log" 2>&1
        
        if [ $? -ne 0 ]; then
            echo "ERROR: Failed to run evaluation for Model: $MODEL, Dataset: $DATASET"
            echo "Check log/eval_${MODEL_SAFE}_${DATASET}.log for details"
        else
            echo "SUCCESS: Completed evaluation for Model: $MODEL, Dataset: $DATASET"
            echo "Result saved to: $OUTPUT_FILE"
        fi
        
        echo ""
    done
done

echo "=========================================="
echo "All evaluations completed!"
echo "=========================================="

