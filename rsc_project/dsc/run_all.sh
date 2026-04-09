SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Starting Full Pipeline Execution"
echo "=========================================="
echo ""

# 1단계: 랭킹 평가
echo "=========================================="
echo "Step 1: Running Ranking Evaluation"
echo "=========================================="
echo ""

bash run_all_eval.sh

if [ $? -ne 0 ]; then
    echo "ERROR: Step 1 (Ranking Evaluation) failed!"
    exit 1
fi

echo ""
echo "Step 1 completed successfully!"
echo ""


# 2단계: 성능/비용 비교
echo "=========================================="
echo "Step 2: Running Performance/Cost Comparison"
echo "=========================================="
echo ""

bash run_all_cmp.sh

if [ $? -ne 0 ]; then
    echo "ERROR: Step 2 (Performance/Cost Comparison) failed!"
    exit 1
fi

echo ""
echo "Step 2 completed successfully!"
echo ""

echo "=========================================="
echo "Full Pipeline Completed Successfully!"
echo "=========================================="

