"""
Fixed-budget SC baseline 실험 스크립트.

기존 40개 reasoning pool을 재사용하여 다음 method를 평가합니다.
  1. SC@1, SC@3, SC@5, SC@10
  2. Fixed-Budget-Matched
  3. FSC

새로운 reasoning chain generation은 수행하지 않습니다.
"""

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ablation import (
    MAX_BUDGET,
    clip_budget,
    compute_fixed_budget_matched,
    get_ours_budgets,
    load_probe_ablation_data,
    verify_prefix_path_usage,
)
from evaluate_ours_method import (
    dataset_matches_selection,
    evaluate_ours_json,
    find_sc40_json,
    probe_to_eval_dataset_name,
)
from evaluation.metrics import extract_pool_size_from_stages_config

# ============================================================================
# 실험 설정: 사용할 모델 목록
# ============================================================================
MODELS = ["Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct", "google/gemma-3-4b-it"]
# MODELS = ["meta-llama/Llama-3.1-8B-Instruct"]

# ============================================================================
# 실험 설정: 사용할 데이터셋 목록
# ============================================================================
DATASETS = [
    "MATH500_test",
    "AMC23",
    "AIME_2024",
    "AIME_2025",
    "MMLU-Pro",
    "MMLU-Pro_full",
    "GPQA-Diamond",
]
# DATASETS = None  # probe의 모든 데이터셋

DEFAULT_P_VALUE = "p_1.0"

SC_BASELINE_KS = [1, 3, 5, 10]
FSC_METHOD = "FSC"
FIXED_BUDGET_METHOD = "Fixed-Budget-Matched"

RESULT_METHODS = [f"SC@{k}" for k in SC_BASELINE_KS] + [FIXED_BUDGET_METHOD, FSC_METHOD]

RESULT_FILENAME = "fixed_budget_baseline_result.txt"


def compute_sc_k_budgets(sample_ids: List[str], k: int, max_n: int = MAX_BUDGET) -> Dict[str, int]:
    """모든 sample에 동일한 fixed budget K 적용."""
    budget = clip_budget(k, max_n=max_n)
    return {sid: budget for sid in sample_ids}


def run_baseline_sanity_checks(
    method_name: str,
    sample_ids: List[str],
    fsc_budgets: Dict[str, int],
    method_budgets: Dict[str, int],
    input_json_path: str,
    pool_size: Optional[int],
    fixed_budget: Optional[int] = None,
    sc_k: Optional[int] = None,
    max_n: int = MAX_BUDGET,
) -> bool:
    """baseline budget 할당 및 평가 전제 sanity check."""
    ok = True

    for sid in sample_ids:
        b = method_budgets.get(sid)
        if b is None or b < 1 or b > max_n:
            print(f"  [SANITY FAIL] {method_name}: budget out of range for {sid}: {b}")
            ok = False

    if sc_k is not None:
        for sid in sample_ids:
            if method_budgets.get(sid) != sc_k:
                print(
                    f"  [SANITY FAIL] {method_name}: expected budget {sc_k} for {sid}, "
                    f"got {method_budgets.get(sid)}"
                )
                ok = False

    if method_name == FIXED_BUDGET_METHOD and fixed_budget is not None:
        fsc_mean = sum(fsc_budgets.values()) / len(fsc_budgets) if fsc_budgets else 0.0
        method_mean = sum(method_budgets.values()) / len(method_budgets) if method_budgets else 0.0
        if abs(method_mean - fsc_mean) > 0.51:
            print(
                f"  [SANITY FAIL] Fixed-Budget-Matched mean mismatch: "
                f"fsc={fsc_mean:.4f}, method={method_mean:.4f}, fixed={fixed_budget}"
            )
            ok = False

    if not verify_prefix_path_usage(input_json_path, method_budgets, pool_size, max_n=max_n):
        print(f"  [SANITY FAIL] {method_name}: prefix path usage verification failed")
        ok = False

    if ok:
        avg_budget = sum(method_budgets.values()) / len(method_budgets) if method_budgets else 0.0
        print(
            f"  [SANITY OK] {method_name}: budgets in [1,{max_n}], "
            f"avg_budget={avg_budget:.2f}, n_samples={len(sample_ids)}"
        )
    return ok


def _is_baseline_metric_line(line: str) -> bool:
    return any(line.startswith(m + ":") for m in RESULT_METHODS)


def _baseline_metric_sort_key(line: str):
    for i, method in enumerate(RESULT_METHODS):
        if line.startswith(method + ":"):
            return i
    return 999


def save_baseline_result(
    result_path: str,
    dataset_name: str,
    json_path: str,
    metrics: Dict[str, Any],
    metric_name: str,
) -> None:
    """eval_result.txt와 유사한 형식으로 fixed_budget_baseline_result.txt에 기록."""
    if os.path.exists(result_path):
        with open(result_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    else:
        lines = []

    metric_line = (
        f"{metric_name}: {metrics.get('ours_accuracy', 0.0):.4f} "
        f"(avg_chains={metrics.get('ours_avg_chains', 0.0):.2f}, "
        f"avg_input_tokens={metrics.get('ours_avg_input_tokens', 0.0):.2f}, "
        f"avg_output_tokens={metrics.get('ours_avg_output_tokens', 0.0):.2f})\n"
    )

    dataset_header = f"## Dataset: {dataset_name}"
    section_start_idx = -1
    section_end_idx = -1

    for i, line in enumerate(lines):
        if dataset_header in line:
            section_start_idx = i
            break

    if section_start_idx >= 0:
        for i in range(section_start_idx + 1, len(lines)):
            if lines[i].startswith("## Dataset:"):
                section_end_idx = i
                break
        if section_end_idx < 0:
            section_end_idx = len(lines)

        baseline_lines = []
        baseline_line_indices = []
        last_non_baseline_idx = section_start_idx
        for i in range(section_start_idx, section_end_idx):
            if _is_baseline_metric_line(lines[i]):
                baseline_lines.append(lines[i])
                baseline_line_indices.append(i)
            else:
                line_stripped = lines[i].strip()
                if (
                    line_stripped
                    and not line_stripped.startswith("=")
                    and not line_stripped.startswith("##")
                    and not line_stripped.startswith("Loading")
                ):
                    last_non_baseline_idx = i

        for idx in sorted(baseline_line_indices, reverse=True):
            lines.pop(idx)
            if idx <= last_non_baseline_idx:
                last_non_baseline_idx -= 1

        replaced = False
        for i, orig_line in enumerate(baseline_lines):
            if orig_line.startswith(f"{metric_name}:"):
                baseline_lines[i] = metric_line
                replaced = True
                break
        if not replaced:
            baseline_lines.append(metric_line)

        baseline_lines.sort(key=_baseline_metric_sort_key)
        insert_idx = last_non_baseline_idx + 1
        for ml in reversed(baseline_lines):
            lines.insert(insert_idx, ml)
        if insert_idx < len(lines) and lines[insert_idx].strip() != "":
            lines.insert(insert_idx, "\n")
    else:
        new_section = (
            f"\n{'=' * 70}\n"
            f"## Dataset: {dataset_name}\n"
            f"{'=' * 70}\n"
            f"Loading JSON file: {json_path}\n"
            f"Loaded {metrics.get('total_samples', 0)} samples\n\n"
            f"{metric_line}\n"
        )
        lines.extend(new_section.splitlines(keepends=True))

    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"\n[{RESULT_FILENAME}] 저장: {result_path}")


def resolve_baseline_result_path(results_dir: str, model_name: str, input_json: str, stages_config: str) -> str:
    """evaluate_ours_method.py와 동일한 경로 규칙으로 결과 파일 경로 결정."""
    input_parts = input_json.split(os.sep)
    try:
        results_dir_basename = os.path.basename(results_dir.rstrip(os.sep))
        if results_dir_basename in input_parts:
            results_idx = input_parts.index(results_dir_basename)
        else:
            results_idx = -1
            for i, part in enumerate(input_parts):
                if "results" in part.lower():
                    results_idx = i
                    break

        if results_idx >= 0:
            model_base_parts = []
            idx = results_idx + 1
            while idx < len(input_parts) and input_parts[idx] not in [stages_config, "Stage1", "Stage2"]:
                model_base_parts.append(input_parts[idx])
                idx += 1
            filtered_parts = [p for p in model_base_parts if p not in ["thinking", "non-thinking"]]
            if filtered_parts:
                actual_model_path = os.path.join(*filtered_parts)
                model_base_dir = os.path.join(results_dir, actual_model_path)
            else:
                model_base_dir = os.path.join(results_dir, model_name)
        else:
            model_base_dir = os.path.join(results_dir, model_name)
    except Exception:
        model_base_dir = os.path.join(results_dir, model_name)

    if "thinking" in input_parts or "non-thinking" in input_parts:
        mode_subdir = "thinking" if "thinking" in input_parts else "non-thinking"
        model_base_dir = os.path.join(model_base_dir, mode_subdir)

    return os.path.join(model_base_dir, stages_config, RESULT_FILENAME)


def process_baseline_dataset(
    results_dir: str,
    model_name: str,
    dataset_name: str,
    probe_samples: Dict[str, Dict[str, Any]],
    stages_config: str,
    p_value: str,
    max_n: int = MAX_BUDGET,
) -> Optional[Dict[str, Any]]:
    input_json = find_sc40_json(results_dir, model_name, dataset_name, stages_config)
    if input_json is None:
        print(f"  경고: SC(n=40) 파일을 찾을 수 없습니다 (모델: {model_name}, 데이터셋: {dataset_name})")
        return None

    print(f"  입력 파일: {input_json}")

    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    sample_ids = [str(s.get("id", "")) for s in data]
    pool_size = extract_pool_size_from_stages_config(stages_config)

    fsc_budgets = get_ours_budgets(sample_ids, probe_samples, max_n=max_n)
    fixed_budgets, fixed_budget = compute_fixed_budget_matched(fsc_budgets, max_n=max_n)

    method_budgets: Dict[str, Dict[str, int]] = {}
    for k in SC_BASELINE_KS:
        method_budgets[f"SC@{k}"] = compute_sc_k_budgets(sample_ids, k, max_n=max_n)
    method_budgets[FIXED_BUDGET_METHOD] = fixed_budgets
    method_budgets[FSC_METHOD] = fsc_budgets

    print(f"    FSC avg budget: {sum(fsc_budgets.values()) / len(fsc_budgets):.2f}")
    print(f"    Fixed-Budget-Matched budget: {fixed_budget}")

    result_path = resolve_baseline_result_path(results_dir, model_name, input_json, stages_config)
    results: Dict[str, Any] = {}

    for method_name, budget_dict in method_budgets.items():
        print(f"\n    --- {method_name} ---")
        sc_k = int(method_name.split("@")[1]) if method_name.startswith("SC@") else None
        fixed_b = fixed_budget if method_name == FIXED_BUDGET_METHOD else None

        if not run_baseline_sanity_checks(
            method_name,
            sample_ids,
            fsc_budgets,
            budget_dict,
            input_json,
            pool_size,
            fixed_budget=fixed_b,
            sc_k=sc_k,
            max_n=max_n,
        ):
            print(f"  경고: {method_name} sanity check 실패 (계속 진행)")

        metrics = evaluate_ours_json(
            input_json,
            dataset_name,
            budget_dict,
            pool_size=pool_size,
            subset_n=max_n,
            num_trials=1,
            max_n=max_n,
        )
        save_baseline_result(
            result_path,
            probe_to_eval_dataset_name(dataset_name),
            input_json,
            metrics,
            metric_name=method_name,
        )
        results[method_name] = metrics

    return {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "p_value": p_value,
        "result_path": result_path,
        "results": results,
    }


def _model_matches(model_name: str, selected_models: List[str]) -> bool:
    for selected_model in selected_models:
        if model_name == selected_model:
            return True
        if "gemma" in model_name.lower() and "gemma" in selected_model.lower():
            if model_name.replace("google/", "") == selected_model.replace("google/", ""):
                return True
        if "qwen" in model_name.lower() and "qwen" in selected_model.lower():
            if model_name.replace("Qwen/", "") == selected_model.replace("Qwen/", ""):
                return True
        if "llama" in model_name.lower() and "llama" in selected_model.lower():
            if model_name.replace("meta-llama/", "") == selected_model.replace("meta-llama/", ""):
                return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed-budget SC baseline 실험")
    parser.add_argument("--probe_dir", type=str, default="math",
                        help="probe 디렉토리 경로 (기본값: math → probe/math)")
    parser.add_argument("--results_dir", type=str, default="experiments/results",
                        help="results 디렉토리 경로")
    parser.add_argument("--stages_config", type=str, default="stages-40",
                        help="stages 설정 (기본값: stages-40)")
    parser.add_argument("--p_value", type=str, default=DEFAULT_P_VALUE,
                        help=f"처리할 p_value (기본값: {DEFAULT_P_VALUE})")
    parser.add_argument("--model", type=str, default=None,
                        help="특정 모델만 처리")
    parser.add_argument("--dataset", type=str, default=None,
                        help="특정 데이터셋만 처리")
    parser.add_argument("--max_n", type=int, default=MAX_BUDGET,
                        help=f"최대 budget (기본값: {MAX_BUDGET})")
    args = parser.parse_args()

    if args.stages_config.isdigit():
        args.stages_config = f"stages-{args.stages_config}"

    project_root = os.path.dirname(os.path.abspath(__file__))

    if not os.path.isabs(args.probe_dir):
        normalized = args.probe_dir.replace("\\", "/").strip("/")
        if normalized == "probe" or normalized.startswith("probe/"):
            probe_dir = os.path.join(project_root, args.probe_dir)
        else:
            probe_dir = os.path.join(project_root, "probe", args.probe_dir)
    else:
        probe_dir = args.probe_dir

    if not os.path.isabs(args.results_dir):
        results_dir = os.path.join(project_root, args.results_dir)
    else:
        results_dir = args.results_dir

    print(f"probe 디렉토리에서 데이터 로드: {probe_dir}")
    probe_data = load_probe_ablation_data(probe_dir)
    if not probe_data:
        print("Error: probe 데이터를 찾을 수 없습니다")
        return

    selected_models = [args.model] if args.model else MODELS
    selected_datasets = DATASETS
    total_processed = 0
    total_failed = 0

    for model_name, p_map in probe_data.items():
        if selected_models is not None and not _model_matches(model_name, selected_models):
            continue

        print(f"\n{'=' * 70}")
        print(f"모델: {model_name}")
        print(f"{'=' * 70}")

        for p_value, ds_map in p_map.items():
            if args.p_value is not None and p_value != args.p_value:
                continue

            for dataset_name, probe_samples in ds_map.items():
                if not dataset_matches_selection(dataset_name, selected_datasets, args.dataset):
                    continue

                eval_label = probe_to_eval_dataset_name(dataset_name)
                print(f"\n  p 값: {p_value}")
                print(f"  데이터셋: {dataset_name}" + (f" -> {eval_label}" if eval_label != dataset_name else ""))

                result = process_baseline_dataset(
                    results_dir,
                    model_name,
                    dataset_name,
                    probe_samples,
                    stages_config=args.stages_config,
                    p_value=p_value,
                    max_n=args.max_n,
                )
                if result:
                    total_processed += 1
                    print("    ✓ 완료")
                else:
                    total_failed += 1
                    print("    ✗ 실패")

    print(f"\n{'=' * 70}")
    print("Fixed-budget SC baseline 실험 완료!")
    print(f"  성공: {total_processed}개")
    print(f"  실패: {total_failed}개")
    print(f"{'=' * 70}")
    print("\n[참고] 새로운 reasoning chain generation은 수행하지 않습니다.")
    print("       기존 stages-40/Stage1 pool의 앞에서부터 budget개 경로만 재사용합니다.")


if __name__ == "__main__":
    main()
