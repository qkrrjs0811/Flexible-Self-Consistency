"""
FSC(Ours) ablation 실험 스크립트.

기존 40개 reasoning pool을 재사용하여 다음 3가지 ablation method를 평가합니다.
  1. FSC-Discrete
  2. Fixed-Budget-Matched
  3. Random-Budget-Matched-seed42

새로운 reasoning chain generation은 수행하지 않습니다.
"""

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluation.metrics import extract_pool_size_from_stages_config, normalize_answer_eval
from evaluate_ours_method import (
    dataset_matches_selection,
    evaluate_ours_json,
    find_sc40_json,
    probe_to_eval_dataset_name,
)

# ============================================================================
# 실험 설정: 사용할 모델 목록
# ============================================================================
# None이면 probe 폴더의 모든 모델을 처리합니다.
# 특정 모델만 지정하려면 리스트에 모델 이름을 추가하세요.
# 예: MODELS = ["Qwen2.5-7B-Inst", "gemma-3-4b-it"]
MODELS = ["Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct", "google/gemma-3-4b-it"]
# MODELS = ["meta-llama/Llama-3.1-8B-Instruct"]

# ============================================================================
# 실험 설정: 사용할 데이터셋 목록
# ============================================================================
# None이면 probe 폴더의 모든 데이터셋을 처리합니다.
# eval_result.txt 섹션명 기준 (run_all_cmp.sh DATASETS와 동일).
# probe JSON basename(MATH500 등)과 다른 경우 evaluate_ours_method.PROBE_TO_EVAL_DATASET 으로 자동 매핑됩니다.
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

OURS_FSC_METHOD = "Ours (FSC)"

ABLATION_METHODS = [
    "FSC-Discrete",
    "Fixed-Budget-Matched",
    "Random-Budget-Matched-seed42",
]

RESULT_METHODS = [OURS_FSC_METHOD] + ABLATION_METHODS

DEFAULT_RANDOM_SEED = 42
MAX_BUDGET = 40


def load_probe_ablation_data(
    probe_dir: str,
) -> Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]]:
    """
    probe 폴더에서 number_of_paths 및 entropy_pred 로드.

    Returns:
        {model_name: {p_value: {dataset_name: {sample_id: {"number_of_paths": int,
                                                             "entropy_pred": float}, ...}}}}
    """
    result: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]] = {}

    if not os.path.exists(probe_dir):
        raise FileNotFoundError(f"probe 디렉토리를 찾을 수 없습니다: {probe_dir}")

    for root, _dirs, files in os.walk(probe_dir):
        p_value = os.path.basename(root)
        if not p_value.startswith("p_"):
            continue

        model_dir = os.path.dirname(root)
        model_name = os.path.basename(model_dir)
        if not model_name:
            continue

        for filename in files:
            if not filename.endswith(".json"):
                continue
            dataset_name = os.path.splitext(filename)[0]
            json_path = os.path.join(root, filename)
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"  경고: {json_path} 로드 실패: {e}")
                continue

            if not isinstance(data, list):
                continue

            sample_dict: Dict[str, Dict[str, Any]] = {}
            for item in data:
                sample_id = str(item.get("id", ""))
                if not sample_id:
                    continue
                entry: Dict[str, Any] = {}
                if item.get("number_of_paths") is not None:
                    entry["number_of_paths"] = int(item["number_of_paths"])
                if item.get("entropy_pred") is not None:
                    entry["entropy_pred"] = float(item["entropy_pred"])
                if entry:
                    sample_dict[sample_id] = entry

            if sample_dict:
                result.setdefault(model_name, {}).setdefault(p_value, {})[dataset_name] = sample_dict
                print(
                    f"  로드: {model_name}/{p_value}/{dataset_name} "
                    f"({len(sample_dict)}개 샘플)"
                )

    return result


def clip_budget(budget: int, max_n: int = MAX_BUDGET) -> int:
    return max(1, min(max_n, int(budget)))


def get_ours_budgets(
    sample_ids: List[str],
    probe_samples: Dict[str, Dict[str, Any]],
    max_n: int = MAX_BUDGET,
) -> Dict[str, int]:
    """기존 Ours와 동일한 샘플별 budget (number_of_paths 없으면 max_n)."""
    budgets: Dict[str, int] = {}
    for sample_id in sample_ids:
        if sample_id in probe_samples and probe_samples[sample_id].get("number_of_paths") is not None:
            raw = int(probe_samples[sample_id]["number_of_paths"])
        else:
            raw = max_n
        budgets[sample_id] = clip_budget(raw, max_n=max_n)
    return budgets


def compute_fsc_discrete_budgets(
    sample_ids: List[str],
    probe_samples: Dict[str, Dict[str, Any]],
    max_n: int = MAX_BUDGET,
) -> Tuple[Dict[str, int], float]:
    """
    entropy_pred 기반 binary discrete budget.
    entropy_pred < mean -> 1, else -> 40
    """
    entropy_values = [
        float(probe_samples[sid]["entropy_pred"])
        for sid in sample_ids
        if sid in probe_samples and probe_samples[sid].get("entropy_pred") is not None
    ]
    mean_entropy = sum(entropy_values) / len(entropy_values) if entropy_values else 0.0

    budgets: Dict[str, int] = {}
    for sample_id in sample_ids:
        if sample_id in probe_samples and probe_samples[sample_id].get("entropy_pred") is not None:
            entropy_pred = float(probe_samples[sample_id]["entropy_pred"])
            raw = 1 if entropy_pred < mean_entropy else max_n
        else:
            raw = max_n
        budgets[sample_id] = clip_budget(raw, max_n=max_n)
    return budgets, mean_entropy


def compute_fixed_budget_matched(
    ours_budgets: Dict[str, int],
    max_n: int = MAX_BUDGET,
) -> Tuple[Dict[str, int], int]:
    """Ours 평균 budget을 round하여 모든 샘플에 동일 적용."""
    if not ours_budgets:
        fixed = max_n
    else:
        fixed = round(sum(ours_budgets.values()) / len(ours_budgets))
    fixed = clip_budget(fixed, max_n=max_n)
    return {sid: fixed for sid in ours_budgets}, fixed


def compute_random_budget_matched(
    sample_ids: List[str],
    ours_budgets: Dict[str, int],
    seed: int = DEFAULT_RANDOM_SEED,
    max_n: int = MAX_BUDGET,
) -> Dict[str, int]:
    """Ours budget multiset을 shuffle하여 샘플에 재할당."""
    budget_list = [clip_budget(ours_budgets[sid], max_n=max_n) for sid in sample_ids]
    shuffled = budget_list[:]
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    return {sid: shuffled[i] for i, sid in enumerate(sample_ids)}


def _extract_reasoning_paths(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    reasoning_paths = sample.get("Reasoning Paths", [])
    if reasoning_paths:
        return reasoning_paths

    paths: List[Dict[str, Any]] = []
    cot_count = 0
    while f"CoT_{cot_count}" in sample:
        path_entry = {
            f"CoT_{cot_count}": sample.get(f"CoT_{cot_count}", ""),
            f"Final Answer_{cot_count}": sample.get(f"Final Answer_{cot_count}", ""),
        }
        if f"Tokens_{cot_count}" in sample:
            path_entry[f"Tokens_{cot_count}"] = sample.get(f"Tokens_{cot_count}")
        if f"input_tokens_{cot_count}" in sample:
            path_entry[f"input_tokens_{cot_count}"] = sample.get(f"input_tokens_{cot_count}")
        paths.append(path_entry)
        cot_count += 1
    return paths


def verify_prefix_path_usage(
    input_json_path: str,
    budget_dict: Dict[str, int],
    pool_size: Optional[int],
    max_n: int = MAX_BUDGET,
) -> bool:
    """각 샘플이 pool 앞에서부터 budget개 경로만 사용하는지 spot-check."""
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pattern_answer = re.compile(r"^Final Answer_(\d+)$")
    pattern_tokens = re.compile(r"^Tokens_(\d+)$")

    for sample in data:
        sample_id = str(sample.get("id", ""))
        budget = clip_budget(budget_dict.get(sample_id, max_n), max_n=max_n)
        reasoning_paths = _extract_reasoning_paths(sample)
        if pool_size is not None:
            reasoning_paths = reasoning_paths[:pool_size]
        if not reasoning_paths:
            continue

        subset_paths = reasoning_paths[: min(len(reasoning_paths), max_n)]
        expected_paths = subset_paths[: min(budget, len(subset_paths))]

        answers = []
        tokens = []
        for path_entry in expected_paths:
            answer_val = None
            token_val = 0
            for key, value in path_entry.items():
                if pattern_answer.match(key):
                    answer_val = str(value) if value is not None else None
                elif pattern_tokens.match(key):
                    token_val = int(value) if value is not None else 0
            if answer_val is not None:
                answers.append(answer_val)
                tokens.append(token_val)

        if len(answers) != min(budget, len(subset_paths)):
            print(
                f"  [SANITY FAIL] prefix path usage: sample={sample_id}, "
                f"expected={min(budget, len(subset_paths))}, got={len(answers)}"
            )
            return False

    return True


def run_sanity_checks(
    method_name: str,
    sample_ids: List[str],
    ours_budgets: Dict[str, int],
    ablation_budgets: Dict[str, int],
    input_json_path: str,
    pool_size: Optional[int],
    fixed_budget: Optional[int] = None,
    random_seed: Optional[int] = None,
    max_n: int = MAX_BUDGET,
) -> bool:
    """ablation budget 할당 및 평가 전제 sanity check."""
    ok = True

    for sid in sample_ids:
        b = ablation_budgets.get(sid)
        if b is None or b < 1 or b > max_n:
            print(f"  [SANITY FAIL] {method_name}: budget out of range for {sid}: {b}")
            ok = False

    if method_name == "Fixed-Budget-Matched" and fixed_budget is not None:
        ours_mean = sum(ours_budgets.values()) / len(ours_budgets) if ours_budgets else 0.0
        ablation_mean = sum(ablation_budgets.values()) / len(ablation_budgets) if ablation_budgets else 0.0
        if abs(ablation_mean - ours_mean) > 0.51:
            print(
                f"  [SANITY FAIL] Fixed-Budget-Matched mean mismatch: "
                f"ours={ours_mean:.4f}, ablation={ablation_mean:.4f}, fixed={fixed_budget}"
            )
            ok = False

    if method_name.startswith("Random-Budget-Matched"):
        ours_list = sorted(ours_budgets[sid] for sid in sample_ids)
        ablation_list = sorted(ablation_budgets[sid] for sid in sample_ids)
        if ours_list != ablation_list:
            print(f"  [SANITY FAIL] Random-Budget-Matched multiset mismatch (seed={random_seed})")
            ok = False

    if not verify_prefix_path_usage(input_json_path, ablation_budgets, pool_size, max_n=max_n):
        print(f"  [SANITY FAIL] {method_name}: prefix path usage verification failed")
        ok = False

    if ok:
        avg_budget = sum(ablation_budgets.values()) / len(ablation_budgets) if ablation_budgets else 0.0
        print(
            f"  [SANITY OK] {method_name}: budgets in [1,{max_n}], "
            f"avg_budget={avg_budget:.2f}, n_samples={len(sample_ids)}"
        )
    return ok


def _is_ablation_metric_line(line: str) -> bool:
    if line.startswith(OURS_FSC_METHOD + ":"):
        return True
    if any(line.startswith(m + ":") for m in ABLATION_METHODS):
        return True
    return bool(re.match(r"^Random-Budget-Matched-seed\d+:", line))


def _ablation_metric_sort_key(line: str):
    for i, method in enumerate(RESULT_METHODS):
        if line.startswith(method + ":"):
            return i
    m = re.match(r"^Random-Budget-Matched-seed(\d+):", line)
    if m:
        return len(ABLATION_METHODS) + 1 + int(m.group(1))
    return 999


def save_ablation_result(
    ablation_result_path: str,
    dataset_name: str,
    json_path: str,
    metrics: Dict[str, Any],
    metric_name: str,
) -> None:
    """eval_result.txt와 유사한 형식으로 ablation_result.txt에 기록."""
    if os.path.exists(ablation_result_path):
        with open(ablation_result_path, "r", encoding="utf-8") as f:
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

        ablation_lines = []
        ablation_line_indices = []
        last_non_ablation_idx = section_start_idx
        for i in range(section_start_idx, section_end_idx):
            if _is_ablation_metric_line(lines[i]):
                ablation_lines.append(lines[i])
                ablation_line_indices.append(i)
            else:
                line_stripped = lines[i].strip()
                if (
                    line_stripped
                    and not line_stripped.startswith("=")
                    and not line_stripped.startswith("##")
                    and not line_stripped.startswith("Loading")
                ):
                    last_non_ablation_idx = i

        for idx in sorted(ablation_line_indices, reverse=True):
            lines.pop(idx)
            if idx <= last_non_ablation_idx:
                last_non_ablation_idx -= 1

        replaced = False
        for i, orig_line in enumerate(ablation_lines):
            if orig_line.startswith(f"{metric_name}:"):
                ablation_lines[i] = metric_line
                replaced = True
                break
        if not replaced:
            ablation_lines.append(metric_line)

        ablation_lines.sort(key=_ablation_metric_sort_key)
        insert_idx = last_non_ablation_idx + 1
        for ml in reversed(ablation_lines):
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

    os.makedirs(os.path.dirname(ablation_result_path), exist_ok=True)
    with open(ablation_result_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"\n[ablation_result.txt] 저장: {ablation_result_path}")


def resolve_ablation_result_path(results_dir: str, model_name: str, input_json: str, stages_config: str) -> str:
    """evaluate_ours_method.py와 동일한 경로 규칙으로 ablation_result.txt 경로 결정."""
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

    return os.path.join(model_base_dir, stages_config, "ablation_result.txt")


def process_ablation_dataset(
    results_dir: str,
    model_name: str,
    dataset_name: str,
    probe_samples: Dict[str, Dict[str, Any]],
    stages_config: str,
    p_value: str,
    random_seed: int = DEFAULT_RANDOM_SEED,
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

    ours_budgets = get_ours_budgets(sample_ids, probe_samples, max_n=max_n)
    fsc_discrete_budgets, mean_entropy = compute_fsc_discrete_budgets(
        sample_ids, probe_samples, max_n=max_n
    )
    fixed_budgets, fixed_budget = compute_fixed_budget_matched(ours_budgets, max_n=max_n)
    random_budgets = compute_random_budget_matched(
        sample_ids, ours_budgets, seed=random_seed, max_n=max_n
    )

    method_budgets = {
        OURS_FSC_METHOD: ours_budgets,
        "FSC-Discrete": fsc_discrete_budgets,
        "Fixed-Budget-Matched": fixed_budgets,
        f"Random-Budget-Matched-seed{random_seed}": random_budgets,
    }

    print(f"    Ours avg budget: {sum(ours_budgets.values()) / len(ours_budgets):.2f}")
    print(f"    FSC-Discrete mean entropy_pred: {mean_entropy:.4f}")
    print(f"    Fixed-Budget-Matched budget: {fixed_budget}")

    ablation_result_path = resolve_ablation_result_path(
        results_dir, model_name, input_json, stages_config
    )
    results: Dict[str, Any] = {}

    for method_name, budget_dict in method_budgets.items():
        print(f"\n    --- {method_name} ---")
        fixed_b = fixed_budget if method_name == "Fixed-Budget-Matched" else None
        rs = random_seed if method_name.startswith("Random-Budget-Matched") else None
        if method_name != OURS_FSC_METHOD:
            if not run_sanity_checks(
                method_name,
                sample_ids,
                ours_budgets,
                budget_dict,
                input_json,
                pool_size,
                fixed_budget=fixed_b,
                random_seed=rs,
                max_n=max_n,
            ):
                print(f"  경고: {method_name} sanity check 실패 (계속 진행)")
        else:
            if not verify_prefix_path_usage(input_json, budget_dict, pool_size, max_n=max_n):
                print(f"  경고: {OURS_FSC_METHOD} prefix path usage 검증 실패 (계속 진행)")
            else:
                avg_budget = sum(budget_dict.values()) / len(budget_dict) if budget_dict else 0.0
                print(
                    f"  [SANITY OK] {OURS_FSC_METHOD}: budgets in [1,{max_n}], "
                    f"avg_budget={avg_budget:.2f}, n_samples={len(sample_ids)}"
                )

        metrics = evaluate_ours_json(
            input_json,
            dataset_name,
            budget_dict,
            pool_size=pool_size,
            subset_n=max_n,
            num_trials=1,
            max_n=max_n,
        )
        save_ablation_result(
            ablation_result_path,
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
        "ablation_result_path": ablation_result_path,
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
    parser = argparse.ArgumentParser(description="FSC(Ours) ablation 실험")
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
    parser.add_argument("--random_seed", type=int, default=DEFAULT_RANDOM_SEED,
                        help=f"Random-Budget-Matched shuffle seed (기본값: {DEFAULT_RANDOM_SEED})")
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

                result = process_ablation_dataset(
                    results_dir,
                    model_name,
                    dataset_name,
                    probe_samples,
                    stages_config=args.stages_config,
                    p_value=p_value,
                    random_seed=args.random_seed,
                    max_n=args.max_n,
                )
                if result:
                    total_processed += 1
                    print("    ✓ 완료")
                else:
                    total_failed += 1
                    print("    ✗ 실패")

    print(f"\n{'=' * 70}")
    print("Ablation 실험 완료!")
    print(f"  성공: {total_processed}개")
    print(f"  실패: {total_failed}개")
    print(f"{'=' * 70}")
    print("\n[참고] 새로운 reasoning chain generation은 수행하지 않습니다.")
    print("       기존 stages-40/Stage1 pool의 앞에서부터 budget개 경로만 재사용합니다.")


if __name__ == "__main__":
    main()
