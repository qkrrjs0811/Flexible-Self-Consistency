"""
FSC vs full SC@40 문제별 최종 답변 일치 여부 분석.

기존 40개 reasoning pool을 재사용하여 sample별로 SC@40과 FSC의 majority vote 결과를 비교합니다.
새로운 reasoning chain generation은 수행하지 않습니다.
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ablation import (
    MAX_BUDGET,
    _extract_reasoning_paths,
    clip_budget,
    get_ours_budgets,
    load_probe_ablation_data,
    verify_prefix_path_usage,
)
from evaluate_ours_method import (
    dataset_matches_selection,
    find_sc40_json,
    probe_to_eval_dataset_name,
)
from evaluation.metrics import extract_pool_size_from_stages_config, normalize_answer_eval

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

SC_BUDGET = 40
RESULT_TXT_FILENAME = "sc_fsc_transition_result.txt"
RESULT_JSON_FILENAME = "sc_fsc_transition_per_sample.json"

_PATTERN_ANSWER = re.compile(r"^Final Answer_(\d+)$")


def extract_answers_from_paths(reasoning_paths: List[Dict[str, Any]]) -> List[str]:
    """reasoning path entry 목록에서 Final Answer 값을 순서대로 추출."""
    answers: List[str] = []
    for path_entry in reasoning_paths:
        answer_val = None
        for key, value in path_entry.items():
            if _PATTERN_ANSWER.match(key):
                answer_val = str(value) if value is not None else None
                break
        if answer_val is not None:
            answers.append(answer_val)
    return answers


def majority_vote_answer(answers: List[str]) -> str:
    """정규화된 답변 기준 majority voting (evaluate_ours_method.py와 동일)."""
    if not answers:
        return ""
    normalized_answers = [normalize_answer_eval(ans) for ans in answers]
    return Counter(normalized_answers).most_common(1)[0][0]


def compute_prefix_majority(
    reasoning_paths: List[Dict[str, Any]],
    budget: int,
    pool_size: Optional[int],
    max_n: int = MAX_BUDGET,
) -> str:
    """pool 앞에서부터 budget개 chain에 대한 majority vote 결과(정규화 답)."""
    paths = reasoning_paths
    if pool_size is not None:
        paths = paths[:pool_size]
    subset = paths[: min(len(paths), max_n)]
    effective_budget = clip_budget(budget, max_n=max_n)
    prefix_paths = subset[: min(effective_budget, len(subset))]
    answers = extract_answers_from_paths(prefix_paths)
    return majority_vote_answer(answers)


def _ratio(count: int, total: int) -> float:
    return count / total if total > 0 else 0.0


def _category_counts(
    samples: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    total = len(samples)
    both_correct = sum(1 for s in samples if s["sc_correct"] and s["fsc_correct"])
    sc_wrong_fsc_correct = sum(
        1 for s in samples if (not s["sc_correct"]) and s["fsc_correct"]
    )
    sc_correct_fsc_wrong = sum(
        1 for s in samples if s["sc_correct"] and (not s["fsc_correct"])
    )
    both_wrong = sum(
        1 for s in samples if (not s["sc_correct"]) and (not s["fsc_correct"])
    )
    agreement = sum(1 for s in samples if s["sc_fsc_same_answer"])

    return {
        "total_samples": total,
        "answer_agreement_rate": _ratio(agreement, total),
        "both_correct": {"count": both_correct, "ratio": _ratio(both_correct, total)},
        "sc_wrong_fsc_correct": {
            "count": sc_wrong_fsc_correct,
            "ratio": _ratio(sc_wrong_fsc_correct, total),
        },
        "sc_correct_fsc_wrong": {
            "count": sc_correct_fsc_wrong,
            "ratio": _ratio(sc_correct_fsc_wrong, total),
        },
        "both_wrong": {"count": both_wrong, "ratio": _ratio(both_wrong, total)},
    }


def analyze_dataset(
    input_json_path: str,
    probe_samples: Dict[str, Dict[str, Any]],
    pool_size: Optional[int],
    max_n: int = MAX_BUDGET,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """단일 dataset에 대해 sample별 SC/FSC 비교 및 summary 계산."""
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sample_ids = [str(s.get("id", "")) for s in data]
    fsc_budgets = get_ours_budgets(sample_ids, probe_samples, max_n=max_n)
    sc_budgets = {sid: SC_BUDGET for sid in sample_ids}

    per_sample: List[Dict[str, Any]] = []

    for sample in data:
        sample_id = str(sample.get("id", ""))
        gold_answer = str(sample.get("answer", "")).strip()
        gold_norm = normalize_answer_eval(gold_answer)

        reasoning_paths = _extract_reasoning_paths(sample)
        sc_budget = clip_budget(sc_budgets.get(sample_id, SC_BUDGET), max_n=max_n)
        fsc_budget = clip_budget(fsc_budgets.get(sample_id, max_n), max_n=max_n)

        sc_answer = compute_prefix_majority(
            reasoning_paths, sc_budget, pool_size, max_n=max_n
        )
        fsc_answer = compute_prefix_majority(
            reasoning_paths, fsc_budget, pool_size, max_n=max_n
        )

        sc_correct = sc_answer == gold_norm
        fsc_correct = fsc_answer == gold_norm
        sc_fsc_same_answer = sc_answer == fsc_answer

        per_sample.append(
            {
                "sample_id": sample_id,
                "gold_answer": gold_answer,
                "sc_answer": sc_answer,
                "fsc_answer": fsc_answer,
                "sc_correct": sc_correct,
                "fsc_correct": fsc_correct,
                "sc_fsc_same_answer": sc_fsc_same_answer,
                "number_of_paths": fsc_budget,
            }
        )

    summary = _category_counts(per_sample)
    return summary, per_sample


def run_sanity_checks(
    input_json_path: str,
    probe_samples: Dict[str, Dict[str, Any]],
    pool_size: Optional[int],
    max_n: int = MAX_BUDGET,
) -> bool:
    """SC@40 / FSC budget 및 prefix path 사용 sanity check."""
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sample_ids = [str(s.get("id", "")) for s in data]
    fsc_budgets = get_ours_budgets(sample_ids, probe_samples, max_n=max_n)
    sc_budgets = {sid: SC_BUDGET for sid in sample_ids}
    ok = True

    for sid in sample_ids:
        sc_b = sc_budgets.get(sid)
        fsc_b = fsc_budgets.get(sid)
        if sc_b != SC_BUDGET:
            print(f"  [SANITY FAIL] SC@40 must use 40 chains for {sid}, got {sc_b}")
            ok = False
        if fsc_b is None or fsc_b < 1 or fsc_b > max_n:
            print(f"  [SANITY FAIL] FSC budget out of range for {sid}: {fsc_b}")
            ok = False

    if not verify_prefix_path_usage(input_json_path, sc_budgets, pool_size, max_n=max_n):
        print("  [SANITY FAIL] SC@40 prefix path usage verification failed")
        ok = False

    if not verify_prefix_path_usage(input_json_path, fsc_budgets, pool_size, max_n=max_n):
        print("  [SANITY FAIL] FSC prefix path usage verification failed")
        ok = False

    if ok:
        avg_fsc = sum(fsc_budgets.values()) / len(fsc_budgets) if fsc_budgets else 0.0
        print(
            f"  [SANITY OK] SC@40=40 chains, FSC budgets in [1,{max_n}], "
            f"avg_fsc_budget={avg_fsc:.2f}, n_samples={len(sample_ids)}"
        )
    return ok


def resolve_model_result_dir(
    results_dir: str,
    model_name: str,
    input_json: str,
    stages_config: str,
) -> str:
    """evaluate_ours_method.py와 동일한 경로 규칙으로 model 결과 디렉토리 결정."""
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

    return os.path.join(model_base_dir, stages_config)


def format_summary_section(
    dataset_name: str,
    json_path: str,
    summary: Dict[str, Any],
) -> str:
    """txt 요약 섹션 문자열 생성."""
    lines = [
        f"\n{'=' * 70}\n",
        f"## Dataset: {dataset_name}\n",
        f"{'=' * 70}\n",
        f"Loading JSON file: {json_path}\n",
        f"Total samples: {summary['total_samples']}\n\n",
        f"Answer Agreement Rate: {summary['answer_agreement_rate']:.4f}\n",
        f"Both Correct: {summary['both_correct']['count']} "
        f"({summary['both_correct']['ratio']:.4f})\n",
        f"SC Wrong -> FSC Correct: {summary['sc_wrong_fsc_correct']['count']} "
        f"({summary['sc_wrong_fsc_correct']['ratio']:.4f})\n",
        f"SC Correct -> FSC Wrong: {summary['sc_correct_fsc_wrong']['count']} "
        f"({summary['sc_correct_fsc_wrong']['ratio']:.4f})\n",
        f"Both Wrong: {summary['both_wrong']['count']} "
        f"({summary['both_wrong']['ratio']:.4f})\n",
    ]
    return "".join(lines)


def save_txt_summary(
    result_path: str,
    dataset_name: str,
    json_path: str,
    summary: Dict[str, Any],
) -> None:
    """dataset별 summary를 sc_fsc_transition_result.txt에 누적 저장."""
    section = format_summary_section(dataset_name, json_path, summary)
    dataset_header = f"## Dataset: {dataset_name}"

    if os.path.exists(result_path):
        with open(result_path, "r", encoding="utf-8") as f:
            content = f.read()
        if dataset_header in content:
            pattern = re.compile(
                rf"\n={{{70}}}\n## Dataset: {re.escape(dataset_name)}\n.*?(?=\n={{{70}}}\n## Dataset: |\Z)",
                re.DOTALL,
            )
            content = pattern.sub("", content)
        content = content.rstrip() + section
    else:
        content = section.lstrip("\n")

    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\n[{RESULT_TXT_FILENAME}] 저장: {result_path}")


def save_per_sample_json(
    json_path: str,
    all_results: Dict[str, Dict[str, Dict[str, Any]]],
) -> None:
    """model-dataset별 per-sample JSON 저장."""
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"[{RESULT_JSON_FILENAME}] 저장: {json_path}")


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
    parser = argparse.ArgumentParser(
        description="FSC vs SC@40 문제별 최종 답변 일치 여부 분석"
    )
    parser.add_argument(
        "--probe_dir",
        type=str,
        default="math",
        help="probe 디렉토리 경로 (기본값: math → probe/math)",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="experiments/results",
        help="results 디렉토리 경로",
    )
    parser.add_argument(
        "--stages_config",
        type=str,
        default="stages-40",
        help="stages 설정 (기본값: stages-40)",
    )
    parser.add_argument(
        "--p_value",
        type=str,
        default=DEFAULT_P_VALUE,
        help=f"처리할 p_value (기본값: {DEFAULT_P_VALUE})",
    )
    parser.add_argument("--model", type=str, default=None, help="특정 모델만 처리")
    parser.add_argument("--dataset", type=str, default=None, help="특정 데이터셋만 처리")
    parser.add_argument(
        "--max_n",
        type=int,
        default=MAX_BUDGET,
        help=f"최대 budget (기본값: {MAX_BUDGET})",
    )
    parser.add_argument(
        "--json_output",
        type=str,
        default=None,
        help=f"per-sample JSON 출력 경로 (기본값: {{results_dir}}/{RESULT_JSON_FILENAME})",
    )
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

    json_output_path = args.json_output
    if json_output_path is None:
        json_output_path = os.path.join(results_dir, RESULT_JSON_FILENAME)
    elif not os.path.isabs(json_output_path):
        json_output_path = os.path.join(project_root, json_output_path)

    print(f"probe 디렉토리에서 데이터 로드: {probe_dir}")
    probe_data = load_probe_ablation_data(probe_dir)
    if not probe_data:
        print("Error: probe 데이터를 찾을 수 없습니다")
        return

    selected_models = [args.model] if args.model else MODELS
    selected_datasets = DATASETS
    pool_size = extract_pool_size_from_stages_config(args.stages_config)

    all_results: Dict[str, Dict[str, Dict[str, Any]]] = {}
    if os.path.exists(json_output_path):
        try:
            with open(json_output_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if isinstance(existing, dict):
                all_results = existing
        except Exception as e:
            print(f"  경고: 기존 JSON 로드 실패, 새로 작성합니다: {e}")

    total_processed = 0
    total_failed = 0

    for model_name, p_map in probe_data.items():
        if selected_models is not None and not _model_matches(model_name, selected_models):
            continue

        print(f"\n{'=' * 70}")
        print(f"모델: {model_name}")
        print(f"{'=' * 70}")

        model_results: Dict[str, Dict[str, Any]] = all_results.setdefault(model_name, {})

        for p_value, ds_map in p_map.items():
            if args.p_value is not None and p_value != args.p_value:
                continue

            for dataset_name, probe_samples in ds_map.items():
                if not dataset_matches_selection(dataset_name, selected_datasets, args.dataset):
                    continue

                eval_label = probe_to_eval_dataset_name(dataset_name)
                print(f"\n  p 값: {p_value}")
                print(f"  데이터셋: {dataset_name}" + (f" -> {eval_label}" if eval_label != dataset_name else ""))

                input_json = find_sc40_json(
                    results_dir, model_name, dataset_name, args.stages_config
                )
                if input_json is None:
                    print(
                        f"  경고: SC(n=40) 파일을 찾을 수 없습니다 "
                        f"(모델: {model_name}, 데이터셋: {dataset_name})"
                    )
                    total_failed += 1
                    continue

                print(f"  입력 파일: {input_json}")

                if not run_sanity_checks(
                    input_json, probe_samples, pool_size, max_n=args.max_n
                ):
                    print("  경고: sanity check 실패 (계속 진행)")

                summary, samples = analyze_dataset(
                    input_json, probe_samples, pool_size, max_n=args.max_n
                )

                result_dir = resolve_model_result_dir(
                    results_dir, model_name, input_json, args.stages_config
                )
                txt_path = os.path.join(result_dir, RESULT_TXT_FILENAME)
                save_txt_summary(txt_path, eval_label, input_json, summary)

                model_results[eval_label] = {
                    "summary": summary,
                    "samples": samples,
                }
                total_processed += 1
                print("    ✓ 완료")

    save_per_sample_json(json_output_path, all_results)

    print(f"\n{'=' * 70}")
    print("FSC vs SC@40 transition 분석 완료!")
    print(f"  성공: {total_processed}개")
    print(f"  실패: {total_failed}개")
    print(f"{'=' * 70}")
    print("\n[참고] 새로운 reasoning chain generation은 수행하지 않습니다.")
    print("       기존 stages-40/Stage1 pool의 앞에서부터 budget개 경로만 재사용합니다.")


if __name__ == "__main__":
    main()
