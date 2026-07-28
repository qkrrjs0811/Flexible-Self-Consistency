"""
FSC 엔트로피–오라클 chain 수 상관 및 SC@40 vs FSC 전이 분석.

리뷰어 지적(출력 엔트로피가 marginal value의 proxy일 뿐)에 대응하기 위한 분석:
  (A) 오라클 필요 chain 수 (min_k_gold, min_k_full_sc)
  (B) entropy_pred / number_of_paths 와 오라클 k의 상관·정렬
  (C) SC@40 vs FSC per-sample 전이 (answer agreement)

probe는 p_1.0만 사용합니다. 기존 Stage1 pool(40 chains)을 재사용하며
새로운 reasoning chain generation은 수행하지 않습니다.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ablation import (
    MAX_BUDGET,
    _extract_reasoning_paths,
    clip_budget,
    get_ours_budgets,
)
from evaluate_ours_method import (
    DATASETS,
    MODELS,
    dataset_matches_selection,
    find_sc40_json,
    probe_to_eval_dataset_name,
)
from evaluation.metrics import extract_cot_data, normalize_answer_eval
from sc_fsc_transition_analysis import (
    SC_BUDGET,
    _category_counts,
    _model_matches,
    compute_prefix_majority,
    extract_answers_from_paths,
    majority_vote_answer,
)

DEFAULT_P_VALUE = "p_1.0"
RESULT_TXT_FILENAME = "fsc_entropy_oracle_analysis_result.txt"

_PREFIX_KS = [1, 3, 5, 10, 20, 40]
_PATTERN_ANSWER = re.compile(r"^Final Answer_(\d+)$")


def load_probe_with_uncertainty(
    probe_dir: str,
    p_value: str = DEFAULT_P_VALUE,
) -> Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]]:
    """probe에서 p_value 하나만 로드 (entropy_pred, number_of_paths, uncertainty_pred)."""
    result: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]] = {}
    if not os.path.exists(probe_dir):
        raise FileNotFoundError(f"probe 디렉토리를 찾을 수 없습니다: {probe_dir}")

    for root, _dirs, files in os.walk(probe_dir):
        if os.path.basename(root) != p_value:
            continue
        model_name = os.path.basename(os.path.dirname(root))
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
                if item.get("uncertainty_pred") is not None:
                    entry["uncertainty_pred"] = float(item["uncertainty_pred"])
                if entry:
                    sample_dict[sample_id] = entry
            if sample_dict:
                result.setdefault(model_name, {}).setdefault(p_value, {})[dataset_name] = sample_dict
                print(
                    f"  로드: {model_name}/{p_value}/{dataset_name} "
                    f"({len(sample_dict)}개 샘플)"
                )
    return result


def _majority_normalized(norm_seq: List[str]) -> str:
    if not norm_seq:
        return ""
    return Counter(norm_seq).most_common(1)[0][0]


def min_chains_for_full_sc_majority(sample: dict) -> Tuple[Optional[int], int]:
    """prefix majority가 SC@N majority와 일치하는 최소 k."""
    paths = sample.get("Reasoning Paths", [])
    answers, _ = extract_cot_data(paths)
    if not answers:
        return None, 0
    norm = [normalize_answer_eval(a) for a in answers]
    full_maj = _majority_normalized(norm)
    n = len(norm)
    for k in range(1, n + 1):
        if _majority_normalized(norm[:k]) == full_maj:
            return k, n
    return n, n


def min_chains_for_gold_match(sample: dict) -> Tuple[Optional[int], int]:
    """prefix majority가 정답과 일치하는 최소 k (없으면 N)."""
    paths = sample.get("Reasoning Paths", [])
    answers, _ = extract_cot_data(paths)
    if not answers:
        return None, 0
    norm = [normalize_answer_eval(a) for a in answers]
    n = len(norm)
    gt_norm = normalize_answer_eval(str(sample.get("answer", "")).strip())
    for k in range(1, n + 1):
        if _majority_normalized(norm[:k]) == gt_norm:
            return k, n
    return n, n


def _safe_corr(x: List[float], y: List[float]) -> Tuple[Optional[float], Optional[float]]:
    """(pearson_r, spearman_r). 분산이 0이면 None."""
    if len(x) < 2 or len(y) < 2 or len(x) != len(y):
        return None, None
    try:
        pr = pearsonr(x, y)
        sr = spearmanr(x, y)
        pearson_val = float(pr.statistic) if not math.isnan(pr.statistic) else None
        spearman_val = float(sr.statistic) if not math.isnan(sr.statistic) else None
        return pearson_val, spearman_val
    except Exception:
        return None, None


def _fmt_corr(r: Optional[float]) -> str:
    return f"{r:.4f}" if r is not None else "N/A"


def _fmt_float(v: Optional[float]) -> str:
    return f"{v:.4f}" if v is not None else "N/A"


def _summary_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    p25_idx = max(0, int(0.25 * (n - 1)))
    p75_idx = max(0, int(0.75 * (n - 1)))
    return {
        "mean": sum(sorted_vals) / n,
        "median": float(median(sorted_vals)),
        "p25": sorted_vals[p25_idx],
        "p75": sorted_vals[p75_idx],
        "min": sorted_vals[0],
        "max": sorted_vals[-1],
    }


def _prefix_accuracy_at_k(
    samples: List[dict],
    k: int,
    pool_size: Optional[int],
    max_n: int,
) -> float:
    correct = 0
    total = 0
    for sample in samples:
        sample_id = str(sample.get("id", ""))
        if not sample_id:
            continue
        gold_norm = normalize_answer_eval(str(sample.get("answer", "")).strip())
        reasoning_paths = _extract_reasoning_paths(sample)
        pred = compute_prefix_majority(reasoning_paths, k, pool_size, max_n=max_n)
        total += 1
        if pred == gold_norm:
            correct += 1
    return correct / total if total > 0 else 0.0


def analyze_dataset(
    input_json_path: str,
    probe_samples: Dict[str, Dict[str, Any]],
    pool_size: Optional[int],
    max_n: int = MAX_BUDGET,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """단일 dataset: 오라클·상관·SC/FSC 전이 통합 분석."""
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sample_ids = [str(s.get("id", "")) for s in data]
    fsc_budgets = get_ours_budgets(sample_ids, probe_samples, max_n=max_n)

    per_sample: List[Dict[str, Any]] = []
    entropy_vals: List[float] = []
    uncertainty_vals: List[float] = []
    min_k_for_uncertainty: List[float] = []
    paths_vals: List[float] = []
    min_k_gold_vals: List[float] = []
    min_k_sc_vals: List[float] = []
    abs_err_gold: List[float] = []
    abs_err_sc: List[float] = []

    for sample in data:
        sample_id = str(sample.get("id", ""))
        gold_answer = str(sample.get("answer", "")).strip()
        gold_norm = normalize_answer_eval(gold_answer)

        reasoning_paths = _extract_reasoning_paths(sample)
        min_k_gold, n_chains = min_chains_for_gold_match(sample)
        min_k_sc, _ = min_chains_for_full_sc_majority(sample)

        probe_entry = probe_samples.get(sample_id, {})
        entropy_pred = probe_entry.get("entropy_pred")
        uncertainty_pred = probe_entry.get("uncertainty_pred")
        number_of_paths = fsc_budgets.get(sample_id, max_n)

        sc_answer = compute_prefix_majority(
            reasoning_paths, SC_BUDGET, pool_size, max_n=max_n
        )
        fsc_answer = compute_prefix_majority(
            reasoning_paths, number_of_paths, pool_size, max_n=max_n
        )
        sc_correct = sc_answer == gold_norm
        fsc_correct = fsc_answer == gold_norm

        sc_at_1_paths = reasoning_paths[:1] if reasoning_paths else []
        sc_at_1_answers = extract_answers_from_paths(sc_at_1_paths)
        sc_at_1_answer = majority_vote_answer(sc_at_1_answers)
        sc_at_1_correct = sc_at_1_answer == gold_norm

        entry: Dict[str, Any] = {
            "sample_id": sample_id,
            "gold_answer": gold_answer,
            "min_k_gold": min_k_gold,
            "min_k_full_sc": min_k_sc,
            "n_chains": n_chains,
            "entropy_pred": entropy_pred,
            "uncertainty_pred": uncertainty_pred,
            "number_of_paths": number_of_paths,
            "sc_correct": sc_correct,
            "fsc_correct": fsc_correct,
            "sc_fsc_same_answer": sc_answer == fsc_answer,
            "sc_at_1_correct": sc_at_1_correct,
        }
        per_sample.append(entry)

        if min_k_gold is not None and entropy_pred is not None:
            entropy_vals.append(float(entropy_pred))
            min_k_gold_vals.append(float(min_k_gold))
            paths_vals.append(float(number_of_paths))
            abs_err_gold.append(abs(float(number_of_paths) - float(min_k_gold)))
            if uncertainty_pred is not None:
                uncertainty_vals.append(float(uncertainty_pred))
                min_k_for_uncertainty.append(float(min_k_gold))
        if min_k_sc is not None and entropy_pred is not None:
            min_k_sc_vals.append(float(min_k_sc))
            abs_err_sc.append(abs(float(number_of_paths) - float(min_k_sc)))

    # prefix k별 marginal accuracy
    prefix_acc: Dict[int, float] = {}
    for k in _PREFIX_KS:
        prefix_acc[k] = _prefix_accuracy_at_k(data, k, pool_size, max_n=max_n)

    sc_at_1_fail_rate = (
        sum(1 for s in per_sample if not s["sc_at_1_correct"]) / len(per_sample)
        if per_sample
        else 0.0
    )

    pearson_e_gold, spearman_e_gold = _safe_corr(entropy_vals, min_k_gold_vals)
    pearson_p_gold, spearman_p_gold = _safe_corr(paths_vals, min_k_gold_vals)
    pearson_e_sc, spearman_e_sc = _safe_corr(entropy_vals, min_k_sc_vals)
    pearson_p_sc, spearman_p_sc = _safe_corr(paths_vals, min_k_sc_vals)

    # uncertainty vs gold (optional)
    pearson_u_gold, spearman_u_gold = (
        _safe_corr(uncertainty_vals, min_k_for_uncertainty)
        if uncertainty_vals
        else (None, None)
    )

    n_sufficient_gold = sum(
        1
        for s in per_sample
        if s["min_k_gold"] is not None
        and s["number_of_paths"] >= int(s["min_k_gold"])
    )
    n_sufficient_sc = sum(
        1
        for s in per_sample
        if s["min_k_full_sc"] is not None
        and s["number_of_paths"] >= int(s["min_k_full_sc"])
    )
    n_with_oracle = sum(1 for s in per_sample if s["min_k_gold"] is not None)

    transition = _category_counts(
        [
            {
                "sc_correct": s["sc_correct"],
                "fsc_correct": s["fsc_correct"],
                "sc_fsc_same_answer": s["sc_fsc_same_answer"],
            }
            for s in per_sample
        ]
    )

    summary: Dict[str, Any] = {
        "total_samples": len(per_sample),
        "matched_probe_samples": sum(1 for sid in sample_ids if sid in probe_samples),
        "oracle_min_k_gold": _summary_stats(min_k_gold_vals),
        "oracle_min_k_full_sc": _summary_stats(min_k_sc_vals),
        "entropy_pred": _summary_stats(entropy_vals),
        "number_of_paths": _summary_stats(paths_vals),
        "correlation": {
            "pearson_entropy_vs_min_k_gold": pearson_e_gold,
            "spearman_entropy_vs_min_k_gold": spearman_e_gold,
            "pearson_paths_vs_min_k_gold": pearson_p_gold,
            "spearman_paths_vs_min_k_gold": spearman_p_gold,
            "pearson_entropy_vs_min_k_full_sc": pearson_e_sc,
            "spearman_entropy_vs_min_k_full_sc": spearman_e_sc,
            "pearson_paths_vs_min_k_full_sc": pearson_p_sc,
            "spearman_paths_vs_min_k_full_sc": spearman_p_sc,
            "pearson_uncertainty_vs_min_k_gold": pearson_u_gold,
            "spearman_uncertainty_vs_min_k_gold": spearman_u_gold,
        },
        "budget_alignment": {
            "mae_paths_vs_min_k_gold": (
                sum(abs_err_gold) / len(abs_err_gold) if abs_err_gold else None
            ),
            "mae_paths_vs_min_k_full_sc": (
                sum(abs_err_sc) / len(abs_err_sc) if abs_err_sc else None
            ),
            "fraction_paths_ge_min_k_gold": (
                n_sufficient_gold / n_with_oracle if n_with_oracle else None
            ),
            "fraction_paths_ge_min_k_full_sc": (
                n_sufficient_sc / n_with_oracle if n_with_oracle else None
            ),
        },
        "marginal_sc_accuracy": prefix_acc,
        "sc_at_1_failure_rate": sc_at_1_fail_rate,
        "sc_at_40_correct_rate": (
            sum(1 for s in per_sample if s["sc_correct"]) / len(per_sample)
            if per_sample
            else 0.0
        ),
        "transition": transition,
    }
    return summary, per_sample


def format_section(
    model_name: str,
    dataset_name: str,
    input_json_path: str,
    probe_p_value: str,
    summary: Dict[str, Any],
) -> str:
    """dataset 단위 txt 섹션."""
    lines: List[str] = [
        f"\n{'=' * 70}\n",
        f"## Model: {model_name} | Dataset: {dataset_name} | Probe: {probe_p_value}\n",
        f"{'=' * 70}\n",
        f"Stage1 JSON: {input_json_path}\n",
        f"Total samples: {summary['total_samples']}\n",
        f"Matched probe samples: {summary['matched_probe_samples']}\n\n",
    ]

    # (A) Oracle
    lines.append("--- (A) Oracle Required Chains ---\n")
    for key, label in [
        ("oracle_min_k_gold", "min_k_gold (prefix majority == gold)"),
        ("oracle_min_k_full_sc", "min_k_full_sc (prefix majority == SC@40 majority)"),
        ("entropy_pred", "entropy_pred (probe)"),
        ("number_of_paths", "number_of_paths (FSC budget)"),
    ]:
        stats = summary.get(key, {})
        if not stats:
            lines.append(f"{label}: N/A\n")
            continue
        lines.append(
            f"{label}: mean={stats['mean']:.2f}, median={stats['median']:.2f}, "
            f"p25={stats['p25']:.2f}, p75={stats['p75']:.2f}, "
            f"min={stats['min']:.2f}, max={stats['max']:.2f}\n"
        )
    lines.append(
        f"SC@40 correct rate: {summary['sc_at_40_correct_rate']:.4f}\n"
    )
    lines.append(
        f"SC@1 failure rate (single-sample wrong): {summary['sc_at_1_failure_rate']:.4f}\n\n"
    )

    # Marginal accuracy
    lines.append("Marginal SC accuracy by prefix k:\n")
    for k in _PREFIX_KS:
        acc = summary["marginal_sc_accuracy"].get(k, 0.0)
        lines.append(f"  k={k:2d}: {acc:.4f}\n")
    lines.append("\n")

    # (B) Correlation
    lines.append("--- (B) Entropy / FSC Budget vs Oracle ---\n")
    corr = summary["correlation"]
    align = summary["budget_alignment"]
    pairs = [
        ("entropy_pred", "min_k_gold", "pearson_entropy_vs_min_k_gold", "spearman_entropy_vs_min_k_gold"),
        ("number_of_paths", "min_k_gold", "pearson_paths_vs_min_k_gold", "spearman_paths_vs_min_k_gold"),
        ("entropy_pred", "min_k_full_sc", "pearson_entropy_vs_min_k_full_sc", "spearman_entropy_vs_min_k_full_sc"),
        ("number_of_paths", "min_k_full_sc", "pearson_paths_vs_min_k_full_sc", "spearman_paths_vs_min_k_full_sc"),
        ("uncertainty_pred", "min_k_gold", "pearson_uncertainty_vs_min_k_gold", "spearman_uncertainty_vs_min_k_gold"),
    ]
    for x_name, y_name, p_key, s_key in pairs:
        lines.append(
            f"Pearson({x_name}, {y_name}): {_fmt_corr(corr.get(p_key))}  |  "
            f"Spearman({x_name}, {y_name}): {_fmt_corr(corr.get(s_key))}\n"
        )
    lines.append(
        f"MAE(number_of_paths, min_k_gold): {_fmt_float(align.get('mae_paths_vs_min_k_gold'))}\n"
    )
    lines.append(
        f"MAE(number_of_paths, min_k_full_sc): {_fmt_float(align.get('mae_paths_vs_min_k_full_sc'))}\n"
    )
    lines.append(
        f"Fraction(number_of_paths >= min_k_gold): "
        f"{_fmt_float(align.get('fraction_paths_ge_min_k_gold'))}\n"
    )
    lines.append(
        f"Fraction(number_of_paths >= min_k_full_sc): "
        f"{_fmt_float(align.get('fraction_paths_ge_min_k_full_sc'))}\n\n"
    )

    # (C) Transition
    lines.append("--- (C) SC@40 vs FSC Transition ---\n")
    tr = summary["transition"]
    lines.append(f"Answer Agreement Rate: {tr['answer_agreement_rate']:.4f}\n")
    lines.append(
        f"Both Correct: {tr['both_correct']['count']} ({tr['both_correct']['ratio']:.4f})\n"
    )
    lines.append(
        f"SC Wrong -> FSC Correct: {tr['sc_wrong_fsc_correct']['count']} "
        f"({tr['sc_wrong_fsc_correct']['ratio']:.4f})\n"
    )
    lines.append(
        f"SC Correct -> FSC Wrong: {tr['sc_correct_fsc_wrong']['count']} "
        f"({tr['sc_correct_fsc_wrong']['ratio']:.4f})\n"
    )
    lines.append(
        f"Both Wrong: {tr['both_wrong']['count']} ({tr['both_wrong']['ratio']:.4f})\n"
    )
    return "".join(lines)


def save_txt_report(result_path: str, sections: List[str], header: str) -> None:
    os.makedirs(os.path.dirname(result_path) or ".", exist_ok=True)
    content = header + "".join(sections)
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\n[{RESULT_TXT_FILENAME}] 저장: {result_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FSC 엔트로피–오라클 chain 수 상관 및 SC/FSC 전이 분석 (probe p_1.0)"
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
        "--output",
        type=str,
        default=None,
        help=f"출력 txt 경로 (기본값: {{results_dir}}/{RESULT_TXT_FILENAME})",
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

    output_path = args.output
    if output_path is None:
        output_path = os.path.join(results_dir, RESULT_TXT_FILENAME)
    elif not os.path.isabs(output_path):
        output_path = os.path.join(project_root, output_path)

    from evaluation.metrics import extract_pool_size_from_stages_config

    pool_size = extract_pool_size_from_stages_config(args.stages_config)

    print(f"probe 디렉토리에서 데이터 로드: {probe_dir}")
    probe_data = load_probe_with_uncertainty(probe_dir, p_value=args.p_value)
    if not probe_data:
        print("Error: probe 데이터를 찾을 수 없습니다")
        return

    selected_models = [args.model] if args.model else MODELS
    selected_datasets = DATASETS

    header = (
        f"{'=' * 70}\n"
        f"FSC Entropy-Oracle Analysis\n"
        f"Probe: {args.p_value} | Stages: {args.stages_config}\n"
        f"{'=' * 70}\n"
    )

    sections: List[str] = []
    total_processed = 0
    total_failed = 0

    for model_name, p_map in probe_data.items():
        if selected_models is not None and not _model_matches(model_name, selected_models):
            continue

        print(f"\n{'=' * 70}")
        print(f"모델: {model_name}")
        print(f"{'=' * 70}")

        if args.p_value not in p_map:
            print(f"  경고: {args.p_value} 없음, 스킵")
            continue

        ds_map = p_map[args.p_value]

        for dataset_name, probe_samples in ds_map.items():
            if not dataset_matches_selection(dataset_name, selected_datasets, args.dataset):
                continue

            eval_label = probe_to_eval_dataset_name(dataset_name)
            print(f"\n  데이터셋: {dataset_name}" + (f" -> {eval_label}" if eval_label != dataset_name else ""))

            input_json = find_sc40_json(
                results_dir, model_name, dataset_name, args.stages_config
            )
            if input_json is None:
                print(
                    f"  경고: Stage1 JSON 없음 "
                    f"(모델: {model_name}, 데이터셋: {dataset_name})"
                )
                total_failed += 1
                continue

            print(f"  입력 파일: {input_json}")

            summary, _ = analyze_dataset(
                input_json, probe_samples, pool_size, max_n=args.max_n
            )
            section = format_section(
                model_name, eval_label, input_json, args.p_value, summary
            )
            sections.append(section)
            total_processed += 1
            print("    ✓ 완료")

    save_txt_report(output_path, sections, header)

    print(f"\n{'=' * 70}")
    print("FSC Entropy-Oracle 분석 완료!")
    print(f"  성공: {total_processed}개 dataset")
    print(f"  실패: {total_failed}개 dataset")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
