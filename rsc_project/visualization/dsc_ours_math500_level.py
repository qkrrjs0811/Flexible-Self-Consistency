#!/usr/bin/env python3
"""
DSC vs Ours(FSC) path 수 집계 — MATH500_test, AIME_2025 등 Stage1+probe 지원 데이터셋.

(1) DSC greedy split × level(또는 unknown) 교차표 + path 사용 요약
(2) Ours: probe/math/<모델>/p_<p>/{dataset}.json 의 number_of_paths (id join)

DSC: dsc_pre와 동일 — greedy allocate, allocate_window, DSC_pre() (cmp_cal 규칙).

Ours probe 파일명:
  - MATH500_test: MATH500_test.json 또는 MATH500.json
  - 그 외: {dataset_name}.json (예: AIME_2025.json)

사용:
  cd rsc_project
  # MATH500 per-sample
  python3 visualization/dsc_ours_math500_level.py --dataset_name MATH500_test --per_sample
  # AIME 2025 per-sample (4모델 일괄)
  bash visualization/run_per_sample_paths_aime2025.sh

결과 (기본 visualization/analysis_dsc_fsc):
  - {dataset}_{model}_dsc_pre_level.json
  - {dataset}_{model}_ours_level_paths.json
  - {dataset}_{model}_per_sample_paths.json  (--per_sample)
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import random
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from typing import Any, Dict, List, Optional

# 이 파일: rsc_project/visualization/… → 프로젝트 루트는 상위 디렉터리
_RSC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _RSC_ROOT not in sys.path:
    sys.path.insert(0, _RSC_ROOT)
_DSC_DIR = os.path.join(_RSC_ROOT, "dsc")
if _DSC_DIR not in sys.path:
    sys.path.insert(0, _DSC_DIR)

from evaluation.metrics import normalize_answer_eval  # noqa: E402
from rsc.aggregator import SCAggregator  # noqa: E402

from cmp_cal import (  # noqa: E402
    DSC_pre,
    allocate_window,
    clean_data,
    load_sc_samples_from_json,
    split_easy_hard,
    split_easy_hard_greedy,
)

_VIS_DIR = os.path.dirname(os.path.abspath(__file__))


def model_to_probe_slug(model: str) -> str:
    """HF id → probe 폴더명 (예: Qwen/Qwen2.5-7B-Instruct → Qwen2.5-7B-Instruct)."""
    if "/" not in model:
        return model
    return model.split("/", 1)[1]


def probe_p_dirname(p: float) -> str:
    """probe 하위 폴더 p_1.0, p_0.75 등."""
    if float(p).is_integer():
        return f"p_{int(float(p))}.0"
    text = f"{float(p):g}"
    if "." not in text:
        text = f"{float(p):.1f}".rstrip("0").rstrip(".")
    return f"p_{text}"


# probe/math 등: eval 데이터셋명은 MATH500_test인데 probe 파일만 MATH500.json 인 경우
_OURS_PROBE_JSON_BASENAMES: Dict[str, List[str]] = {
    "MATH500_test": ["MATH500_test.json", "MATH500.json"],
}


def _ours_probe_json_basenames(dataset_name: str) -> List[str]:
    if dataset_name in _OURS_PROBE_JSON_BASENAMES:
        return list(_OURS_PROBE_JSON_BASENAMES[dataset_name])
    return [f"{dataset_name}.json"]


def resolve_ours_probe_path(
    *,
    probe_dir: str,
    model_name: str,
    dataset_name: str,
    ours_p: float,
) -> tuple[Optional[str], List[str]]:
    """존재하는 probe JSON 경로를 반환. 없으면 (None, 시도한 경로들)."""
    slug = model_to_probe_slug(model_name)
    p_key = probe_p_dirname(ours_p)
    base = os.path.join(probe_dir, slug, p_key)
    tried: List[str] = []
    for basename in _ours_probe_json_basenames(dataset_name):
        path = os.path.join(base, basename)
        tried.append(os.path.abspath(path))
        if os.path.isfile(path):
            return path, tried
    return None, tried


def _problem_type_from_item(item: dict) -> Optional[str]:
    """AIME 등: metadata.problem_type (str 또는 list) → 단일 문자열."""
    md = item.get("metadata") or {}
    pt = md.get("problem_type")
    if pt is None:
        return None
    if isinstance(pt, list):
        parts = [str(x).strip() for x in pt if x is not None and str(x).strip()]
        return ", ".join(parts) if parts else None
    text = str(pt).strip()
    return text or None


def _clip_n_paths(number_of_paths: Optional[Any], max_n: int) -> Optional[int]:
    """evaluate_ours calculate_n_sc와 동일 클립."""
    if number_of_paths is None:
        return None
    return max(1, min(max_n, int(number_of_paths)))


def _level_str_sort_key(k: str):
    """JSON용 level 키 정렬: 1, 2, … 숫자 오름차순, 그다음 기타 문자열, 마지막 unknown."""
    if k == "unknown":
        return (2, 0, "")
    try:
        return (0, int(k), "")
    except ValueError:
        return (1, 0, k)


def _ordered_crosstab(ct: dict) -> dict:
    """easy → hard → 그 외 행 순, 각 행 안에서는 level 1..5 → unknown 등 정렬."""
    outers = []
    seen = set()
    for label in ("easy", "hard"):
        if label in ct:
            outers.append(label)
            seen.add(label)
    outers.extend(sorted(set(ct) - seen))

    out = {}
    for outer in outers:
        inner = ct[outer]
        keys = sorted(inner.keys(), key=_level_str_sort_key)
        out[outer] = {k: inner[k] for k in keys}
    return out


def _ordered_level_totals(level_totals: dict) -> dict:
    keys = sorted(level_totals.keys(), key=_level_str_sort_key)
    return {k: level_totals[k] for k in keys}


def _prepare_predict_batch_for_dsc(
    predict_batch: list,
    *,
    subset_n: int,
    sc_num: int,
    bootstrap_seed: int,
    trial: int,
) -> list:
    """cmp_cal 메인 루프와 동일: pool에서 subset 샘플링 후 앞에서 sc_num개."""
    local_subset_n = min(subset_n, len(predict_batch)) if predict_batch else 0
    if local_subset_n <= 0:
        return []
    if local_subset_n >= len(predict_batch):
        subset_predict_batch = predict_batch
    else:
        random.seed(bootstrap_seed + trial)
        subset_indices = random.sample(range(len(predict_batch)), local_subset_n)
        subset_predict_batch = [predict_batch[i] for i in subset_indices]
    local_sc_num = min(sc_num, len(subset_predict_batch))
    if local_sc_num <= 0:
        return []
    return subset_predict_batch[:local_sc_num]


def _dsc_pre_paths_used(
    predict_batch: list,
    *,
    initial_window_size: int,
    extend_window_size: int,
) -> int:
    """cmp_cal DSC_pre()와 동일한 사용 path 수 (len(dsc_pre_batch))."""
    if not predict_batch:
        return 0
    dummy_tokens = {"input": 0, "output": [0] * len(predict_batch)}
    dsc_pre_batch, _ = DSC_pre(
        predict_batch,
        predict_batch,
        dummy_tokens,
        dummy_tokens,
        -1,
        initial_window_size,
        extend_window_size=extend_window_size,
    )
    return len(dsc_pre_batch)


def _summarize_path_counts(counts: list) -> dict:
    if not counts:
        return {"n": 0, "mean": None, "min": None, "max": None, "histogram": {}}
    ct = Counter(counts)
    return {
        "n": len(counts),
        "mean": sum(counts) / len(counts),
        "min": min(counts),
        "max": max(counts),
        "histogram": {str(k): ct[k] for k in sorted(ct.keys())},
    }


def _dsc_paths_for_sample(
    allocate: int,
    predict_batch: list,
    initial_window_size: int,
    *,
    subset_n: int,
    sc_num: int,
    window_size: int,
    bootstrap_seed: int,
    trial: int,
) -> tuple[str, int]:
    """문항 하나: greedy easy/hard 라벨 + cmp_cal DSC_pre path 수."""
    if allocate == 1:
        lab = "easy"
    elif allocate == -1:
        lab = "hard"
    else:
        lab = f"other({allocate})"
    pb = _prepare_predict_batch_for_dsc(
        predict_batch,
        subset_n=subset_n,
        sc_num=sc_num,
        bootstrap_seed=bootstrap_seed,
        trial=trial,
    )
    n_paths = _dsc_pre_paths_used(
        pb,
        initial_window_size=initial_window_size,
        extend_window_size=window_size,
    )
    return lab, n_paths


def _build_paths_by_easy_hard_level(
    pre_batch: list,
    allocate_list_greedy: list,
    window_list_actual: list,
    level_by_idx: list,
    *,
    subset_n: int,
    sc_num: int,
    window_size: int,
    bootstrap_seed: int,
    trial: int,
) -> dict:
    """easy/hard × level 별 dsc_pre에서 쓴 path 수 요약."""
    bucket: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for i, a in enumerate(allocate_list_greedy):
        lab, n_paths = _dsc_paths_for_sample(
            a,
            pre_batch[i],
            window_list_actual[i],
            subset_n=subset_n,
            sc_num=sc_num,
            window_size=window_size,
            bootstrap_seed=bootstrap_seed,
            trial=trial,
        )
        lv = level_by_idx[i]
        key_lv = str(lv) if lv is not None else "unknown"
        bucket[lab][key_lv].append(n_paths)

    out = {}
    for lab in sorted(bucket.keys(), key=lambda x: (0, x) if x in ("easy", "hard") else (1, x)):
        inner = {}
        for lk in sorted(bucket[lab].keys(), key=_level_str_sort_key):
            inner[lk] = _summarize_path_counts(bucket[lab][lk])
        out[lab] = inner
    return out


def build_per_sample_paths(
    raw_array: list,
    pre_batch: list,
    allocate_list_greedy: list,
    window_list_actual: list,
    level_by_idx: list,
    *,
    subset_n: int,
    sc_num: int,
    window_size: int,
    bootstrap_seed: int,
    trial: int,
    ours_by_id: Optional[Dict[str, int]] = None,
    max_n: int = 40,
) -> List[dict]:
    """문항별 DSC(dsc_pre)/Ours path 수 (id로 join)."""
    rows: List[dict] = []
    for i, item in enumerate(raw_array):
        a = allocate_list_greedy[i]
        lab, dsc_paths = _dsc_paths_for_sample(
            a,
            pre_batch[i],
            window_list_actual[i],
            subset_n=subset_n,
            sc_num=sc_num,
            window_size=window_size,
            bootstrap_seed=bootstrap_seed,
            trial=trial,
        )
        lv = level_by_idx[i]
        sid = item.get("id")
        ours_paths: Optional[int] = None
        if sid and ours_by_id is not None and sid in ours_by_id:
            ours_paths = _clip_n_paths(ours_by_id[sid], max_n)
        row = {
            "id": sid,
            "level": lv,
            "dsc_easy_hard": lab,
            "dsc_paths": dsc_paths,
            "ours_paths": ours_paths,
        }
        pt = _problem_type_from_item(item)
        if pt is not None:
            row["problem_type"] = pt
        rows.append(row)
    return rows


def load_ours_paths_by_id(probe_path: str) -> Dict[str, int]:
    """probe JSON에서 sample id → number_of_paths."""
    with open(probe_path, encoding="utf-8") as f:
        rows: List[dict] = json.load(f)
    out: Dict[str, int] = {}
    for row in rows:
        sid = row.get("id")
        n = row.get("number_of_paths")
        if sid is not None and n is not None:
            out[str(sid)] = int(n)
    return out


def _build_pre_lists(data_list: list, dataset_name: str):
    aggregator = SCAggregator(dataset_name=dataset_name)
    pre = []
    pre_answer = []
    pre_batch = []
    pre_tokens = []
    greedy_pre = []
    greedy_pre_answer = []
    greedy_pre_batch = []
    greedy_pre_tokens = []

    for tem in data_list:
        pre.append(tem)
        pre_answer.append(
            normalize_answer_eval(tem.get("answer", tem.get("output", "")))
        )
        shu_batch = tem["completion"]
        cur_output = tem.get("output_tokens")
        if cur_output is not None and len(cur_output) == len(shu_batch):
            out_tok = deepcopy(cur_output)
        else:
            out_tok = [0] * len(shu_batch)
        pre_tokens.append({"input": tem.get("input_tokens", 0), "output": out_tok})

        predict_batch = [
            normalize_answer_eval(aggregator.extract_answer(item)) for item in shu_batch
        ]
        pre_batch.append(deepcopy(predict_batch))

        greedy_pre.append(tem)
        greedy_pre_answer.append(
            normalize_answer_eval(tem.get("answer", tem.get("output", "")))
        )
        greedy_pre_batch.append(deepcopy(predict_batch))
        greedy_pre_tokens.append(
            {"input": tem.get("input_tokens", 0), "output": deepcopy(out_tok)}
        )

    return (
        pre,
        pre_answer,
        pre_batch,
        pre_tokens,
        greedy_pre,
        greedy_pre_answer,
        greedy_pre_batch,
        greedy_pre_tokens,
    )


def _prepare_dsc_run(
    *,
    results_dir: str,
    model_name: str,
    stages_config: str,
    dataset_name: str,
    n: int,
    judge_window_size: int,
    window_size: int,
    interplot_size: int,
    bootstrap_seed: int,
    trial: int,
) -> dict:
    """Stage1 + dsc_eval 로드 후 greedy allocate까지 (문항별 분석용 상태 포함)."""
    model_safe = model_name.replace("/", "_")
    stage1_path = os.path.join(
        results_dir,
        model_name,
        f"stages-{stages_config}",
        "Stage1",
        f"{dataset_name}.json",
    )
    eval_path = os.path.join(
        results_dir,
        model_name,
        f"stages-{stages_config}",
        "dsc_eval",
        f"{dataset_name}_{model_safe}_eval_strategy_random_prompt_short_batchsize_8_iter_5_n_1_temp_0.jsonl",
    )

    if not os.path.isfile(stage1_path):
        raise FileNotFoundError(stage1_path)
    if not os.path.isfile(eval_path):
        raise FileNotFoundError(eval_path)

    with open(stage1_path, "r", encoding="utf-8") as f:
        raw_array = json.load(f)
    level_by_idx = [
        (item.get("metadata") or {}).get("level") for item in raw_array
    ]

    data_list = load_sc_samples_from_json(stage1_path)
    (
        pre,
        pre_answer,
        pre_batch,
        pre_tokens,
        _greedy_pre,
        _greedy_pre_answer,
        greedy_pre_batch,
        greedy_pre_tokens,
    ) = _build_pre_lists(data_list, dataset_name)

    with open(eval_path, "r", encoding="utf-8") as f:
        eval_lines = [json.loads(line) for line in f if line.strip()]
    eval_result = eval_lines[-1] if eval_lines else {}

    pre_batch, eval_result, pre, pre_answer, pre_tokens, greedy_pre_tokens, greedy_pre_batch = clean_data(
        pre_batch,
        eval_result,
        pre,
        pre_answer,
        pre_tokens,
        greedy_pre_tokens,
        greedy_pre_batch,
        n,
    )

    random.seed(bootstrap_seed + trial)
    with contextlib.redirect_stdout(io.StringIO()):
        split_easy_hard(
            eval_result, pre_batch, judge_window_size, window_size, n, pre_tokens
        )
        allocate_list_greedy = split_easy_hard_greedy(
            eval_result, pre_batch, judge_window_size, window_size, n
        )[-1]
        window_list_actual = allocate_window(
            eval_result, pre_batch, n, interplot_size=interplot_size, window_size=window_size
        )

    if len(allocate_list_greedy) != len(level_by_idx):
        raise RuntimeError(
            f"길이 불일치: dsc_pre allocate={len(allocate_list_greedy)}, levels={len(level_by_idx)}"
        )

    return {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "stages_config": stages_config,
        "stage1_path": stage1_path,
        "eval_path": eval_path,
        "raw_array": raw_array,
        "level_by_idx": level_by_idx,
        "pre_batch": pre_batch,
        "allocate_list_greedy": allocate_list_greedy,
        "window_list_actual": window_list_actual,
    }


def _summarize_dsc_prepared(
    prepared: dict,
    *,
    n: int,
    judge_window_size: int,
    window_size: int,
    interplot_size: int,
    bootstrap_seed: int,
    trial: int,
    subset_n: int,
) -> dict:
    raw_array = prepared["raw_array"]
    level_by_idx = prepared["level_by_idx"]
    pre_batch = prepared["pre_batch"]
    allocate_list_greedy = prepared["allocate_list_greedy"]
    window_list_actual = prepared["window_list_actual"]

    easy_hard_labels = []
    for a in allocate_list_greedy:
        if a == 1:
            easy_hard_labels.append("easy")
        elif a == -1:
            easy_hard_labels.append("hard")
        else:
            easy_hard_labels.append(f"other({a})")

    ct = defaultdict(lambda: defaultdict(int))
    for lab, lv in zip(easy_hard_labels, level_by_idx):
        key_lv = str(lv) if lv is not None else "unknown"
        ct[lab][key_lv] += 1

    easy_n = sum(1 for lab in easy_hard_labels if lab == "easy")
    hard_n = sum(1 for lab in easy_hard_labels if lab == "hard")
    level_totals_raw = {
        str(lv) if lv is not None else "unknown": level_by_idx.count(lv)
        for lv in sorted(set(level_by_idx), key=lambda x: (x is None, x or 0))
    }
    paths_summary = _build_paths_by_easy_hard_level(
        pre_batch,
        allocate_list_greedy,
        window_list_actual,
        level_by_idx,
        subset_n=subset_n,
        sc_num=n,
        window_size=window_size,
        bootstrap_seed=bootstrap_seed,
        trial=trial,
    )
    return {
        "model": prepared["model_name"],
        "dataset": prepared["dataset_name"],
        "stages_config": prepared["stages_config"],
        "n": n,
        "judge_window_size": judge_window_size,
        "window_size": window_size,
        "interplot_size": interplot_size,
        "bootstrap_seed": bootstrap_seed,
        "trial": trial,
        "subset_n": subset_n,
        "dsc_method": "dsc_pre",
        "easy_count": easy_n,
        "hard_count": hard_n,
        "crosstab": _ordered_crosstab({k: dict(v) for k, v in ct.items()}),
        "level_totals": _ordered_level_totals(level_totals_raw),
        "dsc_greedy_split_paths_used_by_easy_hard_level": paths_summary,
        "stage1_path": prepared["stage1_path"],
        "eval_path": prepared["eval_path"],
        "samples_total": len(raw_array),
    }


def run_one_model(
    *,
    results_dir: str,
    model_name: str,
    stages_config: str,
    dataset_name: str,
    n: int,
    judge_window_size: int,
    window_size: int,
    interplot_size: int,
    bootstrap_seed: int,
    trial: int,
    subset_n: int,
    return_prepared: bool = False,
):
    prepared = _prepare_dsc_run(
        results_dir=results_dir,
        model_name=model_name,
        stages_config=stages_config,
        dataset_name=dataset_name,
        n=n,
        judge_window_size=judge_window_size,
        window_size=window_size,
        interplot_size=interplot_size,
        bootstrap_seed=bootstrap_seed,
        trial=trial,
    )
    summary = _summarize_dsc_prepared(
        prepared,
        n=n,
        judge_window_size=judge_window_size,
        window_size=window_size,
        interplot_size=interplot_size,
        bootstrap_seed=bootstrap_seed,
        trial=trial,
        subset_n=subset_n,
    )
    if return_prepared:
        return summary, prepared
    return summary


def build_per_sample_output(
    prepared: dict,
    *,
    model_name: str,
    dataset_name: str,
    stages_config: str,
    n: int,
    judge_window_size: int,
    window_size: int,
    interplot_size: int,
    bootstrap_seed: int,
    trial: int,
    subset_n: int,
    ours_p: float,
    probe_file: Optional[str],
    ours_by_id: Optional[Dict[str, int]],
) -> dict:
    samples = build_per_sample_paths(
        prepared["raw_array"],
        prepared["pre_batch"],
        prepared["allocate_list_greedy"],
        prepared["window_list_actual"],
        prepared["level_by_idx"],
        subset_n=subset_n,
        sc_num=n,
        window_size=window_size,
        bootstrap_seed=bootstrap_seed,
        trial=trial,
        ours_by_id=ours_by_id,
        max_n=n,
    )
    return {
        "model": model_name,
        "dataset": dataset_name,
        "samples": samples,
    }


def run_ours_from_probe(
    *,
    probe_path: str,
    model_name: str,
    dataset_name: str,
    ours_p: float,
    max_n: int,
) -> dict:
    """Ours: probe JSON에서 level별 number_of_paths(클립) 요약."""
    with open(probe_path, encoding="utf-8") as f:
        rows: List[dict] = json.load(f)

    by_level: Dict[str, List[int]] = defaultdict(list)
    level_totals_raw: Dict[str, int] = defaultdict(int)
    missing_paths = 0

    for row in rows:
        meta = row.get("metadata") or {}
        lv = meta.get("level")
        key_lv = str(lv) if lv is not None else "unknown"
        level_totals_raw[key_lv] += 1
        clipped = _clip_n_paths(row.get("number_of_paths"), max_n)
        if clipped is None:
            missing_paths += 1
            continue
        by_level[key_lv].append(clipped)

    paths_by_level = {}
    for lk in sorted(by_level.keys(), key=_level_str_sort_key):
        paths_by_level[lk] = _summarize_path_counts(by_level[lk])

    return {
        "method": "Ours",
        "model": model_name,
        "dataset": dataset_name,
        "probe_p": ours_p,
        "max_n_clip": max_n,
        "probe_file": os.path.abspath(probe_path),
        "samples_total": len(rows),
        "samples_missing_number_of_paths": missing_paths,
        "paths_used_by_level": paths_by_level,
        "level_totals": _ordered_level_totals(dict(level_totals_raw)),
    }


def _default_output_dir() -> str:
    return os.path.join(_VIS_DIR, "analysis_dsc_fsc")


def _output_basename(dataset_name: str, model_safe: str) -> str:
    return f"{dataset_name}_{model_safe}_dsc_pre_level.json"


def _output_basename_ours(dataset_name: str, model_safe: str) -> str:
    return f"{dataset_name}_{model_safe}_ours_level_paths.json"


def _output_basename_per_sample(dataset_name: str, model_safe: str) -> str:
    return f"{dataset_name}_{model_safe}_per_sample_paths.json"


def main():
    ap = argparse.ArgumentParser(
        description="DSC vs Ours path 수 (MATH500_test, AIME_2025, …) + optional per-sample JSON"
    )
    ap.add_argument(
        "--results_dir",
        default=os.path.join(_RSC_ROOT, "experiments/results"),
        help="experiments/results 디렉터리",
    )
    ap.add_argument(
        "--output_dir",
        default=_default_output_dir(),
        help="모델별 JSON 저장 디렉터리 (기본: visualization/analysis_dsc_fsc)",
    )
    ap.add_argument(
        "--dataset_name",
        default="MATH500_test",
        help="Stage1/dsc_eval/probe basename (예: MATH500_test, AIME_2025, AIME_2024)",
    )
    ap.add_argument("--stages_config", default="40")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--judge_window_size", type=int, default=24)
    ap.add_argument("--window_size", type=int, default=4)
    ap.add_argument("--bootstrap_seed", type=int, default=137)
    ap.add_argument("--trial", type=int, default=0)
    ap.add_argument(
        "--subset_n",
        type=int,
        default=40,
        help="DSC path 집계 시 pool에서 뽑는 개수 (cmp_cal --subset_n, 기본 40)",
    )
    ap.add_argument(
        "--interplot_size",
        type=int,
        default=16,
        help="dsc_pre allocate_window interplot_size (run_all_cmp.sh 기본 16)",
    )
    ap.add_argument(
        "--probe_dir",
        default=os.path.join(_RSC_ROOT, "probe", "math"),
        help="Ours probe 루트 (기본: probe/math)",
    )
    ap.add_argument(
        "--ours_p",
        type=float,
        default=1.0,
        help="Ours probe 하위 디렉터리 p_<값> (기본 1.0 → p_1.0)",
    )
    ap.add_argument(
        "--skip_ours",
        action="store_true",
        help="Ours probe JSON 생성/저장 생략",
    )
    ap.add_argument(
        "--models",
        nargs="*",
        default=[
            "Qwen/Qwen2.5-3B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
            "Qwen/Qwen2.5-14B-Instruct",
            "google/gemma-3-4b-it",
        ],
    )
    ap.add_argument(
        "--print_json",
        action="store_true",
        help="저장 후 전체 결과를 표준 출력에도 인쇄",
    )
    ap.add_argument(
        "--per_sample",
        action="store_true",
        help="문항별 dsc_paths / ours_paths를 per_sample_paths.json으로 저장",
    )
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_out = []
    all_out_ours = []
    saved_paths = []

    for model in args.models:
        model_safe = model.replace("/", "_")
        out_name = _output_basename(args.dataset_name, model_safe)
        out_path = os.path.join(args.output_dir, out_name)
        prepared = None
        try:
            if args.per_sample:
                r, prepared = run_one_model(
                    results_dir=args.results_dir,
                    model_name=model,
                    stages_config=args.stages_config,
                    dataset_name=args.dataset_name,
                    n=args.n,
                    judge_window_size=args.judge_window_size,
                    window_size=args.window_size,
                    bootstrap_seed=args.bootstrap_seed,
                    trial=args.trial,
                    subset_n=args.subset_n,
                    interplot_size=args.interplot_size,
                    return_prepared=True,
                )
            else:
                r = run_one_model(
                    results_dir=args.results_dir,
                    model_name=model,
                    stages_config=args.stages_config,
                    dataset_name=args.dataset_name,
                    n=args.n,
                    judge_window_size=args.judge_window_size,
                    window_size=args.window_size,
                    interplot_size=args.interplot_size,
                    bootstrap_seed=args.bootstrap_seed,
                    trial=args.trial,
                    subset_n=args.subset_n,
                )
            r["output_file"] = os.path.abspath(out_path)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(r, f, indent=2, ensure_ascii=False)
            saved_paths.append(out_path)
            all_out.append(r)
        except (FileNotFoundError, RuntimeError) as e:
            err_obj = {
                "model": model,
                "dataset": args.dataset_name,
                "error": str(e),
                "output_file": os.path.abspath(out_path),
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(err_obj, f, indent=2, ensure_ascii=False)
            saved_paths.append(out_path)
            all_out.append(err_obj)
            prepared = None

        if args.per_sample and prepared is not None:
            per_name = _output_basename_per_sample(args.dataset_name, model_safe)
            per_path = os.path.join(args.output_dir, per_name)
            probe_path, _probe_tried = resolve_ours_probe_path(
                probe_dir=args.probe_dir,
                model_name=model,
                dataset_name=args.dataset_name,
                ours_p=args.ours_p,
            )
            ours_by_id = None
            probe_abs = None
            if probe_path and not args.skip_ours:
                try:
                    ours_by_id = load_ours_paths_by_id(probe_path)
                    probe_abs = os.path.abspath(probe_path)
                except (OSError, ValueError, TypeError) as e:
                    print(
                        f"  [per_sample] Ours probe 로드 실패 ({model}): {e}",
                        file=sys.stderr,
                    )
            try:
                per_obj = build_per_sample_output(
                    prepared,
                    model_name=model,
                    dataset_name=args.dataset_name,
                    stages_config=args.stages_config,
                    n=args.n,
                    judge_window_size=args.judge_window_size,
                    window_size=args.window_size,
                    interplot_size=args.interplot_size,
                    bootstrap_seed=args.bootstrap_seed,
                    trial=args.trial,
                    subset_n=args.subset_n,
                    ours_p=args.ours_p,
                    probe_file=probe_abs,
                    ours_by_id=ours_by_id,
                )
                per_obj["output_file"] = os.path.abspath(per_path)
                with open(per_path, "w", encoding="utf-8") as f:
                    json.dump(per_obj, f, indent=2, ensure_ascii=False)
                saved_paths.append(per_path)
            except (RuntimeError, ValueError, TypeError) as e:
                err_per = {
                    "model": model,
                    "dataset": args.dataset_name,
                    "error": str(e),
                    "output_file": os.path.abspath(per_path),
                }
                with open(per_path, "w", encoding="utf-8") as f:
                    json.dump(err_per, f, indent=2, ensure_ascii=False)
                saved_paths.append(per_path)

        if not args.skip_ours:
            ours_name = _output_basename_ours(args.dataset_name, model_safe)
            ours_out = os.path.join(args.output_dir, ours_name)
            probe_path, probe_tried = resolve_ours_probe_path(
                probe_dir=args.probe_dir,
                model_name=model,
                dataset_name=args.dataset_name,
                ours_p=args.ours_p,
            )
            try:
                if probe_path is None:
                    raise FileNotFoundError(
                        "probe JSON 없음 (시도 경로): " + "; ".join(probe_tried)
                    )
                ro = run_ours_from_probe(
                    probe_path=probe_path,
                    model_name=model,
                    dataset_name=args.dataset_name,
                    ours_p=args.ours_p,
                    max_n=args.n,
                )
                ro["output_file"] = os.path.abspath(ours_out)
                with open(ours_out, "w", encoding="utf-8") as f:
                    json.dump(ro, f, indent=2, ensure_ascii=False)
                saved_paths.append(ours_out)
                all_out_ours.append(ro)
            except (FileNotFoundError, OSError, ValueError, TypeError) as e:
                err_o = {
                    "method": "Ours",
                    "model": model,
                    "dataset": args.dataset_name,
                    "error": str(e),
                    "probe_paths_tried": probe_tried,
                    "output_file": os.path.abspath(ours_out),
                }
                with open(ours_out, "w", encoding="utf-8") as f:
                    json.dump(err_o, f, indent=2, ensure_ascii=False)
                saved_paths.append(ours_out)
                all_out_ours.append(err_o)

    print("Saved:", file=sys.stderr)
    for p in saved_paths:
        print(f"  {os.path.abspath(p)}", file=sys.stderr)

    if args.print_json:
        print(
            json.dumps({"dsc": all_out, "ours": all_out_ours}, indent=2, ensure_ascii=False)
        )


if __name__ == "__main__":
    main()
