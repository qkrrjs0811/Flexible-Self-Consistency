"""
MMLU-Pro 카테고리별 집계 JSON 병합.

파일명: {stages_dir}/{dataset_name}_metrics_by_category.json
섹션: rsc (run_eval), dsc (cmp_cal), ours (evaluate_ours_method)
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

AGGREGATE_FILENAME = "MMLU-Pro_metrics_by_category.json"

# MMLU-Pro 서브샘플(1400) 및 전체 test(MMLU-Pro_full) 등 동일 로직을 쓰는 데이터셋 이름
MMLU_PRO_DATASET_NAMES = frozenset({"MMLU-Pro", "MMLU-Pro_full"})


def is_mmlu_pro_dataset(dataset_name: str) -> bool:
    return dataset_name in MMLU_PRO_DATASET_NAMES


def merge_mmlu_pro_category_section(
    eval_result_dir: str,
    section: str,
    payload: Dict[str, Any],
    *,
    dataset_name: str = "MMLU-Pro",
    extra_meta: Optional[Dict[str, Any]] = None,
) -> str:
    """
    eval_result_dir(예: .../stages-40) 아래에 집계 JSON을 읽고/쓰며
    payload를 sections[section]에 저장한다.

    Returns:
        저장된 파일의 절대 경로
    """
    os.makedirs(eval_result_dir, exist_ok=True)
    aggregate_filename = f"{dataset_name}_metrics_by_category.json"
    path = os.path.join(eval_result_dir, aggregate_filename)

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            data = {}
    else:
        data = {}

    data.setdefault("dataset", dataset_name)
    if extra_meta:
        meta = data.setdefault("meta", {})
        meta.update(extra_meta)

    sections = data.setdefault("sections", {})
    sections[section] = payload

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return os.path.abspath(path)
