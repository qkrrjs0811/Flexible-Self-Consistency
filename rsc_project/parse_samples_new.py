#!/usr/bin/env python3
"""
samples_new JSON의 CoT 텍스트에 run_all_experiments.sh 평가 파이프라인과 동일한
답변 추출·정규화를 적용하여 samples_new_parsing에 저장합니다.

사용 함수:
  - rsc.aggregator.SCAggregator.extract_answer()
  - evaluation.metrics.normalize_answer_eval()
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluation.metrics import normalize_answer_eval
from rsc.aggregator import SCAggregator

DEFAULT_INPUT_ROOT = SCRIPT_DIR / "samples_new"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "samples_new_parsing"

FILENAME_TO_DATASET = {
    "AIME_2024.json": "AIME_2024",
    "AIME_2025.json": "AIME_2025",
    "MATH500.json": "MATH500_test",
    "MATH.json": "MATH_train",
    "AMC23.json": "AMC23",
    "GPQA-Diamond.json": "GPQA-Diamond",
}


def dataset_name_from_path(json_path: Path) -> str:
    mapped = FILENAME_TO_DATASET.get(json_path.name)
    if mapped:
        return mapped
    stem = json_path.stem
    if stem == "MATH500":
        return "MATH500_test"
    return stem


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


def _find_cot_key(path_entry: Dict[str, Any]) -> Optional[str]:
    for key in path_entry:
        if re.match(r"^CoT_\d+$", key):
            return key
    return None


def _final_answer_key_for_cot(cot_key: str) -> str:
    idx = cot_key.split("_", 1)[1]
    return f"Final Answer_{idx}"


def parse_path_entry(path_entry: Dict[str, Any], aggregator: SCAggregator) -> Dict[str, Any]:
    """단일 reasoning path entry의 Final Answer를 CoT에서 재추출."""
    parsed_entry = deepcopy(path_entry)
    cot_key = _find_cot_key(parsed_entry)
    if not cot_key:
        return parsed_entry

    cot_text = parsed_entry.get(cot_key) or ""
    extracted = aggregator.extract_answer(cot_text)
    normalized = normalize_answer_eval(extracted) if extracted else ""
    parsed_entry[_final_answer_key_for_cot(cot_key)] = normalized
    return parsed_entry


def parse_sample(sample: Dict[str, Any], aggregator: SCAggregator) -> Dict[str, Any]:
    """샘플 하나를 파싱하여 Reasoning Paths 형식으로 반환."""
    parsed = deepcopy(sample)

    reasoning_paths = _extract_reasoning_paths(parsed)
    parsed_paths = [parse_path_entry(path_entry, aggregator) for path_entry in reasoning_paths]
    parsed["Reasoning Paths"] = parsed_paths

    # 평면 형식 키 제거 (run_eval 저장 형식과 동일)
    keys_to_remove = []
    for key in list(parsed.keys()):
        if re.match(r"^(CoT|Final Answer|Tokens|input_tokens)_\d+$", key):
            keys_to_remove.append(key)
    for key in keys_to_remove:
        parsed.pop(key, None)

    return parsed


def parse_json_file(input_path: Path, output_path: Path) -> Dict[str, int]:
    dataset_name = dataset_name_from_path(input_path)
    aggregator = SCAggregator(dataset_name=dataset_name)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {input_path}")

    parsed_data = []
    changed_answers = 0
    total_answers = 0

    for sample in data:
        original_paths = _extract_reasoning_paths(sample)
        parsed_sample = parse_sample(sample, aggregator)
        parsed_paths = parsed_sample.get("Reasoning Paths", [])

        for orig_entry, new_entry in zip(original_paths, parsed_paths):
            cot_key = _find_cot_key(orig_entry)
            if not cot_key:
                continue
            final_key = _final_answer_key_for_cot(cot_key)
            total_answers += 1
            old_val = str(orig_entry.get(final_key, ""))
            new_val = str(new_entry.get(final_key, ""))
            if old_val != new_val:
                changed_answers += 1

        parsed_data.append(parsed_sample)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(parsed_data, f, ensure_ascii=False, indent=2)

    return {
        "samples": len(parsed_data),
        "total_answers": total_answers,
        "changed_answers": changed_answers,
        "dataset_name": dataset_name,
    }


def collect_json_files(input_root: Path) -> List[Path]:
    return sorted(input_root.rglob("*.json"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="samples_new JSON에 실험 파이프라인과 동일한 답변 파싱 적용"
    )
    parser.add_argument(
        "--input_root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="입력 루트 디렉토리 (기본: samples_new)",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="출력 루트 디렉토리 (기본: samples_new_parsing)",
    )
    args = parser.parse_args()

    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()

    json_files = collect_json_files(input_root)
    if not json_files:
        print(f"[오류] JSON 파일을 찾을 수 없습니다: {input_root}")
        sys.exit(1)

    print(f"입력: {input_root}")
    print(f"출력: {output_root}")
    print(f"처리할 파일 수: {len(json_files)}")
    print("=" * 60)

    for input_path in json_files:
        rel_path = input_path.relative_to(input_root)
        output_path = output_root / rel_path
        print(f"[처리 중] {rel_path} ...", flush=True)
        stats = parse_json_file(input_path, output_path)
        print(
            f"  완료: dataset={stats['dataset_name']}, "
            f"samples={stats['samples']}, "
            f"answers={stats['total_answers']}, "
            f"changed={stats['changed_answers']} -> {output_path.relative_to(output_root)}"
        )

    print("=" * 60)
    print("모든 파싱이 완료되었습니다.")


if __name__ == "__main__":
    main()
