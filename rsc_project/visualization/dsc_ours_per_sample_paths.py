#!/usr/bin/env python3
"""
문항별 DSC vs Ours(FSC) chain 수 JSON 생성.

`dsc_ours_math500_level.py` 와 동일 파이프라인이며, 기본적으로 `--per_sample` 을 켭니다.

지원 데이터셋 (Stage1 + dsc_eval + probe/math 가 있으면 동일 패턴):
  - MATH500_test
  - AIME_2025, AIME_2024
  - AMC23, GSM8K, GPQA-Diamond, …

예:
  cd rsc_project
  python3 visualization/dsc_ours_per_sample_paths.py --dataset_name AIME_2025
  python3 visualization/dsc_ours_per_sample_paths.py --dataset_name MATH500_test \\
      --models Qwen/Qwen2.5-7B-Instruct
  bash visualization/run_per_sample_paths_aime2025.sh
"""
from __future__ import annotations

import sys

from dsc_ours_math500_level import main as _main


def main() -> None:
    if "--per_sample" not in sys.argv:
        sys.argv.append("--per_sample")
    _main()


if __name__ == "__main__":
    main()
