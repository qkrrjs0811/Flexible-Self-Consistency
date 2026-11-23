"""
MATH_train.json: metadata.level(1–5)별로 SC 고유 답 개수 평균을
맞음(Correct) / 틀림(Incorrect)으로 나누어 막대 그래프로 시각화.

입력: --model 로 experiments/results/{model}/{stages}/Stage1/MATH_train.json 을 찾거나,
      --stage1_dir 로 Stage1 폴더를 직접 지정.

분석 규칙은 visualize_answer_diversity.py 의 analyze_sample 과 동일
(majority vote 후 정답과 비교, 고유 답은 정규화 전 원문 문자열 기준).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
current_dir = script_dir
rsc_project_dir = None
for _ in range(10):
    if os.path.basename(current_dir) == "rsc_project" and os.path.exists(
        os.path.join(current_dir, "evaluation")
    ):
        rsc_project_dir = current_dir
        break
    parent = os.path.dirname(current_dir)
    if parent == current_dir:
        break
    current_dir = parent

if rsc_project_dir is None:
    rsc_project_dir = os.path.dirname(script_dir)

if rsc_project_dir and os.path.exists(rsc_project_dir):
    if rsc_project_dir not in sys.path:
        sys.path.insert(0, rsc_project_dir)

if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from evaluation.metrics import extract_cot_data, normalize_answer_eval


def analyze_sample(sample: dict) -> dict:
    """
    sample 한 개에 대해:
      - SC majority 기준 정오 판정
      - 고유 답 개수(원문 문자열 기준)
    를 계산한다.
    """
    paths = sample.get("Reasoning Paths", [])
    answers, _ = extract_cot_data(paths)
    if not answers:
        return {
            "num_chains": 0,
            "is_correct": False,
            "diversity_metrics": {"num_unique_answers": 0},
        }

    normalized = [normalize_answer_eval(a) for a in answers]
    majority = Counter(normalized).most_common(1)[0][0] if normalized else ""
    gt_norm = normalize_answer_eval(sample.get("answer", ""))
    is_correct = majority == gt_norm

    # 고유 답 개수는 기존 주석 설명대로 정규화 전 원문 기준
    raw_unique = len(set(str(a).strip() for a in answers if a is not None))
    return {
        "num_chains": len(answers),
        "is_correct": bool(is_correct),
        "diversity_metrics": {"num_unique_answers": raw_unique},
    }


def _set_latex_paper_mpl_style() -> None:
    """논문 표(LaTeX Computer Modern)에 가깝게: serif 본문 + mathtext cm."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Latin Modern Roman",
                "Computer Modern Roman",
                "Times New Roman",
                "Nimbus Roman No9 L",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "cm",
            "mathtext.default": "regular",
            "axes.labelcolor": "#000000",
            "xtick.color": "#000000",
            "ytick.color": "#000000",
            "axes.labelsize": 21,
            "axes.titlesize": 22,
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 18,
            "axes.unicode_minus": False,
            "axes.linewidth": 0.9,
        }
    )


_set_latex_paper_mpl_style()

LEVELS = [1, 2, 3, 4, 5]
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def _parse_level(sample: dict) -> int | None:
    md = sample.get("metadata")
    if not isinstance(md, dict):
        return None
    lv = md.get("level")
    if lv is None:
        return None
    try:
        n = int(lv)
    except (TypeError, ValueError):
        return None
    if 1 <= n <= 5:
        return n
    return None


def aggregate_by_level(json_path: str) -> Tuple[Dict[int, dict], int, int]:
    """
    Returns:
        per_level: level -> { 'correct_mean', 'incorrect_mean', 'correct_n', 'incorrect_n' }
        n_parsed: samples with valid level + non-empty answers
        n_skipped: skipped (no level or empty CoT)
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    correct_vals: Dict[int, List[float]] = {lv: [] for lv in LEVELS}
    incorrect_vals: Dict[int, List[float]] = {lv: [] for lv in LEVELS}
    n_skipped = 0
    n_parsed = 0

    for sample in data:
        if not isinstance(sample, dict):
            n_skipped += 1
            continue
        lv = _parse_level(sample)
        if lv is None:
            n_skipped += 1
            continue

        result = analyze_sample(sample)
        nu = result["diversity_metrics"]["num_unique_answers"]
        if result["num_chains"] == 0:
            n_skipped += 1
            continue

        n_parsed += 1
        if result["is_correct"]:
            correct_vals[lv].append(float(nu))
        else:
            incorrect_vals[lv].append(float(nu))

    per_level: Dict[int, dict] = {}
    for lv in LEVELS:
        c = correct_vals[lv]
        inc = incorrect_vals[lv]
        per_level[lv] = {
            "correct_mean": float(np.mean(c)) if c else 0.0,
            "incorrect_mean": float(np.mean(inc)) if inc else 0.0,
            "correct_n": len(c),
            "incorrect_n": len(inc),
        }

    return per_level, n_parsed, n_skipped


def print_summary(per_level: Dict[int, dict]) -> None:
    print("\n" + "=" * 72)
    print("MATH level 별 평균 고유 답 개수 (sample 평균)")
    print("=" * 72)
    for lv in LEVELS:
        d = per_level[lv]
        print(
            f"  Level {lv}: Correct mean={d['correct_mean']:.3f} (n={d['correct_n']}), "
            f"Incorrect mean={d['incorrect_mean']:.3f} (n={d['incorrect_n']})"
        )
    print("=" * 72)


def plot_level_diversity(
    per_level: Dict[int, dict],
    output_path: str,
) -> None:
    n = len(LEVELS)
    x = np.arange(n, dtype=float)

    correct_y = [per_level[lv]["correct_mean"] for lv in LEVELS]
    incorrect_y = [per_level[lv]["incorrect_mean"] for lv in LEVELS]

    fig, ax = plt.subplots(1, 1, figsize=(11, 7.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # 첨부 예시 스타일과 유사한 선/마커 강조 표현
    correct_color = "#2f7fbd"
    incorrect_color = "#d65a4a"
    ax.plot(
        x,
        correct_y,
        label="Correct",
        color=correct_color,
        linewidth=6.0,
        marker="o",
        markersize=17,
        markerfacecolor=correct_color,
        markeredgecolor="white",
        markeredgewidth=2.8,
        zorder=3,
    )
    ax.plot(
        x,
        incorrect_y,
        label="Incorrect",
        color=incorrect_color,
        linewidth=6.0,
        marker="o",
        markersize=17,
        markerfacecolor=incorrect_color,
        markeredgecolor="white",
        markeredgewidth=2.8,
        zorder=3,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"Level {lv}" for lv in LEVELS])
    ax.tick_params(axis="x", which="major", labelsize=34, colors="#000000", pad=22, width=2.2, length=9)
    ax.tick_params(axis="y", which="major", labelsize=32, colors="#000000", pad=10, width=2.2, length=0)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=2,
        frameon=False,
        fontsize=42,
        handlelength=1.9,
        columnspacing=1.7,
    )
    ax.grid(axis="y", alpha=0.9, color="#c7c1b8", linestyle="-", linewidth=3.2)
    ax.grid(axis="x", visible=False)

    # 예시처럼 좌/하단 축만 남기고 굵게 표현
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#1f2f3f")
    ax.spines["bottom"].set_color("#1f2f3f")
    ax.spines["left"].set_linewidth(3.8)
    ax.spines["bottom"].set_linewidth(3.8)

    ymax_data = max(correct_y + incorrect_y) if (correct_y + incorrect_y) else 1.0
    ax.set_ylim(0, ymax_data + max(1.2, ymax_data * 0.22))

    # 각 점 위에 값 표시(색상별)
    offset = max(0.18, ymax_data * 0.06)
    for i, y in enumerate(correct_y):
        ax.text(
            x[i],
            y + offset,
            f"{y:.2f}",
            ha="center",
            va="bottom",
            fontsize=36,
            fontweight="bold",
            color=correct_color,
        )
    for i, y in enumerate(incorrect_y):
        ax.text(
            x[i],
            y + offset,
            f"{y:.2f}",
            ha="center",
            va="bottom",
            fontsize=36,
            fontweight="bold",
            color=incorrect_color,
        )

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"시각화 저장: {output_path}")
    plt.close()


def normalize_stages_config(value: str) -> str:
    v = (value or "").strip()
    if not v:
        return v
    if v.isdigit():
        return f"stages-{v}"
    return v


def model_to_filename_tag(model: str) -> str:
    """모델 문자열을 파일명에 안전한 태그로 변환."""
    s = (model or "").strip()
    if not s:
        return "unknown_model"
    s = s.replace("\\", "/").strip("/")
    s = s.replace("/", "_")
    safe_chars = []
    for ch in s:
        if ch.isalnum() or ch in {".", "_", "-"}:
            safe_chars.append(ch)
        else:
            safe_chars.append("_")
    tag = "".join(safe_chars).strip("_")
    return tag or "unknown_model"


def normalize_model_key(model: str) -> str:
    """
    모델 문자열 비교용 정규화 키.
    - 소문자화
    - 영숫자/._- 외 문자는 제거
    """
    s = (model or "").strip().replace("\\", "/")
    return "".join(
        ch for ch in s.lower() if ch.isalnum() or ch in {".", "_", "-", "/"}
    )


def find_math_train_for_model(
    model: str,
    stages_config: str,
    results_root: str,
) -> tuple[str | None, str | None]:
    """
    experiments/results 아래에서 MATH_train.json 경로를 찾는다.

    Returns:
        (path, err): 찾으면 (path, None), 실패 시 (None, 에러 메시지)
    """
    sc = normalize_stages_config(stages_config)
    if not sc:
        return None, "stages_config 가 비었습니다."

    model = model.strip().strip("/").replace("\\", "/")
    if not model:
        return None, "--model 이 비었습니다."

    # 1) results_root / model / stages-XX / Stage1 / MATH_train.json (model에 슬래시 포함 가능)
    cand = os.path.join(results_root, model, sc, "Stage1", "MATH_train.json")
    if os.path.isfile(cand):
        return cand, None

    # 2) 짧은 이름: .../results/**/<basename>/stages-XX/Stage1/MATH_train.json 이 유일한지 탐색
    target_leaf = os.path.basename(model)
    model_key = normalize_model_key(model)
    target_leaf_key = normalize_model_key(target_leaf)
    matches: List[str] = []
    if not os.path.isdir(results_root):
        return None, f"results 디렉터리 없음: {results_root}"

    for root, dirs, files in os.walk(results_root):
        if os.path.basename(root) != "Stage1" or "MATH_train.json" not in files:
            continue
        if os.path.basename(os.path.dirname(root)) != sc:
            continue
        model_dir = os.path.dirname(os.path.dirname(root))
        leaf = os.path.basename(model_dir)
        rel = os.path.relpath(model_dir, results_root).replace("\\", "/")
        leaf_key = normalize_model_key(leaf)
        rel_key = normalize_model_key(rel)
        if (
            leaf == target_leaf
            or rel == model
            or leaf_key == target_leaf_key
            or rel_key == model_key
        ):
            matches.append(os.path.join(root, "MATH_train.json"))

    if len(matches) == 1:
        return matches[0], None
    if len(matches) > 1:
        preview = "\n".join(matches[:8])
        more = f"\n... 외 {len(matches) - 8}개" if len(matches) > 8 else ""
        return (
            None,
            f"모델 이름이 여러 경로와 일치합니다. --model 에 전체 상대 경로를 주세요.\n{preview}{more}",
        )
    return (
        None,
        f"MATH_train.json 을 찾지 못했습니다. model={model!r}, stages={sc!r}, tried:\n  {cand}",
    )


def resolve_math_train_path(args: argparse.Namespace) -> tuple[str | None, str | None]:
    if args.stage1_dir:
        d = args.stage1_dir
        if not os.path.isabs(d) and rsc_project_dir:
            d = os.path.join(rsc_project_dir, d)
        cand = os.path.join(d, "MATH_train.json")
        if os.path.isfile(cand):
            return cand, None
        return None, f"파일 없음: {cand}"

    if not args.model:
        return (
            None,
            "--model 을 지정하거나 --stage1_dir 로 Stage1 폴더를 지정하세요.",
        )

    results_root = args.results_dir
    if not os.path.isabs(results_root) and rsc_project_dir:
        results_root = os.path.join(rsc_project_dir, results_root)
    return find_math_train_for_model(args.model, args.stages_config, results_root)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MATH_train.json: level 1–5 vs mean unique SC answers (Correct/Incorrect)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=(
            "experiments/results 기준 모델 경로 또는 폴더명. "
            f"기본값: {DEFAULT_MODEL}. "
            "예: Qwen/Qwen2.5-7B-Instruct 또는 Qwen2.5-7B-Instruct(유일할 때)"
        ),
    )
    parser.add_argument(
        "--stages_config",
        type=str,
        default="stages-40",
        help="stages 폴더명 (40, 100, stages-40, stages-100)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="experiments/results",
        help="rsc_project 기준 결과 루트 (--model 사용 시)",
    )
    parser.add_argument(
        "--stage1_dir",
        type=str,
        default=None,
        help="Stage1 디렉터리 직접 지정(우선). 내부의 MATH_train.json 사용",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="출력 PNG (기본: visualization/answer_diversity_math_by_level.png)",
    )
    args = parser.parse_args()
    args.stages_config = normalize_stages_config(args.stages_config)

    json_path, err = resolve_math_train_path(args)
    if not json_path:
        print(f"Error: {err or 'MATH_train.json 을 찾을 수 없습니다.'}", file=sys.stderr)
        sys.exit(1)

    print(f"MATH_train: {json_path}")
    per_level, n_parsed, n_skipped = aggregate_by_level(json_path)
    print(f"유효 샘플(level 1–5, CoT 있음): {n_parsed}, 스킵: {n_skipped}")

    print_summary(per_level)

    if args.output_file:
        out = (
            args.output_file
            if os.path.isabs(args.output_file)
            else os.path.abspath(args.output_file)
        )
    else:
        vis_dir = os.path.join(rsc_project_dir or script_dir, "visualization")
        os.makedirs(vis_dir, exist_ok=True)
        model_tag = model_to_filename_tag(args.model)
        script_name = os.path.splitext(os.path.basename(__file__))[0]
        out = os.path.join(vis_dir, script_name, f"{script_name}_{model_tag}.png")

    plot_level_diversity(per_level, out)
    print("완료.")


if __name__ == "__main__":
    main()
