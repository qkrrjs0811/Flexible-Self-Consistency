"""
MATH_train.json 기반 결과를 한 figure(좌/우 2패널)에 합쳐서 시각화합니다.

좌측 패널:
  - `visualize_sc_consensus_chains_by_level.py`의 막대 플롯 로직을 재사용
  - Level 1~5 별 모델(min required chain) 평균을 모델별 막대로 표시

우측 패널:
  - `visualize_answer_diversity_math.py`의 Correct/Incorrect 선 플롯 로직을 재사용
  - Level 1~5 별 (정답과 일치하는지) Correct/Incorrect 그룹의 unique answer 평균을 표시

기본 동작:
  - 좌측: 전체 모델을 자동 탐색(stages-pattern), metric은 `--left-metric gold` 기본값
  - 우측: `--right-model` 1개 모델에 대해 unique answers를 계산

출력:
  - `--output-mode combined`(기본): 좌우를 한 PNG에 저장
  - `--output-mode split`: 좌측 패널·우측 패널을 각각 별도 PNG로 저장
  - `--output-mode both`: 위 둘을 모두 저장
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sys
from collections import Counter
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


script_dir = os.path.dirname(os.path.abspath(__file__))
current_dir = script_dir
rsc_project_dir = None
for _ in range(10):
    if os.path.basename(current_dir) == "rsc_project" and os.path.exists(
        os.path.join(current_dir, "evaluation"),
    ):
        rsc_project_dir = current_dir
        break
    parent = os.path.dirname(current_dir)
    if parent == current_dir:
        break
    current_dir = parent

if rsc_project_dir is None:
    rsc_project_dir = "/home/albert0811/dev_env/git/repos/RASC/rsc_project"
    if not os.path.exists(rsc_project_dir):
        rsc_project_dir = os.path.abspath(
            os.path.join(os.path.dirname(script_dir), "..", "rsc_project"),
        )

if rsc_project_dir and os.path.exists(rsc_project_dir):
    if rsc_project_dir not in sys.path:
        sys.path.insert(0, rsc_project_dir)

if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from evaluation.metrics import extract_cot_data, normalize_answer_eval


LEVELS = [1, 2, 3, 4, 5]

# combined figure에서 패널 가로 너비 비율 (분리 출력 시 같은 비율로 figure 너비를 나눔)
COMBINED_WIDTH_RATIOS_LEFT = 1.2
COMBINED_WIDTH_RATIOS_RIGHT = 0.8

# 좌측 패널 figure 내 모델 막대 순서 (rel 경로에 부분 문자열 포함 여부로 매칭)
MODEL_PLOT_ORDER_SUBSTR: List[str] = [
    "Qwen2.5-3B-Instruct",
    "Qwen2.5-7B-Instruct",
    "Qwen2.5-14B-Instruct",
    "gemma-3-4b",
]

# 모델 막대 파스텔 팔레트 (반복 사용)
PASTEL_COLORS: List[str] = [
    "#5DA5DA",  # muted blue
    "#F17CB0",  # muted pink
    "#B291C7",  # muted purple
    "#F5A45D",  # muted orange
    "#60BD68",  # muted green
    "#B276B2",  # medium violet
    "#DECF3F",  # mustard yellow
    "#4D4D4D",  # dark gray
]


def _skip_model_rel(rel: str) -> bool:
    """Llama 계열(meta-llama 등)은 시각화에서 제외."""
    return "llama" in rel.replace("\\", "/").lower()


def _set_latex_paper_mpl_style() -> None:
    # 좌측/우측 모두 serif 기반으로 통일(원래 스크립트들의 공통 지향)
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
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#2d3a4b",
            "axes.labelcolor": "#000000",
            "xtick.color": "#000000",
            "ytick.color": "#000000",
            "grid.color": "#c7c1b8",
            "grid.linewidth": 1.0,
            "axes.grid": False,
            "axes.labelsize": 18,
            "axes.titlesize": 19,
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 15,
            "axes.unicode_minus": False,
            "axes.linewidth": 2.2,
        }
    )


_set_latex_paper_mpl_style()


def _panel_fig_sizes_for_split(
    combined_fig_w: float,
    combined_fig_h: float,
) -> tuple[float, float]:
    ratio_sum = COMBINED_WIDTH_RATIOS_LEFT + COMBINED_WIDTH_RATIOS_RIGHT
    w_left = combined_fig_w * (COMBINED_WIDTH_RATIOS_LEFT / ratio_sum)
    w_right = combined_fig_w * (COMBINED_WIDTH_RATIOS_RIGHT / ratio_sum)
    return w_left, w_right


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


def _majority_normalized(norm_seq: List[str]) -> str:
    if not norm_seq:
        return ""
    return Counter(norm_seq).most_common(1)[0][0]


def min_chains_for_full_sc_majority(sample: dict) -> Tuple[int | None, int]:
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


def min_chains_for_gold_match(sample: dict) -> Tuple[int | None, int]:
    paths = sample.get("Reasoning Paths", [])
    answers, _ = extract_cot_data(paths)
    if not answers:
        return None, 0
    norm = [normalize_answer_eval(a) for a in answers]
    n = len(norm)
    gt = str(sample.get("answer", "")).strip()
    gt_norm = normalize_answer_eval(gt)
    for k in range(1, n + 1):
        if _majority_normalized(norm[:k]) == gt_norm:
            return k, n
    return n, n


def discover_math_train_paths(results_root: str, stages_pattern: str) -> List[Tuple[str, str]]:
    """
    Returns:
        [(model_rel_path, json_path), ...]
    """
    out: List[Tuple[str, str]] = []
    if not os.path.isdir(results_root):
        return out
    for root, _dirs, files in os.walk(results_root):
        if os.path.basename(root) != "Stage1" or "MATH_train.json" not in files:
            continue
        stages_dir = os.path.basename(os.path.dirname(root))
        if not fnmatch.fnmatch(stages_dir, stages_pattern):
            continue
        model_dir = os.path.dirname(os.path.dirname(root))
        rel = os.path.relpath(model_dir, results_root).replace("\\", "/")
        out.append((rel, os.path.join(root, "MATH_train.json")))
    out.sort(key=lambda x: x[0])
    return out


def aggregate_file_both_metrics(json_path: str) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """
    Returns:
        by_level_full_sc, by_level_gold
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    by_sc: Dict[int, List[int]] = {lv: [] for lv in LEVELS}
    by_gold: Dict[int, List[int]] = {lv: [] for lv in LEVELS}

    for sample in data:
        if not isinstance(sample, dict):
            continue
        lv = _parse_level(sample)
        if lv is None:
            continue
        k_sc, nc = min_chains_for_full_sc_majority(sample)
        k_gold, nc2 = min_chains_for_gold_match(sample)
        if k_sc is None or nc == 0 or k_gold is None or nc2 == 0:
            continue
        by_sc[lv].append(k_sc)
        by_gold[lv].append(k_gold)

    return by_sc, by_gold


def _summarize(vals: List[int]) -> dict:
    if not vals:
        return {"mean": float("nan"), "n": 0}
    a = np.array(vals, dtype=float)
    return {"mean": float(np.mean(a)), "n": len(vals)}


# 모델 범례 표시 이름 (rel 경로 basename → 표시용)
MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "Qwen2.5-3B-Instruct": "Qwen2.5-3B",
    "Qwen2.5-7B-Instruct": "Qwen2.5-7B",
    "Qwen2.5-14B-Instruct": "Qwen2.5-14B",
    "gemma-3-4b-it": "Gemma-3-4B",
}


def _short_model_label(rel_path: str) -> str:
    base = os.path.basename(rel_path.rstrip("/"))
    return MODEL_DISPLAY_NAMES.get(base, base)


def ordered_model_rel_paths(rel_paths: Iterable[str]) -> List[str]:
    def sort_key(rel: str) -> Tuple[int, str]:
        r = rel.replace("\\", "/")
        for i, sub in enumerate(MODEL_PLOT_ORDER_SUBSTR):
            if sub in r:
                return (i, r)
        return (len(MODEL_PLOT_ORDER_SUBSTR), r)

    return sorted(rel_paths, key=sort_key)


def plot_grouped_models_by_level_ax(
    summaries: Dict[str, Dict[int, dict]],
    ax: plt.Axes,
    ylabel: str,
    legend: bool = True,
) -> None:
    models = ordered_model_rel_paths(summaries.keys())
    n_models = len(models)
    if n_models == 0:
        return

    x = np.arange(len(LEVELS), dtype=float)

    if n_models <= 1:
        width = 0.5
    else:
        width = min(0.8 / n_models, 0.14)
    offsets = (np.arange(n_models) - (n_models - 1) / 2.0) * width * 1.15

    ymax_obs = 0.0
    for mi, model in enumerate(models):
        ys: List[float] = []
        for lv in LEVELS:
            s = summaries[model].get(lv, {})
            ys.append(float(s.get("mean", float("nan"))))
        color = PASTEL_COLORS[mi % len(PASTEL_COLORS)]
        xpos = x + offsets[mi]
        ys_plot = [0.0 if (isinstance(v, float) and np.isnan(v)) else v for v in ys]
        if ys_plot:
            ymax_obs = max(ymax_obs, max(ys_plot))

        ax.bar(
            xpos,
            ys_plot,
            width * 0.98,
            label=_short_model_label(model),
            color=color,
            alpha=0.93,
            edgecolor="#f4f4f4",
            linewidth=1.1,
            zorder=3,
        )

        for xv, yv in zip(xpos, ys_plot):
            if yv > 0:
                ax.text(
                    xv,
                    yv + max(0.12, 0.012 * max(ymax_obs, 1.0)),
                    f"{yv:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=9.0,
                    color="#000000",
                    zorder=5,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([f"Level {lv}" for lv in LEVELS])
    ax.set_ylabel(ylabel, fontsize=16)

    # Easy/Moderate/Hard 구간 강조
    ax.axvspan(x[0] - 0.5, x[0] + 0.5, color="#c8ddd0", alpha=0.52, zorder=0)
    ax.axvspan(x[1] - 0.5, x[-2] + 0.5, color="#ecebe7", alpha=0.35, zorder=0)
    ax.axvspan(x[-1] - 0.5, x[-1] + 0.5, color="#edd0d0", alpha=0.52, zorder=0)

    ax.text(
        x[0],
        (ymax_obs * 1.06) if ymax_obs > 0 else 1.0,
        "Easy",
        ha="center",
        va="bottom",
        fontsize=11,
        color="#4f7a60",
    )
    ax.text(
        x[len(x) // 2],
        (ymax_obs * 1.06) if ymax_obs > 0 else 1.0,
        "Moderate",
        ha="center",
        va="bottom",
        fontsize=11,
        color="#6f6a5d",
    )
    ax.text(
        x[-1],
        (ymax_obs * 1.06) if ymax_obs > 0 else 1.0,
        "Hard",
        ha="center",
        va="bottom",
        fontsize=11,
        color="#9a4f4f",
    )

    ax.tick_params(axis="x", which="major", labelsize=12, colors="#000000", pad=5, width=1.4, length=4)
    ax.tick_params(axis="y", which="major", labelsize=12, colors="#000000", pad=5, width=1.4, length=0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#2d3a4b")
    ax.spines["bottom"].set_color("#2d3a4b")
    ax.spines["left"].set_linewidth(2.2)
    ax.spines["bottom"].set_linewidth(2.2)

    if legend and n_models > 1:
        ax.legend(
            loc="upper center",
            # legend이 axes 바깥(위쪽/가운데 gap)으로 나가면 우측 패널과 겹쳐 보이므로
            # 축 내부로 넣어서 잘림/겹침을 방지
            bbox_to_anchor=(0.5, 1.08),
            frameon=False,
            ncol=n_models,
            handlelength=1.4,
            columnspacing=1.4,
            fontsize=11,
            borderaxespad=0.15,
        )

    ax.grid(axis="y", alpha=0.95, color="#c7c1b8", linestyle="-", linewidth=1.0, zorder=1)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(bottom=0, top=max(1.0, ymax_obs * 1.18))


# -----------------------------
# Right panel: answer diversity
# -----------------------------


def analyze_sample(sample: dict) -> dict:
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

    # unique answer count: 정규화 전 원문 문자열 기준
    raw_unique = len(set(str(a).strip() for a in answers if a is not None))
    return {
        "num_chains": len(answers),
        "is_correct": bool(is_correct),
        "diversity_metrics": {"num_unique_answers": raw_unique},
    }


def aggregate_by_level(json_path: str) -> Dict[int, dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    correct_vals: Dict[int, List[float]] = {lv: [] for lv in LEVELS}
    incorrect_vals: Dict[int, List[float]] = {lv: [] for lv in LEVELS}

    for sample in data:
        if not isinstance(sample, dict):
            continue
        lv = _parse_level(sample)
        if lv is None:
            continue

        result = analyze_sample(sample)
        nu = result["diversity_metrics"]["num_unique_answers"]
        if result["num_chains"] == 0:
            continue

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

    return per_level


def plot_level_diversity_ax(per_level: Dict[int, dict], ax: plt.Axes) -> None:
    n = len(LEVELS)
    x = np.arange(n, dtype=float)
    correct_y = [per_level[lv]["correct_mean"] for lv in LEVELS]
    incorrect_y = [per_level[lv]["incorrect_mean"] for lv in LEVELS]

    # 원래 스크립트 스타일 유지
    correct_color = "#2f7fbd"
    incorrect_color = "#d65a4a"

    ax.plot(
        x,
        correct_y,
        label="Correct",
        color=correct_color,
        linewidth=4.6,
        marker="o",
        markersize=12,
        markerfacecolor=correct_color,
        markeredgecolor="white",
        markeredgewidth=2.2,
        zorder=3,
    )
    ax.plot(
        x,
        incorrect_y,
        label="Incorrect",
        color=incorrect_color,
        linewidth=4.6,
        marker="o",
        markersize=12,
        markerfacecolor=incorrect_color,
        markeredgecolor="white",
        markeredgewidth=2.2,
        zorder=3,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"Level {lv}" for lv in LEVELS])
    # 한 figure 안에서 좌/우 글씨 스케일을 맞추기 위해 tick을 좌측 패널과 동일 밴드로 통일
    ax.tick_params(
        axis="x",
        which="major",
        labelsize=12,
        colors="#000000",
        pad=5,
        width=1.4,
        length=4,
    )
    ax.tick_params(
        axis="y",
        which="major",
        labelsize=12,
        colors="#000000",
        pad=5,
        width=1.4,
        length=0,
    )

    ymax_data = max(correct_y + incorrect_y) if (correct_y + incorrect_y) else 1.0

    ax.legend(
        loc="upper center",
        # 좌측 패널과 동일한 legend y-anchor
        bbox_to_anchor=(0.5, 1.08),
        ncol=2,
        frameon=False,
        fontsize=11,
        # 기본보다 길게 해서 legend의 가로선(라인 샘플) 길이를 더 길게 표현
        handlelength=3.75,
        columnspacing=1.45,
        borderaxespad=0.15,
    )

    ax.grid(axis="y", alpha=0.9, color="#c7c1b8", linestyle="-", linewidth=1.0)
    ax.grid(axis="x", visible=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#2d3a4b")
    ax.spines["bottom"].set_color("#2d3a4b")
    ax.spines["left"].set_linewidth(2.2)
    ax.spines["bottom"].set_linewidth(2.2)

    ax.set_ylim(0, ymax_data + max(1.2, ymax_data * 0.22))

    offset = max(0.18, ymax_data * 0.06)
    for i, y in enumerate(correct_y):
        ax.text(
            x[i],
            y + offset,
            f"{y:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
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
            fontsize=11,
            fontweight="bold",
            color=incorrect_color,
        )


def normalize_stages_config(value: str) -> str:
    v = (value or "").strip()
    if not v:
        return v
    if v.isdigit():
        return f"stages-{v}"
    return v


def normalize_model_key(model: str) -> str:
    s = (model or "").strip().replace("\\", "/")
    return "".join(ch for ch in s.lower() if ch.isalnum() or ch in {".", "_", "-", "/"})


def find_math_train_for_model(model: str, stages_config: str, results_root: str) -> str | None:
    sc = normalize_stages_config(stages_config)
    if not sc:
        return None

    model = model.strip().strip("/").replace("\\", "/")
    if not model:
        return None

    # 1) results_root / model / stages-XX / Stage1 / MATH_train.json (model에 슬래시 포함 가능)
    cand = os.path.join(results_root, model, sc, "Stage1", "MATH_train.json")
    if os.path.isfile(cand):
        return cand

    # 2) 짧은 이름 기반 탐색: .../results/**/<basename>/stages-XX/Stage1/MATH_train.json
    target_leaf = os.path.basename(model)
    model_key = normalize_model_key(model)
    target_leaf_key = normalize_model_key(target_leaf)

    matches: List[str] = []
    if not os.path.isdir(results_root):
        return None

    for root, _dirs, files in os.walk(results_root):
        if os.path.basename(root) != "Stage1" or "MATH_train.json" not in files:
            continue
        if os.path.basename(os.path.dirname(root)) != sc:
            continue
        model_dir = os.path.dirname(os.path.dirname(root))
        leaf = os.path.basename(model_dir)
        rel = os.path.relpath(model_dir, results_root).replace("\\", "/")
        leaf_key = normalize_model_key(leaf)
        rel_key = normalize_model_key(rel)
        if leaf == target_leaf or rel == model or leaf_key == target_leaf_key or rel_key == model_key:
            matches.append(os.path.join(root, "MATH_train.json"))

    if len(matches) == 1:
        return matches[0]
    return None


def resolve_math_train_path_for_right_model(
    *,
    right_model: str,
    right_stages_config: str,
    results_root: str,
    right_stage1_dir: str | None,
) -> str:
    if right_stage1_dir:
        d = right_stage1_dir
        if not os.path.isabs(d) and rsc_project_dir:
            d = os.path.join(rsc_project_dir, d)
        cand = os.path.join(d, "MATH_train.json")
        if os.path.isfile(cand):
            return cand
        raise FileNotFoundError(f"파일 없음: {cand}")

    json_path = find_math_train_for_model(right_model, right_stages_config, results_root)
    if not json_path:
        raise FileNotFoundError(
            f"MATH_train.json 을 찾지 못했습니다. model={right_model!r}, stages={right_stages_config!r}"
        )
    return json_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MATH_train 결과를 좌/우 2패널로 합쳐서 시각화"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="experiments/results",
        help="rsc_project 기준 결과 루트",
    )
    parser.add_argument(
        "--left-stages-pattern",
        type=str,
        default="stages-40",
        help="left 패널용: Stage1 상위 폴더 glob (예: stages-40, stages-*)",
    )
    parser.add_argument(
        "--left-metric",
        type=str,
        choices=["gold", "full_sc"],
        default="gold",
        help="좌측(sc_consensus)에서 정답 일치(gold) 또는 full SC 합의(full_sc) 중 선택",
    )

    parser.add_argument(
        "--right-model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="우측(answer_diversity)에서 사용할 단일 모델",
    )
    parser.add_argument(
        "--right-stages-config",
        type=str,
        default="stages-40",
        help="우측(answer_diversity)용 stages 폴더명 (예: stages-40, stages-100, 40)",
    )
    parser.add_argument(
        "--right-stage1-dir",
        type=str,
        default=None,
        help="우측 패널용 Stage1 폴더 직접 지정(우선)",
    )
    parser.add_argument(
        "--output-png",
        type=str,
        default=None,
        help="combined 모드 또는 both 모드에서 사용할 합본 PNG 경로 (--output-mode split에서는 분리 파일명 생성용 기준 경로)",
    )
    parser.add_argument(
        "--output-mode",
        type=str,
        choices=["combined", "split", "both"],
        default="combined",
        help="combined: 좌우 한 파일 | split: 좌·우 각각 저장 | both: 합본 + 각각 저장",
    )
    parser.add_argument(
        "--output-left-png",
        type=str,
        default=None,
        help="split/both 모드에서 좌측 패널만 저장할 경로 (--output-png 대신 직접 지정 가능)",
    )
    parser.add_argument(
        "--output-right-png",
        type=str,
        default=None,
        help="split/both 모드에서 우측 패널만 저장할 경로 (--output-png 대신 직접 지정 가능)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="기본 이미지(2048x696) 기준 dpi",
    )
    args = parser.parse_args()

    results_root = args.results_dir
    if not os.path.isabs(results_root) and rsc_project_dir:
        results_root = os.path.join(rsc_project_dir, results_root)
    results_root = os.path.abspath(results_root)

    # ----------------------
    # Left: grouped models
    # ----------------------
    pairs = discover_math_train_paths(results_root, args.left_stages_pattern)
    pairs = [(rel, jp) for rel, jp in pairs if not _skip_model_rel(rel)]
    if not pairs:
        raise FileNotFoundError(
            f"MATH_train.json 을 찾지 못했습니다. stages_pattern={args.left_stages_pattern!r} (Llama 제외 후 비었음)"
        )

    summaries_left: Dict[str, Dict[int, dict]] = {}
    for rel, jp in pairs:
        by_sc, by_gold = aggregate_file_both_metrics(jp)
        if args.left_metric == "gold":
            summaries_left[rel] = {lv: _summarize(by_gold[lv]) for lv in LEVELS}
        else:
            summaries_left[rel] = {lv: _summarize(by_sc[lv]) for lv in LEVELS}

    if args.left_metric == "gold":
        ylabel_left = "Required reasoning chains (ground truth)"
    else:
        ylabel_left = "Required reasoning chains (full SC consensus)"

    # ----------------------
    # Right: one model lines
    # ----------------------
    right_json_path = resolve_math_train_path_for_right_model(
        right_model=args.right_model,
        right_stages_config=args.right_stages_config,
        results_root=results_root,
        right_stage1_dir=args.right_stage1_dir,
    )
    per_level_right = aggregate_by_level(right_json_path)

    default_vis_dir = os.path.join(rsc_project_dir or script_dir, "visualization")
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    out_dir = os.path.join(default_vis_dir, script_name)
    os.makedirs(out_dir, exist_ok=True)
    dpi = args.dpi
    # 기준 합본 이미지: 2048 x 696 (dpi 기준 인치 크기)
    fig_w_combined = 2048 / dpi
    fig_h_combined = 696 / dpi

    combined_out: str | None = None
    if args.output_mode in ("combined", "both"):
        if args.output_png:
            combined_out = args.output_png if os.path.isabs(args.output_png) else os.path.abspath(args.output_png)
        else:
            combined_out = os.path.join(out_dir, f"{script_name}.png")

    stem_for_split = None
    if args.output_png:
        stem_for_split, _split_ext = os.path.splitext(
            os.path.basename(args.output_png if os.path.isabs(args.output_png) else os.path.abspath(args.output_png))
        )

    split_left_out = args.output_left_png
    if split_left_out:
        split_left_out = split_left_out if os.path.isabs(split_left_out) else os.path.abspath(split_left_out)
    elif args.output_mode in ("split", "both"):
        stem = stem_for_split or script_name
        split_left_out = os.path.join(out_dir, f"{stem}_panel_left.png")

    split_right_out = args.output_right_png
    if split_right_out:
        split_right_out = split_right_out if os.path.isabs(split_right_out) else os.path.abspath(split_right_out)
    elif args.output_mode in ("split", "both"):
        stem = stem_for_split or script_name
        split_right_out = os.path.join(out_dir, f"{stem}_panel_right.png")

    # ----------------------
    # Combined figure
    # ----------------------
    if args.output_mode in ("combined", "both"):
        assert combined_out is not None

        fig, (ax_left, ax_right) = plt.subplots(
            1,
            2,
            figsize=(fig_w_combined, fig_h_combined),
            dpi=dpi,
            gridspec_kw={
                "width_ratios": [COMBINED_WIDTH_RATIOS_LEFT, COMBINED_WIDTH_RATIOS_RIGHT],
            },
        )
        fig.patch.set_facecolor("white")
        ax_left.set_facecolor("white")
        ax_right.set_facecolor("white")

        plot_grouped_models_by_level_ax(summaries_left, ax_left, ylabel=ylabel_left, legend=True)
        plot_level_diversity_ax(per_level_right, ax_right)

        # 좌/우 사이 패딩 및 상단 범례 공간 확보(원 스크립트 tight_layout을 그대로 쓰면 패널 간 간섭 가능)
        # 좌측 범례가 우측과 겹치거나 잘리지 않도록 여백 재조정
        # bottom/wspace를 줄여서 실제 플롯 영역을 조금 더 키우고,
        # wspace 감소로 오른쪽 패널을 살짝 좌측으로 당김(우측 잘림 방지 목적)
        fig.subplots_adjust(left=0.055, right=0.985, top=0.90, bottom=0.09, wspace=0.13)

        # 기준 이미지 크기와의 호환을 위해 bbox_inches="tight" 없이 figure 크기 고정 저장
        plt.savefig(combined_out, dpi=dpi)
        print(f"시각화 저장(합본): {combined_out}")
        plt.close()

    # ----------------------
    # Separate figures (panels)
    # ----------------------
    if args.output_mode in ("split", "both"):
        assert split_left_out is not None and split_right_out is not None

        w_left, w_right = _panel_fig_sizes_for_split(fig_w_combined, fig_h_combined)

        fig_l, ax_l = plt.subplots(1, 1, figsize=(w_left, fig_h_combined), dpi=dpi)
        fig_l.patch.set_facecolor("white")
        ax_l.set_facecolor("white")
        plot_grouped_models_by_level_ax(summaries_left, ax_l, ylabel=ylabel_left, legend=True)
        fig_l.subplots_adjust(left=0.065, right=0.965, top=0.90, bottom=0.09)
        plt.savefig(split_left_out, dpi=dpi)
        print(f"시각화 저장(좌측): {split_left_out}")
        plt.close(fig_l)

        fig_r, ax_r = plt.subplots(1, 1, figsize=(w_right, fig_h_combined), dpi=dpi)
        fig_r.patch.set_facecolor("white")
        ax_r.set_facecolor("white")
        plot_level_diversity_ax(per_level_right, ax_r)
        fig_r.subplots_adjust(left=0.075, right=0.945, top=0.90, bottom=0.09)
        plt.savefig(split_right_out, dpi=dpi)
        print(f"시각화 저장(우측): {split_right_out}")
        plt.close(fig_r)


if __name__ == "__main__":
    main()

