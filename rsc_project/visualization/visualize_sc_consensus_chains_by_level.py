"""
MATH_train.json: level(1–5)별로 prefix majority가 목표와 일치하는 최소 chain 수 k(평균)을 시각화.

메트릭 (둘 중 선택 또는 동시 출력, --metric):

  full_sc — 전체 SC majority와 동일해지는 최소 k:
    모든 chain의 정규화 답으로 majority M을 구한 뒤, k=1..N에서 첫 k개의 majority M_k가
    M과 같아지는 최소 k.

  gold — 정답(ground truth)과 일치하는 최소 k:
    샘플의 answer를 정규화한 gt와, prefix majority(정규화)가 같아지는 최소 k.
    어떤 k에서도 majority가 gt와 같지 않으면(전체 SC도 오답이면) k=N으로 둠.

입력: experiments/results/**/Stage1/MATH_train.json (기본 stages-40)
출력: 발견된 모델을 한 figure에 (Llama 제외). 순서: Qwen2.5-3B → 7B → 14B → gemma-3-4b → 그 외.
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

from evaluation.metrics import extract_cot_data, normalize_answer_eval

LEVELS = [1, 2, 3, 4, 5]

# figure 내 모델 막대 순서 (rel 경로에 부분 문자열 포함 여부로 매칭, 앞이 우선)
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
    """
    Returns:
        (min_k, n_chains): min_k는 유효할 때 1..n, 답 없으면 (None, 0)
    """
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
    """
    prefix majority(정규화)가 정답과 같아지는 최소 k.
    어떤 k에서도 일치하지 않으면 (전체 N개로도 SC가 정답이 아니면) N 반환.

    Returns:
        (min_k, n_chains): 답 없으면 (None, 0)
    """
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


def discover_math_train_paths(
    results_root: str,
    stages_pattern: str = "stages-*",
) -> List[Tuple[str, str]]:
    """
    Returns:
        [(model_rel_path, json_path), ...]  model_rel_path = results 기준 모델 디렉터리
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


def aggregate_file_both_metrics(
    json_path: str,
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]], int, int]:
    """
    Returns:
        by_level_full_sc, by_level_gold: level -> min_k 리스트
        n_parsed, n_skipped
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    by_sc: Dict[int, List[int]] = {lv: [] for lv in LEVELS}
    by_gold: Dict[int, List[int]] = {lv: [] for lv in LEVELS}
    n_skip = 0
    n_ok = 0
    for sample in data:
        if not isinstance(sample, dict):
            n_skip += 1
            continue
        lv = _parse_level(sample)
        if lv is None:
            n_skip += 1
            continue
        k_sc, nc = min_chains_for_full_sc_majority(sample)
        k_gold, nc2 = min_chains_for_gold_match(sample)
        if k_sc is None or nc == 0 or k_gold is None or nc2 == 0:
            n_skip += 1
            continue
        by_sc[lv].append(k_sc)
        by_gold[lv].append(k_gold)
        n_ok += 1
    return by_sc, by_gold, n_ok, n_skip


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
    """표시용 짧은 이름 (마지막 경로 조각)."""
    base = os.path.basename(rel_path.rstrip("/"))
    return MODEL_DISPLAY_NAMES.get(base, base)


def ordered_model_rel_paths(rel_paths: Iterable[str]) -> List[str]:
    """MODEL_PLOT_ORDER_SUBSTR 순서 우선, 나머지는 rel 경로 문자열 알파벳순."""

    def sort_key(rel: str) -> Tuple[int, str]:
        r = rel.replace("\\", "/")
        for i, sub in enumerate(MODEL_PLOT_ORDER_SUBSTR):
            if sub in r:
                return (i, r)
        return (len(MODEL_PLOT_ORDER_SUBSTR), r)

    return sorted(rel_paths, key=sort_key)


def plot_grouped_models_by_level(
    summaries: Dict[str, Dict[int, dict]],
    output_path: str,
    ylabel: str = "Required reasoning chains",
) -> None:
    """summaries[model][level] = {mean, n}; level 1–5막대에 평균 min-k만 표시."""
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

    fig, ax = plt.subplots(figsize=(10.2, 6.6))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

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
        # 막대 값 annotation: 레벨별 격차를 정량으로 읽기 쉽게 보조
        for xv, yv in zip(xpos, ys_plot):
            if yv > 0:
                ax.text(
                    xv,
                    yv + max(0.12, 0.012 * max(ymax_obs, 1.0)),
                    f"{yv:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=10.5,
                    color="#000000",
                    zorder=5,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([f"Level {lv}" for lv in LEVELS])
    ax.set_ylabel(ylabel)
    # 가장 쉬운/어려운 구간을 옅게 강조해 난이도 대비를 직관적으로 표현
    ax.axvspan(x[0] - 0.5, x[0] + 0.5, color="#c8ddd0", alpha=0.52, zorder=0)   # Easy
    ax.axvspan(x[1] - 0.5, x[-2] + 0.5, color="#ecebe7", alpha=0.35, zorder=0)  # Moderate
    ax.axvspan(x[-1] - 0.5, x[-1] + 0.5, color="#edd0d0", alpha=0.52, zorder=0) # Hard
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

    ax.tick_params(axis="x", which="major", labelsize=15, colors="#000000", pad=8, width=1.8, length=6)
    ax.tick_params(axis="y", which="major", labelsize=15, colors="#000000", pad=8, width=1.8, length=0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#2d3a4b")
    ax.spines["bottom"].set_color("#2d3a4b")
    ax.spines["left"].set_linewidth(2.2)
    ax.spines["bottom"].set_linewidth(2.2)

    if n_models > 1:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.08),
            frameon=False,
            ncol=n_models,
            handlelength=1.6,
            columnspacing=1.4,
        )
    ax.grid(axis="y", alpha=0.95, color="#c7c1b8", linestyle="-", linewidth=1.0, zorder=1)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(bottom=0, top=max(1.0, ymax_obs * 1.18))

    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"시각화 저장: {output_path}")


def _default_png_paths(vis_dir: str, slug_stages: str, metric: str) -> str:
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    out_dir = os.path.join(vis_dir, script_name)
    if metric == "full_sc":
        return os.path.join(
            out_dir,
            f"sc_consensus_chains_by_level_full_sc_all_{slug_stages}.png",
        )
    if metric == "gold":
        return os.path.join(
            out_dir,
            f"sc_consensus_chains_by_level_gold_all_{slug_stages}.png",
        )
    raise ValueError(metric)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MATH_train: level별 최소 chain 수 (full SC 합의 또는 정답 일치)"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="experiments/results",
        help="rsc_project 기준 결과 루트",
    )
    parser.add_argument(
        "--stages-pattern",
        type=str,
        default="stages-40",
        help="Stage1 상위 폴더 이름 glob (예: stages-40, stages-*)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["full_sc", "gold", "both"],
        default="both",
        help="full_sc: 전체 SC majority와 맞추는 k, gold: 정답과 맞추는 k(불가 시 N), both: PNG 두 장",
    )
    parser.add_argument(
        "--output-png",
        type=str,
        default=None,
        help="출력 PNG (--metric both일 때는 stem에 _full_sc / _gold 가 붙음)",
    )
    args = parser.parse_args()

    results_root = args.results_dir
    if not os.path.isabs(results_root) and rsc_project_dir:
        results_root = os.path.join(rsc_project_dir, results_root)
    results_root = os.path.abspath(results_root)

    pairs = discover_math_train_paths(results_root, args.stages_pattern)
    pairs = [(rel, jp) for rel, jp in pairs if not _skip_model_rel(rel)]
    if not pairs:
        print(
            f"MATH_train.json 을 찾지 못했습니다. (Llama 제외 후 비었음) stages_pattern={args.stages_pattern!r}",
            file=sys.stderr,
        )
        sys.exit(1)

    summaries_sc: Dict[str, Dict[int, dict]] = {}
    summaries_gold: Dict[str, Dict[int, dict]] = {}

    for rel, jp in pairs:
        print(f"처리 중: {rel}  <- {jp}")
        by_sc, by_gold, n_ok, n_skip = aggregate_file_both_metrics(jp)
        print(f"  유효 샘플: {n_ok}, 스킵: {n_skip}")
        summaries_sc[rel] = {lv: _summarize(by_sc[lv]) for lv in LEVELS}
        summaries_gold[rel] = {lv: _summarize(by_gold[lv]) for lv in LEVELS}

    vis_dir = os.path.join(rsc_project_dir or script_dir, "visualization")
    os.makedirs(vis_dir, exist_ok=True)
    slug_stages = args.stages_pattern.replace("*", "all")

    ylabel_sc = "Required reasoning chains (full SC consensus)"
    ylabel_gold = "Required reasoning chains (ground truth)"

    def resolve_out(metric_key: str) -> str:
        if args.output_png:
            p = (
                args.output_png
                if os.path.isabs(args.output_png)
                else os.path.abspath(args.output_png)
            )
            if args.metric == "both":
                root, ext = os.path.splitext(p)
                if not ext:
                    ext = ".png"
                suf = "_full_sc" if metric_key == "full_sc" else "_gold"
                return f"{root}{suf}{ext}"
            return p
        return _default_png_paths(vis_dir, slug_stages, metric_key)

    if args.metric in ("full_sc", "both"):
        out_sc = resolve_out("full_sc")
        plot_grouped_models_by_level(summaries_sc, out_sc, ylabel=ylabel_sc)
    if args.metric in ("gold", "both"):
        out_gold = resolve_out("gold")
        plot_grouped_models_by_level(summaries_gold, out_gold, ylabel=ylabel_gold)

    print("완료.")


if __name__ == "__main__":
    main()
