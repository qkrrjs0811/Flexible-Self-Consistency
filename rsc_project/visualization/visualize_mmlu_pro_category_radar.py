"""MMLU-Pro 카테고리별 Accuracy / Input+Output 토큰 레이더 차트."""

import argparse
import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _set_style() -> None:
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
            "axes.unicode_minus": False,
        }
    )


_set_style()

METHOD_ORDER = ["SC", "AC", "ESC", "DSC", "Ours"]
# polar: 카테고리(각도) / 방사형 눈금(accuracy·token) 글자 크기
_CATEGORY_LABEL_FONTSIZE = 14
_RADIAL_LABEL_FONTSIZE = 13
_SUBTITLE_FONTSIZE = 15
# legend 글자 크기
_LEGEND_FONTSIZE = 13
# token 서브플롯 방사형 눈금 (고정)
_TOKEN_RADIAL_TICKS = [0, 6000, 12000, 18000, 24000, 30000]
_TOKEN_RADIAL_MAX = 30000
# 2x1 polar 세로 간격: 아래 subtitle/카테고리 라벨 겹침 방지용
_SUBPLOT_HSPACE = 0.42
_SUBTITLE_Y = -0.16

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_STAGES_DIR = "stages-40"
METRICS_FILENAME = "MMLU-Pro_metrics_by_category.json"

METHOD_STYLES = {
    "SC": {"color": "#95a5a6", "label": "SC", "alpha": 0.75},
    "AC": {"color": "#e67e22", "label": "AC", "alpha": 0.75},
    "ESC": {"color": "#3498db", "label": "ESC", "alpha": 0.75},
    "DSC": {"color": "#9b59b6", "label": "DSC", "alpha": 0.8},
    "Ours": {"color": "#2ecc71", "label": "FSC (Ours)", "alpha": 0.95},
}


def _results_root() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(script_dir, "..", "experiments", "results"))


def _json_path_for_model(model: str, stages_dir: str) -> str:
    """model: results 하위 상대 경로 (예: Qwen/Qwen2.5-7B-Instruct)."""
    m = model.strip().strip("/").replace(os.sep, "/")
    return os.path.normpath(
        os.path.join(_results_root(), m, stages_dir, METRICS_FILENAME)
    )


def _safe_model_slug(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def _output_model_slug(model: str) -> str:
    """PNG 파일명용: 마지막 디렉터리 이름 (예: Qwen2.5-7B-Instruct)."""
    tail = model.strip().rstrip("/").replace("\\", "/").split("/")[-1]
    return _safe_model_slug(tail)


def _collect_categories(data: Dict) -> List[str]:
    sections = data.get("sections", {})
    categories = set()
    for sec in ("rsc", "dsc", "ours"):
        part = sections.get(sec, {})
        if isinstance(part, dict):
            categories.update(part.keys())
    return sorted(categories)


def _extract_metrics(
    data: Dict, categories: List[str]
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    sections = data.get("sections", {})
    rsc = sections.get("rsc", {})
    dsc = sections.get("dsc", {})
    ours = sections.get("ours", {})

    acc_map: Dict[str, List[float]] = {m: [] for m in METHOD_ORDER}
    tok_map: Dict[str, List[float]] = {m: [] for m in METHOD_ORDER}

    for cat in categories:
        for method in ("SC", "AC", "ESC"):
            entry = rsc.get(cat, {}).get(method, {})
            acc = float(entry.get("accuracy", np.nan))
            in_tok = float(entry.get("avg_input_tokens", np.nan))
            out_tok = float(entry.get("avg_output_tokens", np.nan))
            tok = in_tok + out_tok if np.isfinite(in_tok) and np.isfinite(out_tok) else np.nan
            acc_map[method].append(acc)
            tok_map[method].append(tok)

        dsc_entry = dsc.get(cat, {}).get("dsc", {})
        dsc_acc = float(dsc_entry.get("accuracy", np.nan))
        dsc_in = float(dsc_entry.get("avg_input_tokens", np.nan))
        dsc_out = float(dsc_entry.get("avg_output_tokens", np.nan))
        dsc_tok = dsc_in + dsc_out if np.isfinite(dsc_in) and np.isfinite(dsc_out) else np.nan
        acc_map["DSC"].append(dsc_acc)
        tok_map["DSC"].append(dsc_tok)

        ours_entry = ours.get(cat, {}).get("Ours", {})
        ours_acc = float(ours_entry.get("accuracy", np.nan))
        ours_in = float(ours_entry.get("avg_input_tokens", np.nan))
        ours_out = float(ours_entry.get("avg_output_tokens", np.nan))
        ours_tok = ours_in + ours_out if np.isfinite(ours_in) and np.isfinite(ours_out) else np.nan
        acc_map["Ours"].append(ours_acc)
        tok_map["Ours"].append(ours_tok)

    return acc_map, tok_map


def _close(values: List[float]) -> List[float]:
    return values + [values[0]] if values else values


def create_plot(
    categories: List[str],
    acc_map: Dict[str, List[float]],
    tok_map: Dict[str, List[float]],
    output_path: str,
) -> None:
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles_closed = angles + [angles[0]]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 15),
        subplot_kw={"projection": "polar"},
    )
    ax_acc, ax_tok = axes

    for ax in axes:
        ax.set_theta_offset(np.pi / 2.0)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles)
        ax.set_xticklabels(categories, fontsize=_CATEGORY_LABEL_FONTSIZE)
        ax.tick_params(axis="x", pad=12)
        ax.grid(True, alpha=0.35, linestyle="--", linewidth=0.6)

    ax_acc.set_ylim(0, 100)
    ax_acc.set_yticks([20, 40, 60, 80, 100])
    ax_acc.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=_RADIAL_LABEL_FONTSIZE)
    ax_acc.tick_params(axis="y", labelsize=_RADIAL_LABEL_FONTSIZE)

    ax_tok.set_ylim(0, _TOKEN_RADIAL_MAX)
    ax_tok.set_yticks(_TOKEN_RADIAL_TICKS)
    ax_tok.set_yticklabels([f"{v:,}" for v in _TOKEN_RADIAL_TICKS], fontsize=_RADIAL_LABEL_FONTSIZE)
    ax_tok.tick_params(axis="y", labelsize=_RADIAL_LABEL_FONTSIZE)

    for method in METHOD_ORDER:
        style = METHOD_STYLES[method]
        lw = 2.2 if method == "Ours" else 1.8
        acc_vals = _close(acc_map[method])
        tok_vals = _close(tok_map[method])

        ax_acc.plot(angles_closed, acc_vals, color=style["color"], linewidth=lw, alpha=style["alpha"], label=style["label"])
        ax_acc.fill(angles_closed, acc_vals, color=style["color"], alpha=0.12 if method == "Ours" else 0.08)

        ax_tok.plot(angles_closed, tok_vals, color=style["color"], linewidth=lw, alpha=style["alpha"], label=style["label"])
        ax_tok.fill(angles_closed, tok_vals, color=style["color"], alpha=0.12 if method == "Ours" else 0.08)

    for ax, subtitle in ((ax_acc, "Accuracy"), (ax_tok, "Tokens")):
        ax.text(
            0.5,
            _SUBTITLE_Y,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=_SUBTITLE_FONTSIZE,
            fontweight="bold",
        )

    handles, labels = ax_acc.get_legend_handles_labels()
    hmap = {lab: h for h, lab in zip(handles, labels)}
    ordered_labels = [METHOD_STYLES[m]["label"] for m in METHOD_ORDER]
    ordered_handles = [hmap[l] for l in ordered_labels if l in hmap]
    ordered_labels = [l for l in ordered_labels if l in hmap]
    fig.legend(
        ordered_handles,
        ordered_labels,
        loc="upper center",
        ncol=5,
        fontsize=_LEGEND_FONTSIZE,
        frameon=True,
        framealpha=0.95,
        bbox_to_anchor=(0.5, 1.005),
    )

    # polar 축은 tight_layout만으로는 겹침이 남는 경우가 있어,
    # legend 영역(top)과 subplot 간격(hspace)을 명시적으로 확보.
    fig.subplots_adjust(top=0.925, bottom=0.06, hspace=_SUBPLOT_HSPACE)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MMLU-Pro 카테고리별 레이더 차트 (Accuracy / Input+Output 토큰).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
실행 예 (저장소 루트에서, 기본 모델 Qwen/Qwen2.5-7B-Instruct / stages-40):
  python rsc_project/visualization/visualize_mmlu_pro_category_radar.py

다른 모델 (experiments/results 아래 상대 경로):
  python rsc_project/visualization/visualize_mmlu_pro_category_radar.py \\
    --model Qwen/Qwen2.5-14B-Instruct

stages 폴더가 stages-40이 아닐 때:
  python rsc_project/visualization/visualize_mmlu_pro_category_radar.py \\
    --model google/gemma-3-4b-it --stages stages-40

출력 경로 직접 지정:
  python rsc_project/visualization/visualize_mmlu_pro_category_radar.py \\
    --model Qwen/Qwen2.5-7B-Instruct --output_file /tmp/my_radar.png
""",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"results 하위 모델 경로 (기본: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default=DEFAULT_STAGES_DIR,
        help=f"결과 디렉터리명 (기본: {DEFAULT_STAGES_DIR}, 또는 40 → stages-40)",
    )
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()

    stages_dir = args.stages if str(args.stages).startswith("stages-") else f"stages-{args.stages}"
    json_file = _json_path_for_model(args.model, stages_dir)
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"missing json: {json_file}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.output_file:
        out = os.path.abspath(args.output_file)
    else:
        model_slug = _output_model_slug(args.model)
        out = os.path.join(script_dir, f"mmlu_pro_category_radar_{model_slug}.png")

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    categories = _collect_categories(data)
    acc_map, tok_map = _extract_metrics(data, categories)

    create_plot(categories, acc_map, tok_map, out)


if __name__ == "__main__":
    main()
