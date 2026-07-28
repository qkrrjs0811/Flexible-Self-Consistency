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
_CATEGORY_LABEL_FONTSIZE = 15
_RADIAL_LABEL_FONTSIZE = 14
_SUBTITLE_FONTSIZE = 18
# legend 글자 크기
_LEGEND_FONTSIZE = 15
# token 서브플롯 방사형 눈금 개수(0 포함)
_TOKEN_RADIAL_TICK_COUNT = 6
# 2x1 polar 세로 간격: 아래 subtitle/카테고리 라벨 겹침 방지용
_SUBPLOT_HSPACE = 0.45
_SUBTITLE_Y = -0.16
_CARD_BG = "#ffffff"
_GRID_COLOR = "#b9b9b9"
_SPOKE_COLOR = "#cdcdcd"

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_MODELS = [
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "google/gemma-3-4b-it",
]
DEFAULT_STAGES_DIR = "stages-40"
METRICS_FILENAME = "MMLU-Pro_metrics_by_category.json"

METHOD_STYLES = {
    "SC": {"color": "#8fb8de", "label": "SC", "alpha": 0.95},
    "AC": {"color": "#f3bf8f", "label": "AC", "alpha": 0.95},
    "ESC": {"color": "#9fd3a8", "label": "ESC", "alpha": 0.95},
    "DSC": {"color": "#f0a3a3", "label": "DSC", "alpha": 0.95},
    "Ours": {"color": "#c3a6e8", "label": "FSC (Ours)", "alpha": 1.0},
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


def _normalize_model_key(model: str) -> str:
    s = (model or "").strip().replace("\\", "/")
    return "".join(
        ch for ch in s.lower() if ch.isalnum() or ch in {".", "_", "-", "/"}
    )


def _resolve_json_path_for_model(model: str, stages_dir: str) -> str:
    """
    model 입력을 유연하게 해석해 metrics json 경로를 찾는다.
    - 정확 경로: results/<model>/<stages>/MMLU-Pro_metrics_by_category.json
    - 짧은 이름/소문자: leaf 또는 rel path를 대소문자 무시 매칭
    """
    direct = _json_path_for_model(model, stages_dir)
    if os.path.exists(direct):
        return direct

    results_root = _results_root()
    if not os.path.isdir(results_root):
        raise FileNotFoundError(f"missing results root: {results_root}")

    model_norm = _normalize_model_key(model.strip().strip("/"))
    leaf_norm = _normalize_model_key(model.strip().strip("/").split("/")[-1])
    matches: List[str] = []

    for root, _dirs, files in os.walk(results_root):
        if METRICS_FILENAME not in files:
            continue
        if os.path.basename(root) != stages_dir:
            continue

        model_dir = os.path.dirname(root)
        rel_model = os.path.relpath(model_dir, results_root).replace("\\", "/")
        rel_norm = _normalize_model_key(rel_model)
        rel_leaf_norm = _normalize_model_key(os.path.basename(model_dir))
        if rel_norm == model_norm or rel_leaf_norm == leaf_norm:
            matches.append(os.path.join(root, METRICS_FILENAME))

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        preview = "\n".join(matches[:8])
        more = f"\n... 외 {len(matches) - 8}개" if len(matches) > 8 else ""
        raise ValueError(
            "model 이름이 여러 경로와 일치합니다. --model 에 전체 상대 경로를 주세요.\n"
            f"{preview}{more}"
        )

    raise FileNotFoundError(
        f"missing json for model={model!r}, stages={stages_dir!r}\ntried: {direct}"
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


def _token_radial_scale(tok_map: Dict[str, List[float]]) -> Tuple[int, List[int]]:
    """
    모델별 실제 토큰 최대값을 기반으로 토큰 레이더 축 상한/눈금을 계산한다.
    - 최대값에 소폭(3%) 여유를 둔 뒤
    - 눈금 간격(step)은 500 또는 1000의 배수로 계산
      (라벨 숫자가 500/1000 단위로 깔끔하게 떨어지도록)
    """
    vals: List[float] = []
    for method in METHOD_ORDER:
        for v in tok_map.get(method, []):
            if np.isfinite(v) and v >= 0:
                vals.append(float(v))

    max_tok = max(vals) if vals else 1.0
    padded = max_tok * 1.03

    rough_step = max(padded / (_TOKEN_RADIAL_TICK_COUNT - 1), 1.0)
    granularity = 500 if rough_step <= 5000 else 1000
    step = int(np.ceil(rough_step / granularity) * granularity)
    step = max(step, granularity)

    radial_max = step * (_TOKEN_RADIAL_TICK_COUNT - 1)
    ticks = [i * step for i in range(_TOKEN_RADIAL_TICK_COUNT)]
    return radial_max, ticks


def create_plot(
    categories: List[str],
    acc_map: Dict[str, List[float]],
    tok_map: Dict[str, List[float]],
    output_path: str,
    horizontal: bool = False,
) -> None:
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles_closed = angles + [angles[0]]

    if horizontal:
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(18, 8.2),
            subplot_kw={"projection": "polar"},
        )
        ax_acc, ax_tok = axes
    else:
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(12, 15),
            subplot_kw={"projection": "polar"},
        )
        ax_acc, ax_tok = axes
    fig.patch.set_facecolor("#ffffff")

    for ax in axes:
        ax.set_facecolor(_CARD_BG)
        ax.set_theta_offset(np.pi / 2.0)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles)
        ax.set_xticklabels(categories, fontsize=_CATEGORY_LABEL_FONTSIZE, fontweight="normal")
        ax.tick_params(axis="x", pad=12)
        ax.yaxis.grid(True, alpha=0.95, linestyle="-", linewidth=1.0, color=_GRID_COLOR)
        ax.xaxis.grid(True, alpha=0.7, linestyle="-", linewidth=0.9, color=_SPOKE_COLOR)
        ax.spines["polar"].set_color("#2d3a4b")
        ax.spines["polar"].set_linewidth(2.2)

    ax_acc.set_ylim(0, 100)
    ax_acc.set_yticks([20, 40, 60, 80, 100])
    ax_acc.set_yticklabels(
        ["20", "40", "60", "80", "100"], fontsize=_RADIAL_LABEL_FONTSIZE, color="#7a7a80"
    )
    ax_acc.tick_params(axis="y", labelsize=_RADIAL_LABEL_FONTSIZE)

    token_radial_max, token_radial_ticks = _token_radial_scale(tok_map)
    ax_tok.set_ylim(0, token_radial_max)
    ax_tok.set_yticks(token_radial_ticks)
    ax_tok.set_yticklabels(
        [f"{v:,}" for v in token_radial_ticks], fontsize=_RADIAL_LABEL_FONTSIZE, color="#7a7a80"
    )
    ax_tok.tick_params(axis="y", labelsize=_RADIAL_LABEL_FONTSIZE)

    for method in METHOD_ORDER:
        style = METHOD_STYLES[method]
        lw = 3.0 if method == "Ours" else 2.4
        acc_vals = _close(acc_map[method])
        tok_vals = _close(tok_map[method])

        ax_acc.plot(
            angles_closed,
            acc_vals,
            color=style["color"],
            linewidth=lw,
            alpha=style["alpha"],
            label=style["label"],
            marker="o",
            markersize=4.0 if method == "Ours" else 3.4,
        )
        ax_acc.fill(
            angles_closed, acc_vals, color=style["color"], alpha=0.13 if method == "Ours" else 0.075
        )

        ax_tok.plot(
            angles_closed,
            tok_vals,
            color=style["color"],
            linewidth=lw,
            alpha=style["alpha"],
            label=style["label"],
            marker="o",
            markersize=4.0 if method == "Ours" else 3.4,
        )
        ax_tok.fill(
            angles_closed, tok_vals, color=style["color"], alpha=0.13 if method == "Ours" else 0.075
        )

    for ax, subtitle in ((ax_acc, "Accuracy"), (ax_tok, "Tokens")):
        ax.text(
            0.5,
            _SUBTITLE_Y,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=_SUBTITLE_FONTSIZE,
            fontweight="normal",
            color="#000000",
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
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
        handlelength=2.6,
        columnspacing=1.6,
        handletextpad=0.6,
    )

    # polar 축은 tight_layout만으로는 겹침이 남는 경우가 있어 여백을 명시적으로 확보.
    if horizontal:
        fig.subplots_adjust(top=0.86, bottom=0.12, wspace=0.36)
    else:
        fig.subplots_adjust(top=0.925, bottom=0.06, hspace=_SUBPLOT_HSPACE)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MMLU-Pro 카테고리별 레이더 차트 (Accuracy / Input+Output 토큰).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
실행 예 (저장소 루트에서, 기본 4개 모델 / stages-40, 각 모델마다 세로+가로 2종 저장):
  python rsc_project/visualization/visualize_mmlu_pro_category_radar.py

다른 모델 (experiments/results 아래 상대 경로):
  python rsc_project/visualization/visualize_mmlu_pro_category_radar.py \\
    --model Qwen/Qwen2.5-14B-Instruct

stages 폴더가 stages-40이 아닐 때:
  python rsc_project/visualization/visualize_mmlu_pro_category_radar.py \\
    --model google/gemma-3-4b-it --stages stages-40

출력 경로 직접 지정 (단일 모델, 세로: 경로 그대로, 가로: 동일 이름에 _horizontal 접미사):
  python rsc_project/visualization/visualize_mmlu_pro_category_radar.py \\
    --model Qwen/Qwen2.5-7B-Instruct --output_file /tmp/my_radar.png
""",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "results 하위 단일 모델 경로. "
            "미지정 시 기본 4개 모델을 모두 순회 실행."
        ),
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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_name = os.path.splitext(os.path.basename(__file__))[0]

    models_to_run = [args.model] if args.model else DEFAULT_MODELS

    # output_file은 단일 모델 실행에서만 명시 경로로 사용
    if args.output_file and len(models_to_run) > 1:
        raise ValueError("--output_file 은 --model 단일 실행일 때만 사용할 수 있습니다.")

    for model in models_to_run:
        json_file = _resolve_json_path_for_model(model, stages_dir)

        if args.output_file:
            out = os.path.abspath(args.output_file)
        else:
            model_slug = _output_model_slug(model)
            out = os.path.join(script_dir, script_name, f"{script_name}_{model_slug}.png")
        root, ext = os.path.splitext(out)
        out_horizontal = f"{root}_horizontal{ext or '.png'}"

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        categories = _collect_categories(data)
        acc_map, tok_map = _extract_metrics(data, categories)

        create_plot(categories, acc_map, tok_map, out, horizontal=False)
        create_plot(categories, acc_map, tok_map, out_horizontal, horizontal=True)


if __name__ == "__main__":
    main()
