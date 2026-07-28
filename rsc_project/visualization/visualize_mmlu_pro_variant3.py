"""MMLU-Pro 토큰당 성능(Efficiency) 정적 막대 시각화."""

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
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#2d3a4b",
            "axes.labelcolor": "#4a4a4a",
            "xtick.color": "#6a6a6a",
            "ytick.color": "#6a6a6a",
        }
    )


_set_style()

METHOD_ORDER = ["SC", "AC", "ESC", "DSC", "Ours"]
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
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


def _compute_efficiency(
    acc_map: Dict[str, List[float]], tok_map: Dict[str, List[float]]
) -> Dict[str, List[float]]:
    # Efficiency Score = Accuracy / Tokens
    # 시각적 비교를 위해 1k 스케일로 키워 표시(절대값보다 상대 비교가 목적).
    eff_map: Dict[str, List[float]] = {m: [] for m in METHOD_ORDER}
    for method in METHOD_ORDER:
        for acc, tok in zip(acc_map[method], tok_map[method]):
            if np.isfinite(acc) and np.isfinite(tok) and tok > 0:
                eff_map[method].append((acc / tok) * 1000.0)
            else:
                eff_map[method].append(np.nan)
    return eff_map


def create_plot(
    categories: List[str],
    eff_map: Dict[str, List[float]],
    output_path: str,
    logy: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(15, 8))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    x = np.arange(len(categories), dtype=float)
    bar_width = 0.16
    offsets = (np.arange(len(METHOD_ORDER)) - (len(METHOD_ORDER) - 1) / 2.0) * bar_width

    for idx, method in enumerate(METHOD_ORDER):
        vals = np.asarray(eff_map[method], dtype=float)
        mask = np.isfinite(vals)
        if logy:
            mask = mask & (vals > 0)
        style = METHOD_STYLES[method]
        ax.bar(
            x[mask] + offsets[idx],
            vals[mask],
            width=bar_width * 0.95,
            color=style["color"],
            alpha=style["alpha"],
            edgecolor="#5a5a60",
            linewidth=0.8,
            label=style["label"],
            zorder=4 if method == "Ours" else 3,
        )

    if logy:
        positive_vals = []
        for method in METHOD_ORDER:
            vals = np.asarray(eff_map[method], dtype=float)
            positive_vals.extend(vals[np.isfinite(vals) & (vals > 0)].tolist())
        if not positive_vals:
            raise ValueError("log scale을 사용할 양수 efficiency 값이 없습니다.")
        ax.set_yscale("log")
        ax.set_ylim(min(positive_vals) * 0.8, max(positive_vals) * 1.25)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=28, ha="right", fontsize=13)
    ax.set_ylabel("Accuracy / Tokens x 1000", fontsize=16, fontweight="normal")
    ax.grid(axis="y", linestyle="-", linewidth=1.0, alpha=0.95, color="#b9b9b9")
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#2d3a4b")
    ax.spines["bottom"].set_color("#2d3a4b")
    ax.spines["left"].set_linewidth(2.2)
    ax.spines["bottom"].set_linewidth(2.2)
    ax.tick_params(axis="both", labelsize=13, length=6, width=1.8, colors="#6a6a6a")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=5,
        frameon=False,
        fontsize=15,
        handlelength=2.5,
        columnspacing=1.4,
        handletextpad=0.6,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MMLU-Pro 토큰당 성능(Efficiency) 정적 PNG 시각화.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
실행 예 (저장소 루트, 기본 모델):
  python rsc_project/visualization/visualize_mmlu_pro_variant3.py

다른 모델:
  python rsc_project/visualization/visualize_mmlu_pro_variant3.py \\
    --model Qwen/Qwen2.5-14B-Instruct

출력 경로 지정(PNG):
  python rsc_project/visualization/visualize_mmlu_pro_variant3.py \\
    --output_file /tmp/mmlu_variant3_efficiency.png
""",
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--stages", type=str, default=DEFAULT_STAGES_DIR)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument(
        "--logy",
        action="store_true",
        help="Y축을 로그 스케일로 표시 (양수 값만 렌더링).",
    )
    args = parser.parse_args()

    stages_dir = args.stages if str(args.stages).startswith("stages-") else f"stages-{args.stages}"
    json_file = _resolve_json_path_for_model(args.model, stages_dir)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.output_file:
        out = os.path.abspath(args.output_file)
    else:
        model_slug = _output_model_slug(args.model)
        script_name = os.path.splitext(os.path.basename(__file__))[0]
        out = os.path.join(script_dir, script_name, f"{script_name}_{model_slug}.png")

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    categories = _collect_categories(data)
    acc_map, tok_map = _extract_metrics(data, categories)
    eff_map = _compute_efficiency(acc_map, tok_map)

    create_plot(categories, eff_map, out, logy=args.logy)


if __name__ == "__main__":
    main()
