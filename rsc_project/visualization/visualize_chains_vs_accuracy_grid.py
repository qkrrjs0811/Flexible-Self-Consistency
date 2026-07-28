"""
Chain 수 vs Accuracy 그리드 시각화

summary_single_*.txt 파일을 파싱하여:
- 각 데이터셋별로 서브플롯 생성
- x축: 사용한 chain 수 (avg_chains)
- y축: accuracy
- SC, ESC, AC, DSC, Ours 5개 방법론을 각 서브플롯에 표시
"""

import re
import os
import sys
import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
import numpy as np
from typing import Dict, List, Tuple, Optional
from math import ceil, sqrt

def _set_latex_paper_mpl_style() -> None:
    """첨부된 figure와 유사한 serif(논문) 스타일."""
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
            "grid.color": "#bcbcbc",
            "grid.linewidth": 1.0,
            "axes.grid": False,
            "axes.unicode_minus": False,
        }
    )


_set_latex_paper_mpl_style()

# ============================================================================
# 분석 설정
# ============================================================================
# 분석할 모델 (None이면 자동으로 찾기)
TARGET_MODEL = 'Qwen2.5-7B-Instruct'  # 또는 None

# 시각화에 사용할 데이터셋 리스트 (None이면 모든 데이터셋 사용)
VISUALIZATION_DATASETS = ['MATH500_test', 'AMC23', 'AIME_2024', 'AIME_2025']

# 데이터셋 이름 매핑 (시각화용 표시 이름)
DATASET_DISPLAY_NAMES = {
    'MATH500_test': 'MATH500',
    'AIME_2024': 'AIME2024',
    'AIME_2025': 'AIME2025',
    'GPQA-Diamond': 'GPQA',
    'MMLU-Pro': 'MMLU-Pro',
}
# ============================================================================

def _apply_reference_like_style(ax, x_min: float, x_max: float, y_min: float, y_max: float) -> None:
    """첨부 이미지 느낌의 미니멀 스타일과 좌상단 강조 영역 적용."""
    ax.set_facecolor("#ffffff")

    ax.grid(axis="y", linestyle="-", alpha=0.95, linewidth=1.0)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(2.2)
    ax.spines["bottom"].set_linewidth(2.2)
    ax.spines["left"].set_color("#2d3a4b")
    ax.spines["bottom"].set_color("#2d3a4b")
    ax.tick_params(axis="both", labelsize=13, length=6, width=1.8, colors="#000000")


def _split_summary_by_model(text: str) -> List[Tuple[str, str]]:
    """
    summary_single_*.txt 를 'Evaluating {model}...' 블록으로 분할.
    Returns: [(model_name, block_text), ...]
    """
    pat = re.compile(r"^Evaluating (.+?)\.\.\.\s*$", re.MULTILINE)
    matches = list(pat.finditer(text))
    blocks: List[Tuple[str, str]] = []
    for i, m in enumerate(matches):
        model = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        blocks.append((model, text[start:end]))
    return blocks


def _parse_table_rows(lines: List[str], start_idx: int) -> Tuple[List[List[str]], int]:
    """
    'Dataset | ...' 헤더 다음의 '-' 라인 이후부터 '=' 라인 전까지를
    '|' 기준으로 split한 rows로 반환.
    Returns: (rows, next_idx)
    """
    i = start_idx
    # seek dashed line
    while i < len(lines) and not lines[i].strip().startswith("-"):
        i += 1
    if i < len(lines) and lines[i].strip().startswith("-"):
        i += 1
    rows: List[List[str]] = []
    while i < len(lines):
        s = lines[i].rstrip("\n")
        st = s.strip()
        if re.match(r"^=+$", st):
            i += 1
            break
        if "|" in s:
            parts = [p.strip() for p in s.split("|")]
            rows.append(parts)
        i += 1
    return rows, i


def parse_summary_single_file(file_path: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    summary_single_*.txt 에서 모델별/데이터셋별 SC/AC/ESC/DSC/Ours의
    accuracy와 avg_chains를 추출.

    Returns:
        { model_name: { dataset_name: { method: {'accuracy': float, 'avg_chains': float} } } }
    """
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    all_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    blocks = _split_summary_by_model(text)
    if not blocks:
        return all_results

    for model, block in blocks:
        lines = block.splitlines()
        acc_map: Dict[str, Dict[str, float]] = {}
        chains_map: Dict[str, Dict[str, float]] = {}

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Accuracy Summary table
            if line == "Accuracy Summary":
                # find header
                while i < len(lines) and "Dataset" not in lines[i]:
                    i += 1
                if i >= len(lines):
                    break
                header = [p.strip() for p in lines[i].split("|")]
                rows, i = _parse_table_rows(lines, i + 1)
                # Expect columns like: Dataset | ... | SC Acc | AC Acc | ESC Acc | DSC Acc | Ours Acc
                col_idx = {name: idx for idx, name in enumerate(header)}
                for r in rows:
                    if len(r) < 2:
                        continue
                    ds = r[col_idx.get("Dataset", 0)].strip()
                    if not ds or ds == "Dataset":
                        continue
                    def _get(col: str) -> Optional[float]:
                        idx = col_idx.get(col)
                        if idx is None or idx >= len(r):
                            return None
                        try:
                            return float(r[idx])
                        except Exception:
                            return None
                    acc_map[ds] = {
                        "SC": _get("SC Acc") or 0.0,
                        "AC": _get("AC Acc") or 0.0,
                        "ESC": _get("ESC Acc") or 0.0,
                        "DSC": _get("DSC Acc") or 0.0,
                        "Ours": _get("Ours Acc") or 0.0,
                    }
                continue

            # Chains Summary table
            if line == "Chains Summary":
                while i < len(lines) and "Dataset" not in lines[i]:
                    i += 1
                if i >= len(lines):
                    break
                header = [p.strip() for p in lines[i].split("|")]
                rows, i = _parse_table_rows(lines, i + 1)
                col_idx = {name: idx for idx, name in enumerate(header)}
                for r in rows:
                    if len(r) < 2:
                        continue
                    ds = r[col_idx.get("Dataset", 0)].strip()
                    if not ds or ds == "Dataset":
                        continue
                    def _get(col: str) -> Optional[float]:
                        idx = col_idx.get(col)
                        if idx is None or idx >= len(r):
                            return None
                        try:
                            return float(r[idx])
                        except Exception:
                            return None
                    chains_map[ds] = {
                        "SC": _get("SC Chains") or 0.0,
                        "AC": _get("AC Chains") or 0.0,
                        "ESC": _get("ESC Chains") or 0.0,
                        "DSC": _get("DSC Chains") or 0.0,
                        "Ours": _get("Ours Chains") or 0.0,
                    }
                continue

            i += 1

        # Merge acc + chains into expected structure
        model_data: Dict[str, Dict[str, float]] = {}
        datasets = set(acc_map.keys()) | set(chains_map.keys())
        for ds in datasets:
            if ds not in acc_map or ds not in chains_map:
                continue
            mdata: Dict[str, Dict[str, float]] = {}
            for method in ["SC", "AC", "ESC", "DSC", "Ours"]:
                mdata[method] = {
                    "accuracy": float(acc_map[ds].get(method, 0.0)),
                    "avg_chains": float(chains_map[ds].get(method, 0.0)),
                }
            model_data[ds] = mdata

        if model_data:
            all_results[model] = model_data

    return all_results


def create_grid_visualization(
    all_results: Dict,
    output_path: str,
    dataset_filter: List[str] = None,
    single_plot: bool = False,
):
    """
    그리드 형태의 Chain 수 vs Accuracy 시각화 생성
    
    각 데이터셋별로 서브플롯을 생성하여 방법론들을 비교
    """
    # 방법론별 색상 및 마커
    method_styles = {
        'SC': {'color': '#1f77b4', 'marker': 'o', 'label': 'SC', 'alpha': 0.95},      # blue
        'AC': {'color': '#ff7f0e', 'marker': 'o', 'label': 'AC', 'alpha': 0.95},      # orange
        'ESC': {'color': '#2ca02c', 'marker': 'o', 'label': 'ESC', 'alpha': 0.95},    # green
        'DSC': {'color': '#d62728', 'marker': 'X', 'label': 'DSC', 'alpha': 0.95},    # red
        'Ours': {'color': '#9467bd', 'marker': 'o', 'label': 'FSC (Ours)', 'alpha': 1.0}  # purple
    }
    
    # legend 항목 순서 요청: SC, AC, ESC, DSC, Ours
    methods = ['SC', 'AC', 'ESC', 'DSC', 'Ours']
    
    # 데이터 수집
    dataset_data = {}
    
    for model_path, dataset_results in all_results.items():
        for dataset_name, methods_data in dataset_results.items():
            if dataset_filter and dataset_name not in dataset_filter:
                continue
            
            display_name = DATASET_DISPLAY_NAMES.get(dataset_name, dataset_name)
            if display_name not in dataset_data:
                dataset_data[display_name] = {}
            
            for method in methods:
                if method in methods_data:
                    data = methods_data[method]
                    avg_chains = data.get('avg_chains')
                    if avg_chains is not None:
                        dataset_data[display_name][method] = {
                            'chains': avg_chains,
                            'accuracy': data['accuracy']
                        }
    
    # 데이터셋 리스트를 지정된 순서로 정렬
    dataset_order = ['GSM8K', 'MATH500', 'AMC23', 'AIME2024', 'AIME2025', 'GPQA', 'MMLU-Pro']
    datasets = []
    for ds in dataset_order:
        if ds in dataset_data:
            datasets.append(ds)
    # 순서에 없는 데이터셋도 추가 (뒤에)
    for ds in dataset_data.keys():
        if ds not in datasets:
            datasets.append(ds)
    n_datasets = len(datasets)
    
    if n_datasets == 0:
        print("Error: 시각화할 데이터가 없습니다.")
        return

    if single_plot:
        fig, ax = plt.subplots(1, 1, figsize=(11, 7))
        fig.patch.set_facecolor("#ffffff")

        all_points = []
        for dataset_name in datasets:
            data = dataset_data[dataset_name]
            for method in methods:
                if method in data:
                    all_points.append((data[method]["chains"], data[method]["accuracy"]))

        if not all_points:
            print("Error: 단일 플롯 시각화용 데이터가 없습니다.")
            return

        x_values = [p[0] for p in all_points]
        y_values = [p[1] for p in all_points]
        x_min, x_max = min(x_values), max(x_values)
        x_margin = (x_max - x_min) * 0.08 if x_max > x_min else max(1.0, x_max * 0.2 if x_max > 0 else 1.0)
        y_min, y_max = min(y_values), max(y_values)
        y_margin = max((y_max - y_min) * 0.35, 8.0)
        global_ymin = max(0.0, y_min - y_margin)
        global_ymax = min(100.0, y_max + y_margin)

        _apply_reference_like_style(
            ax,
            max(0, x_min - x_margin),
            x_max + x_margin,
            global_ymin,
            global_ymax,
        )

        method_to_marker = {
            "SC": "o",
            "AC": "^",
            "ESC": "s",
            "DSC": "X",
            "Ours": "D",
        }
        cmap = plt.get_cmap("tab10")
        dataset_to_color = {ds: cmap(i % 10) for i, ds in enumerate(datasets)}
        baseline_methods = ["SC", "AC", "ESC", "DSC"]
        fsc_reductions_by_method = {m: [] for m in baseline_methods}
        fsc_points = []

        for dataset_name in datasets:
            data = dataset_data[dataset_name]
            ds_color = dataset_to_color[dataset_name]
            dataset_points = []
            if "Ours" in data:
                ours_chains = data["Ours"]["chains"]
                for baseline in baseline_methods:
                    if baseline in data and data[baseline]["chains"] > 0:
                        base_chains = data[baseline]["chains"]
                        reduction = (base_chains - ours_chains) / base_chains * 100.0
                        fsc_reductions_by_method[baseline].append(reduction)

            for method in methods:
                if method not in data:
                    continue
                style = method_styles[method]
                chains = data[method]["chains"]
                accuracy = data[method]["accuracy"]
                dataset_points.append((chains, accuracy))
                if method == "Ours":
                    fsc_points.append((chains, accuracy))

                ax.scatter(
                    chains,
                    accuracy,
                    color=style["color"],
                    marker=method_to_marker.get(method, "o"),
                    s=190,
                    alpha=0.95,
                    edgecolors=ds_color,
                    linewidths=2.2,
                    zorder=5 if method == "Ours" else 3,
                )

                # 방법론 인식 강화를 위해 모든 점에 축약 라벨 표시
                ax.annotate(
                    style["label"].replace("FSC (Ours)", "FSC"),
                    (chains, accuracy),
                    xytext=(0, -13),
                    textcoords="offset points",
                    fontsize=9.8,
                    color=style["color"],
                    ha="center",
                    va="top",
                    zorder=6,
                )

                if method == "Ours":
                    # 겹치는 AIME 라벨은 방향을 분리해 가독성 확보
                    label_offsets = {
                        "AIME2024": (-12, 0),
                        "AIME2025": (0, 16),
                    }
                    dx, dy = label_offsets.get(dataset_name, (7, 7))
                    ax.annotate(
                        dataset_name,
                        (chains, accuracy),
                        xytext=(dx, dy),
                        textcoords="offset points",
                        fontsize=11,
                        color=ds_color,
                        ha="right" if dx < 0 else "left",
                        va="bottom" if dy >= 0 else "top",
                        bbox=dict(
                            boxstyle="round,pad=0.28",
                            facecolor="white",
                            edgecolor=ds_color,
                            alpha=0.85,
                            linewidth=1.2,
                        ),
                        zorder=6,
                    )

            if len(dataset_points) >= 2:
                # 방법론 순서 대신 x축(chains) 기준으로 왼쪽 -> 오른쪽 연결
                dataset_points = sorted(dataset_points, key=lambda p: p[0])
                ds_x = [p[0] for p in dataset_points]
                ds_y = [p[1] for p in dataset_points]
                ax.plot(
                    ds_x,
                    ds_y,
                    linestyle="--",
                    linewidth=1.9,
                    color=ds_color,
                    alpha=0.55,
                    zorder=2,
                )

        # FSC 포인트들을 따라가는 효율 영역(노란색) 추가
        if len(fsc_points) >= 2:
            x_left = max(0, x_min - x_margin)
            x_right = x_max + x_margin
            fsc_sorted = sorted(fsc_points, key=lambda p: p[0])
            y_pad = (global_ymax - global_ymin) * 0.03

            top_path = [(x_left, global_ymax)]
            top_path.extend([(x, min(global_ymax, y + y_pad)) for x, y in fsc_sorted])

            # 영역을 x축(현재 y축 하한)까지 내리기
            bottom_y = global_ymin
            right_x = min(x_right, max(x for x, _ in fsc_sorted) * 1.08)

            zone_points = top_path + [(right_x, bottom_y), (x_left, bottom_y)]
            zone = Polygon(
                zone_points,
                closed=True,
                facecolor="#f2cc63",
                edgecolor="none",
                alpha=0.22,
                zorder=1,
            )
            ax.add_patch(zone)
            ax.text(
                x_left + (x_right - x_left) * 0.01,
                global_ymax - (global_ymax - global_ymin) * 0.02,
                "Most sample-efficient",
                fontsize=18,
                fontweight="bold",
                color="#c48a00",
                ha="left",
                va="top",
                zorder=2,
            )

        ax.set_xlim([max(0, x_min - x_margin), x_max + x_margin])
        ax.set_ylim([global_ymin, global_ymax])
        ax.set_xlabel("Number of Chains", fontsize=18, fontweight="normal")
        ax.set_ylabel("Accuracy (%)", fontsize=18, fontweight="normal")

        # 좌하단 빈 영역에 FSC 효율성 요약 패널 추가
        avg_reductions = {
            m: (np.mean(vals) if vals else 0.0) for m, vals in fsc_reductions_by_method.items()
        }
        panel_text = (
            "Avg Chain Reduction (FSC)\n"
            f"- vs SC: {avg_reductions['SC']:.1f}%\n"
            f"- vs AC: {avg_reductions['AC']:.1f}%\n"
            f"- vs ESC: {avg_reductions['ESC']:.1f}%\n"
            f"- vs DSC: {avg_reductions['DSC']:.1f}%"
        )
        ax.text(
            0.03,
            0.06,
            panel_text,
            transform=ax.transAxes,
            fontsize=15,
            ha="left",
            va="bottom",
            color="#4b2e83",
            bbox=dict(
                boxstyle="round,pad=0.80",
                facecolor="#f5efff",
                edgecolor=method_styles["Ours"]["color"],
                linewidth=2.1,
                alpha=0.9,
            ),
            zorder=7,
        )

        method_handles = [
            Line2D(
                [0],
                [0],
                marker=method_to_marker.get(m, "o"),
                color="none",
                markerfacecolor=method_styles[m]["color"],
                markeredgecolor="#333333",
                markersize=11,
                label=method_styles[m]["label"],
            )
            for m in methods
        ]

        legend1 = ax.legend(
            handles=method_handles,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            frameon=False,
            fontsize=10.5,
            title="Method (color)",
            title_fontsize=11.5,
            borderaxespad=0.0,
        )
        ax.add_artist(legend1)

        plt.tight_layout(rect=[0, 0, 0.82, 1])
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"단일 플롯 시각화 저장 완료: {output_path}")
        plt.close()
        return
    
    # 데이터셋이 4개일 때 2x2 그리드, 그 외에는 한 줄(1 x n)
    if n_datasets == 4:
        n_rows, n_cols = 2, 2
    else:
        n_rows, n_cols = 1, n_datasets
    
    # 그래프 생성
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    fig.patch.set_facecolor("#ffffff")
    
    # axes를 항상 2D 배열로 만들기
    if n_datasets == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    else:
        axes = axes.reshape(n_rows, n_cols)
    
    # 전체 데이터셋의 accuracy 범위 계산 (공통 스케일을 위해)
    all_accuracies = []
    for dataset_name in datasets:
        if dataset_name in dataset_data:
            data = dataset_data[dataset_name]
            for method in methods:
                if method in data:
                    # summary_single_*.txt의 accuracy는 이미 (%) 값임
                    all_accuracies.append(data[method]['accuracy'])
    
    # 전체 범위 계산
    if all_accuracies:
        global_min_acc = min(all_accuracies)
        global_max_acc = max(all_accuracies)
        global_range = global_max_acc - global_min_acc
        # 더 넓은 margin 적용 (50% 이상)
        global_margin = max(global_range * 0.5, 10)  # 최소 10% margin
        global_ymin = max(0, global_min_acc - global_margin)
        global_ymax = min(100, global_max_acc + global_margin)
    else:
        global_ymin = 0
        global_ymax = 100
    
    # 각 서브플롯에 데이터 플롯
    for idx, dataset_name in enumerate(datasets):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        data = dataset_data[dataset_name]
        
        # 첨부 이미지 느낌의 배경/축 스타일
        x_vals = [v['chains'] for v in data.values()]
        x_min = min(x_vals) if x_vals else 0.0
        x_max = max(x_vals) if x_vals else 1.0
        x_margin = (x_max - x_min) * 0.1 if x_max > x_min else max(1.0, x_max * 0.2 if x_max > 0 else 1.0)
        local_xmin = max(0, x_min - x_margin)
        local_xmax = x_max + x_margin
        _apply_reference_like_style(ax, local_xmin, local_xmax, global_ymin, global_ymax)

        # 각 방법론별로 점 표시
        for method in methods:
            if method in data:
                style = method_styles[method]
                chains = data[method]['chains']
                accuracy = data[method]['accuracy']  # summary에서는 이미 % 단위
                
                # 점 크기 (첨부 이미지 톤 반영)
                size = 130
                
                ax.scatter(chains, accuracy,
                          color=style['color'],
                          marker=style['marker'],
                          s=size,
                          alpha=style['alpha'],
                          edgecolors='#f4f4f4' if method != 'DSC' else '#111111',
                          linewidths=1.2,
                          label=style['label'],
                          zorder=5 if method == 'Ours' else 3)

                ax.annotate(
                    style['label'],
                    (chains, accuracy),
                    xytext=(8, 0),
                    textcoords='offset points',
                    fontsize=10,
                    color='#1f1f1f',
                    ha='left',
                    va='center',
                    zorder=6,
                )
        
        # 서브플롯 제목
        ax.set_title(dataset_name, fontsize=15, fontweight='normal', pad=10)
        
        # 축 라벨: 모든 서브플롯에 동일하게 표기
        ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='normal')
        ax.set_xlabel('Number of Examples', fontsize=13, fontweight='normal')
        
        # Y축 범위 설정 (전체 데이터셋의 공통 범위 사용)
        ax.set_ylim([global_ymin, global_ymax])
        
        # 효율성 비교 annotation 추가
        if 'Ours' in data:
            ours_chains = data['Ours']['chains']
            ours_acc = data['Ours']['accuracy']
            
            # 비교할 방법론 선택 (SC를 baseline으로 사용, 없으면 가장 가까운 성능의 방법론)
            comparison_method = None
            if 'SC' in data:
                comparison_method = 'SC'
            else:
                # Ours와 가장 가까운 성능을 가진 방법론 찾기
                min_diff = float('inf')
                for method in ['ESC', 'AC']:
                    if method in data:
                        acc_diff = abs(data[method]['accuracy'] - ours_acc)
                        if acc_diff < min_diff:
                            min_diff = acc_diff
                            comparison_method = method
            
            if comparison_method and comparison_method in data:
                comp_chains = data[comparison_method]['chains']
                comp_acc = data[comparison_method]['accuracy']
                
                # 효율성 계산
                if comp_chains > ours_chains:
                    reduction_pct = ((comp_chains - ours_chains) / comp_chains) * 100
                    speedup = comp_chains / ours_chains
                    
                    # Y축 offset 계산 (AIME는 위로, 나머지는 아래로)
                    # 데이터셋별로 점들의 위치에 따라 동적으로 조정
                    min_acc_in_subplot = min(comp_acc, ours_acc)
                    max_acc_in_subplot = max(comp_acc, ours_acc)
                    
                    # AIME 데이터셋은 위로, 나머지는 아래로
                    if 'AIME' in dataset_name:
                        # 위로 올리기
                        y_offset = max(
                            (global_ymax - global_ymin) * 0.12,  # 최소 12% offset
                            (global_ymax - max_acc_in_subplot) * 0.3  # 또는 최대값까지의 30%
                        )
                        connection_y = max(comp_acc, ours_acc) + y_offset
                        
                        # connection_y가 너무 위로 가지 않도록 제한
                        max_connection_y = global_ymax + (global_ymax - global_ymin) * 0.05
                        connection_y = min(connection_y, max_connection_y)
                    else:
                        # 아래로 내리기
                        y_offset = max(
                            (global_ymax - global_ymin) * 0.12,  # 최소 12% offset
                            (min_acc_in_subplot - global_ymin) * 0.3  # 또는 최소값까지의 30%
                        )
                        connection_y = min(comp_acc, ours_acc) - y_offset
                        
                        # connection_y가 너무 아래로 가지 않도록 제한
                        min_connection_y = global_ymin - (global_ymax - global_ymin) * 0.05
                        connection_y = max(connection_y, min_connection_y)
                    
                    # 1. 비교 방법론 점에서 수직 점선
                    ax.plot([comp_chains, comp_chains],
                           [comp_acc, connection_y],
                           color='#2ecc71',
                           linestyle='--',
                           linewidth=1.5,
                           alpha=0.6,
                           zorder=2)
                    
                    # 2. Ours 점에서 수직 점선
                    ax.plot([ours_chains, ours_chains],
                           [ours_acc, connection_y],
                           color='#2ecc71',
                           linestyle='--',
                           linewidth=1.5,
                           alpha=0.6,
                           zorder=2)
                    
                    # 3. 두 점선을 연결하는 수평 점선 화살표
                    arrow = FancyArrowPatch(
                        (comp_chains, connection_y),
                        (ours_chains, connection_y),
                        arrowstyle='->',
                        linestyle='--',
                        mutation_scale=20,
                        color='#2ecc71',
                        linewidth=2.0,
                        alpha=0.7,
                        zorder=2
                    )
                    ax.add_patch(arrow)
                    
                    # 텍스트 annotation (AIME는 아래, 나머지는 위)
                    mid_x = (comp_chains + ours_chains) / 2
                    
                    # 효율성 텍스트: "-XX.X% fewer chains" (영어 표기, 감소 느낌)
                    # reduction_pct는 양수(%) 감소량이며, 여기서 음수 기호를 붙여 '-XX%' 형태로 표현
                    text = f"-{reduction_pct:.1f}% fewer chains"
                    
                    # AIME는 화살표 위에, 나머지는 위에 텍스트 배치
                    if 'AIME' in dataset_name:
                        text_y = connection_y + (global_ymax - global_ymin) * 0.015
                        va_align = 'bottom'
                    else:
                        text_y = connection_y + (global_ymax - global_ymin) * 0.015
                        va_align = 'bottom'
                    
                    ax.annotate(text,
                               (mid_x, text_y),
                               fontsize=12,
                               fontweight='bold',
                               color='#2ecc71',
                               ha='center',
                               va=va_align,
                               bbox=dict(boxstyle='round,pad=0.65',
                                        facecolor='white',
                                        edgecolor='#2ecc71',
                                        alpha=0.9,
                                        linewidth=2.2),
                               zorder=3)
        
        # X축 범위 설정
        chains_list = [data[m]['chains'] for m in methods if m in data]
        if chains_list:
            min_chains = min(chains_list)
            max_chains = max(chains_list)
            margin = (max_chains - min_chains) * 0.1 if max_chains > min_chains else 5
            ax.set_xlim([max(0, min_chains - margin), max_chains + margin])
    
    # 빈 서브플롯 숨기기
    for idx in range(n_datasets, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    # 범례: title 자리에 상단 중앙 1줄로 배치
    handles, labels = axes[0, 0].get_legend_handles_labels()
    hmap = {lab: h for h, lab in zip(handles, labels)}
    desired_labels = ['SC', 'AC', 'ESC', 'DSC', 'FSC (Ours)']
    ordered_labels = [lab for lab in desired_labels if lab in hmap]
    ordered_handles = [hmap[lab] for lab in ordered_labels]
    fig.legend(
        ordered_handles,
        ordered_labels,
        loc='upper center',
        ncol=5,
        fontsize=15,
        frameon=False,
        bbox_to_anchor=(0.5, 0.95),
    )
    
    # legend 공간 확보
    plt.tight_layout(rect=[0, 0.04, 1, 0.87])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"그리드 시각화 저장 완료: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Chain 수 vs Accuracy 그리드 시각화')
    parser.add_argument('--summary_file', type=str, default=None,
                      help='summary_single_*.txt 파일 경로 (미지정 시 기본 경로 사용)')
    parser.add_argument('--output_file', type=str, default=None,
                       help='출력 이미지 파일 경로')
    parser.add_argument('--model', type=str, default=None,
                      help='특정 모델만 시각화 (None이면 TARGET_MODEL 사용, 부분 문자열 매칭)')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                       help='시각화할 데이터셋 리스트')
    parser.add_argument('--single_plot', action='store_true',
                       help='모든 데이터셋을 하나의 플롯으로 통합')
    
    args = parser.parse_args()
    
    # 데이터셋 필터 설정
    dataset_filter = args.datasets if args.datasets else VISUALIZATION_DATASETS

    # summary_single 파일 경로 해석
    summary_file = args.summary_file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if summary_file is None:
        # 기본값: (이 스크립트와 같은 디렉터리) summary_stages-40_p1.0.txt
        summary_file = os.path.join(script_dir, "summary_stages-40_p1.0.txt")
    elif not os.path.isabs(summary_file):
        # 1) 현재 작업 디렉터리 기준 그대로 존재하면 그대로 사용
        if not os.path.exists(summary_file):
            # 2) 스크립트 디렉터리 기준으로 fallback
            summary_file = os.path.join(script_dir, summary_file)

    if not os.path.exists(summary_file):
        print(f"Error: summary file not found: {summary_file}")
        return

    print(f"Summary 파일: {summary_file}")
    all_results = parse_summary_single_file(summary_file)
    if not all_results:
        print("Error: summary에서 데이터를 파싱하지 못했습니다.")
        return

    # 모델 필터
    target_model = args.model if args.model else TARGET_MODEL
    if target_model:
        filtered = {m: d for m, d in all_results.items() if target_model in m}
        if not filtered:
            print(f"Error: summary 내에서 model filter '{target_model}'와 일치하는 모델을 찾지 못했습니다.")
            print("가능한 모델 예시:")
            for m in sorted(all_results.keys())[:20]:
                print(f"  - {m}")
            return
        all_results = filtered
    
    # 출력 경로 설정
    if args.output_file is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        script_name = os.path.splitext(os.path.basename(__file__))[0]
        output_file = os.path.join(script_dir, script_name, f"{script_name}.png")
    else:
        output_file = args.output_file
    
    # 설정 정보 출력
    print(f"\n=== 설정 정보 ===")
    if dataset_filter is not None and len(dataset_filter) > 0:
        print(f"데이터셋 필터: {', '.join(dataset_filter)}")
    else:
        print(f"데이터셋 필터: 모든 데이터셋")
    print(f"모델 필터: {target_model if target_model else 'None(전체)'}")
    
    # 시각화 생성
    print(f"\n그리드 시각화 생성 중...")
    create_grid_visualization(
        all_results,
        output_file,
        dataset_filter=dataset_filter,
        single_plot=args.single_plot,
    )
    
    print("\n완료!")


if __name__ == '__main__':
    main()
