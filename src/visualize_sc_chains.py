#!/usr/bin/env python3
"""
SC Chain 개수에 따른 성능 분포 시각화 스크립트

eval_result.txt 파일을 읽어서 각 task별로:
1. Accuracy vs num_chains plot
2. Total Tokens (input + output) vs num_chains plot
생성
"""

import argparse
import re
import os
import matplotlib.pyplot as plt
import matplotlib
from typing import Dict, List, Tuple

# 한글 폰트 설정 (필요시)
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False


def parse_eval_result(file_path: str) -> Dict[str, Dict]:
    """
    eval_result.txt 파일을 파싱하여 task별 SC 성능 데이터 추출
    
    Returns:
        {
            'task_name': {
                'num_chains': [5, 10, 15, ...],
                'accuracies': [0.7287, 0.7673, ...],
                'input_tokens': [734.50, 1469.00, ...],
                'output_tokens': [57432.72, 115096.24, ...]
            }
        }
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 각 task 섹션을 분리 (Loading JSON file 라인 기준)
    # 패턴: "Loading JSON file: ..." 으로 시작하는 섹션들
    sections = re.split(r'(?=Loading JSON file:)', content)
    
    results = {}
    
    for section in sections:
        # Task 이름 추출
        task_match = re.search(r'Loading JSON file:.*?/([^/]+)\.json', section)
        if not task_match:
            continue
        task_name = task_match.group(1)
        
        # SC Accuracy 라인들 추출
        sc_pattern = r'SC Accuracy\(N=(\d+), T=\d+\): ([\d.]+) \(avg_input_tokens=([\d.]+), avg_output_tokens=([\d.]+)\)'
        matches = re.findall(sc_pattern, section)
        
        if not matches:
            continue
        
        num_chains = []
        accuracies = []
        input_tokens = []
        output_tokens = []
        
        for match in matches:
            n = int(match[0])
            acc = float(match[1])
            inp_tok = float(match[2])
            out_tok = float(match[3])
            
            num_chains.append(n)
            accuracies.append(acc)
            input_tokens.append(inp_tok)
            output_tokens.append(out_tok)
        
        if num_chains:
            results[task_name] = {
                'num_chains': num_chains,
                'accuracies': accuracies,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens
            }
    
    return results


def create_visualizations(results: Dict[str, Dict], output_dir: str = '.'):
    """
    각 task별로 Accuracy와 Token consumption plot 생성
    
    Args:
        results: parse_eval_result의 반환값
        output_dir: 출력 디렉토리
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Task 이름을 정렬 (일관된 순서를 위해)
    task_names = sorted(results.keys())
    
    # 색상 팔레트 (첨부한 이미지와 유사하게)
    colors = {
        'AIME_2024': '#2E86AB',      # Blue
        'AIME_2025': '#A23B72',      # Purple
        'GPQA-Diamond': '#F18F01',   # Orange
        'GSM8K': '#C73E1D',          # Red
        'MATH500_test': '#6A994E'    # Green
    }
    
    # 각 task별로 2개의 figure 생성
    for task_name in task_names:
        data = results[task_name]
        num_chains = data['num_chains']
        accuracies = data['accuracies']
        input_tokens = data['input_tokens']
        output_tokens = data['output_tokens']
        total_tokens = [inp + out for inp, out in zip(input_tokens, output_tokens)]
        
        color = colors.get(task_name, '#000000')
        
        # Figure 1: Accuracy vs num_chains
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        ax1.plot(num_chains, accuracies, marker='o', linewidth=2, markersize=8, 
                color=color, label=task_name)
        ax1.set_xlabel('Number of Chains (N)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax1.set_title(f'SC Accuracy vs Number of Chains - {task_name}', 
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(fontsize=10)
        ax1.set_xticks(num_chains)
        ax1.set_ylim([0, 1.0])
        
        # 값 표시
        for i, (n, acc) in enumerate(zip(num_chains, accuracies)):
            ax1.annotate(f'{acc:.3f}', (n, acc), 
                        textcoords="offset points", xytext=(0,10), 
                        ha='center', fontsize=9)
        
        plt.tight_layout()
        accuracy_file = os.path.join(output_dir, f'{task_name}_accuracy.png')
        plt.savefig(accuracy_file, dpi=300, bbox_inches='tight')
        plt.close(fig1)
        print(f"Saved: {accuracy_file}")
        
        # Figure 2: Total Tokens vs num_chains
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.plot(num_chains, total_tokens, marker='s', linewidth=2, markersize=8,
                color=color, label=task_name)
        ax2.set_xlabel('Number of Chains (N)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Total Tokens (Input + Output)', fontsize=12, fontweight='bold')
        ax2.set_title(f'Total Token Consumption vs Number of Chains - {task_name}',
                     fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.legend(fontsize=10)
        ax2.set_xticks(num_chains)
        ax2.set_yscale('log')  # 로그 스케일 (첨부 이미지와 유사하게)
        
        # 값 표시
        for i, (n, tok) in enumerate(zip(num_chains, total_tokens)):
            ax2.annotate(f'{tok:.0f}', (n, tok), 
                        textcoords="offset points", xytext=(0,10), 
                        ha='center', fontsize=9)
        
        plt.tight_layout()
        token_file = os.path.join(output_dir, f'{task_name}_tokens.png')
        plt.savefig(token_file, dpi=300, bbox_inches='tight')
        plt.close(fig2)
        print(f"Saved: {token_file}")
    
    # 모든 task를 하나의 figure에 합친 버전도 생성 (선택사항)
    # Accuracy 비교
    fig_all_acc, ax_all_acc = plt.subplots(figsize=(10, 6))
    for task_name in task_names:
        data = results[task_name]
        num_chains = data['num_chains']
        accuracies = data['accuracies']
        color = colors.get(task_name, '#000000')
        ax_all_acc.plot(num_chains, accuracies, marker='o', linewidth=2, 
                       markersize=6, color=color, label=task_name)
    
    ax_all_acc.set_xlabel('Number of Chains (N)', fontsize=12, fontweight='bold')
    ax_all_acc.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax_all_acc.set_title('SC Accuracy vs Number of Chains (All Tasks)', 
                        fontsize=14, fontweight='bold')
    ax_all_acc.grid(True, alpha=0.3, linestyle='--')
    ax_all_acc.legend(fontsize=10, loc='best')
    ax_all_acc.set_xticks([5, 10, 15, 20, 25, 30, 35, 40])
    ax_all_acc.set_ylim([0, 1.0])
    plt.tight_layout()
    all_acc_file = os.path.join(output_dir, 'all_tasks_accuracy.png')
    plt.savefig(all_acc_file, dpi=300, bbox_inches='tight')
    plt.close(fig_all_acc)
    print(f"Saved: {all_acc_file}")
    
    # Token 비교
    fig_all_tok, ax_all_tok = plt.subplots(figsize=(10, 6))
    for task_name in task_names:
        data = results[task_name]
        num_chains = data['num_chains']
        input_tokens = data['input_tokens']
        output_tokens = data['output_tokens']
        total_tokens = [inp + out for inp, out in zip(input_tokens, output_tokens)]
        color = colors.get(task_name, '#000000')
        ax_all_tok.plot(num_chains, total_tokens, marker='s', linewidth=2,
                       markersize=6, color=color, label=task_name)
    
    ax_all_tok.set_xlabel('Number of Chains (N)', fontsize=12, fontweight='bold')
    ax_all_tok.set_ylabel('Total Tokens (Input + Output)', fontsize=12, fontweight='bold')
    ax_all_tok.set_title('Total Token Consumption vs Number of Chains (All Tasks)',
                         fontsize=14, fontweight='bold')
    ax_all_tok.grid(True, alpha=0.3, linestyle='--')
    ax_all_tok.legend(fontsize=10, loc='best')
    ax_all_tok.set_xticks([5, 10, 15, 20, 25, 30, 35, 40])
    ax_all_tok.set_yscale('log')
    plt.tight_layout()
    all_tok_file = os.path.join(output_dir, 'all_tasks_tokens.png')
    plt.savefig(all_tok_file, dpi=300, bbox_inches='tight')
    plt.close(fig_all_tok)
    print(f"Saved: {all_tok_file}")


def main():
    parser = argparse.ArgumentParser(
        description='SC Chain 개수에 따른 성능 분포 시각화'
    )
    parser.add_argument(
        '--eval_result_file',
        type=str,
        required=True,
        help='eval_result.txt 파일 경로'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='출력 디렉토리 (기본값: rsc_project/visualization)'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.eval_result_file):
        raise FileNotFoundError(f"File not found: {args.eval_result_file}")
    
    print(f"Parsing {args.eval_result_file}...")
    results = parse_eval_result(args.eval_result_file)
    
    if not results:
        print("No SC data found in the file.")
        return
    
    print(f"Found {len(results)} tasks: {', '.join(results.keys())}")
    # 출력 디렉토리 설정: 기본은 rsc_project/visualization
    if args.output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        current_dir = script_dir
        rsc_project_dir = None
        for _ in range(10):
            if os.path.basename(current_dir) == 'rsc_project':
                rsc_project_dir = current_dir
                break
            parent = os.path.dirname(current_dir)
            if parent == current_dir:
                break
            current_dir = parent
        if rsc_project_dir is None:
            rsc_project_dir = os.path.join(os.path.dirname(script_dir), '..', 'rsc_project')
            rsc_project_dir = os.path.abspath(rsc_project_dir)
        visualization_dir = os.path.join(rsc_project_dir, 'visualization')
        os.makedirs(visualization_dir, exist_ok=True)
        output_dir = visualization_dir
    else:
        output_dir = args.output_dir

    print(f"\nCreating visualizations...")
    create_visualizations(results, output_dir)
    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == '__main__':
    main()

