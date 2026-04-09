"""
SC(N) 성능 변화 시각화: N=1부터 N=39까지 랜덤 샘플링

각 N값에 대해:
- Reasoning Paths를 랜덤 샘플링하여 N개 선택
- SC accuracy 계산 (majority vote)
- 사용된 토큰 수 계산 (input_tokens + sum of sampled output tokens)
- N이 증가할수록 성능 변화 확인
"""

import json
import os
import sys
import random
import argparse
from collections import Counter
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# 프로젝트 루트 경로 추가 (rsc_project 디렉토리)
script_dir = os.path.dirname(os.path.abspath(__file__))
# rsc_project 디렉토리 찾기 (현재 위치에서 상위로 올라가며 찾기)
current_dir = script_dir
rsc_project_dir = None
for _ in range(10):  # 최대 10단계 상위로 검색
    if os.path.basename(current_dir) == 'rsc_project' and os.path.exists(os.path.join(current_dir, 'evaluation')):
        rsc_project_dir = current_dir
        break
    parent = os.path.dirname(current_dir)
    if parent == current_dir:  # 루트에 도달
        break
    current_dir = parent

# rsc_project를 찾지 못한 경우 절대 경로 사용
if rsc_project_dir is None:
    # 기본 경로: /home/albert0811/dev_env/git/repos/RASC/rsc_project
    rsc_project_dir = '/home/albert0811/dev_env/git/repos/RASC/rsc_project'
    if not os.path.exists(rsc_project_dir):
        # 상대 경로 시도
        rsc_project_dir = os.path.join(os.path.dirname(script_dir), '../../rsc_project')
        rsc_project_dir = os.path.abspath(rsc_project_dir)

if rsc_project_dir and os.path.exists(rsc_project_dir):
    if rsc_project_dir not in sys.path:
        sys.path.insert(0, rsc_project_dir)

from evaluation.metrics import extract_cot_data, normalize_answer_eval, prepare_samples, extract_pool_size_from_stages_config


def _sample_subset_for_trial(answers: List[str],
                            tokens: List[int],
                            subset_n: int,
                            *,
                            seed: int) -> Tuple[List[str], List[int]]:
    """
    answers/tokens에서 subset_n개를 (복원 없이) 랜덤 샘플링.
    evaluation/metrics.py의 _sample_subset_for_trial과 동일한 로직.
    
    Args:
        answers: 답변 리스트
        tokens: 토큰 리스트
        subset_n: 샘플링할 개수
        seed: 랜덤 시드
        
    Returns:
        Tuple[List[str], List[int]]: (샘플링된 답변 리스트, 샘플링된 토큰 리스트)
    """
    if subset_n is None:
        return answers, tokens
    if not answers:
        return answers, tokens
    if subset_n <= 0:
        return [], []
    if len(answers) <= subset_n:
        return answers, tokens

    random.seed(seed)
    idxs = random.sample(range(len(answers)), subset_n)
    sub_answers = [answers[i] for i in idxs]
    sub_tokens = [tokens[i] if i < len(tokens) else 0 for i in idxs]
    return sub_answers, sub_tokens

# ============================================================================
# 시각화에 사용할 데이터셋 리스트 (원하는 데이터셋만 추가/제거)
# ============================================================================
# None이면 모든 데이터셋 사용, 리스트가 비어있으면 모든 데이터셋 사용
# 예시: ['AMC23', 'GSM8K', 'MATH500_test'] - 이 데이터셋들만 시각화
VISUALIZATION_DATASETS = ['GSM8K', 'MATH500_test', 'AMC23', 'AIME_2024', 'AIME_2025']  # None이면 모든 데이터셋, 또는 ['AMC23', 'GSM8K', ...] 형태로 지정

# 데이터셋 이름 매핑 (시각화용 표시 이름)
DATASET_DISPLAY_NAMES = {
    'MATH500_test': 'MATH500',
    'AIME_2024': 'AIME2024',
    'AIME_2025': 'AIME2025',
    # 다른 데이터셋 이름도 필요시 여기에 추가
}
# ============================================================================

# ============================================================================
# 시각화에 사용할 모델 리스트 (원하는 모델만 추가/제거)
# ============================================================================
# - 값은 Stage1 디렉토리 경로의 일부 또는 전체와 매칭되는 문자열입니다.
#   예: 'Qwen2.5-7B-Instruct', 'Qwen/Qwen2.5-7B-Instruct', 'gemma-3-4b-it'
# - None이면 모든 모델 사용, 리스트가 비어있으면 모든 모델 사용
VISUALIZATION_MODELS = ['Qwen2.5-3B-Instruct', 'Qwen2.5-7B-Instruct', 'Qwen2.5-14B-Instruct', "gemma-3-4b-it"]  # None이면 모든 모델, 또는 ['Qwen2.5-7B-Instruct', 'gemma-3-4b-it', ...] 형태로 지정
# ============================================================================


def calculate_sc_accuracy_sampled(answers: List[str], ground_truth: str) -> bool:
    """
    샘플링된 답변들로 SC accuracy 계산 (majority vote)
    
    evaluation/metrics.py의 calculate_sc 함수와 동일한 로직 사용
    
    Args:
        answers: 샘플링된 답변 리스트
        ground_truth: 정답
        
    Returns:
        bool: majority answer가 정답과 일치하는지 여부
    """
    if not answers:
        return False
    
    # 정규화된 답변으로 Counter를 만들어 majority voting (evaluation/metrics.py와 동일)
    normalized_answers = [normalize_answer_eval(ans) for ans in answers]
    most_common_normalized = Counter(normalized_answers).most_common(1)[0][0]
    
    # 정답과 비교
    return most_common_normalized == normalize_answer_eval(ground_truth)


def analyze_sample_at_n(sample: Dict[str, Any], n: int, num_trials: int = 10) -> Dict[str, Any]:
    """
    단일 샘플에 대해 N개의 reasoning path를 랜덤 샘플링하여 SC 결과 계산
    
    Args:
        sample: 샘플 데이터
        n: 샘플링할 reasoning path 개수 (1~39)
        num_trials: 반복 횟수 (여러 번 샘플링하여 평균 계산)
        
    Returns:
        Dict: 분석 결과
            - accuracies: 각 trial의 accuracy 리스트
            - avg_accuracy: 평균 accuracy
            - token_counts: 각 trial의 토큰 수 리스트
            - avg_tokens: 평균 토큰 수
    """
    reasoning_paths = sample.get('Reasoning Paths', [])
    ground_truth = str(sample.get('answer', '')).strip()
    input_tokens = sample.get('input_tokens', 0)
    
    # Reasoning Paths에서 답변과 토큰 추출
    answers, tokens = extract_cot_data(reasoning_paths)
    
    if not answers or len(answers) < n:
        # N개보다 적은 경우, 모든 path 사용
        available_n = len(answers)
        if available_n == 0:
            return {
                'accuracies': [False] * num_trials,
                'avg_accuracy': 0.0,
                'token_counts': [input_tokens] * num_trials,
                'avg_tokens': float(input_tokens)
            }
        n = available_n
    
    accuracies = []
    token_counts = []
    
    for _ in range(num_trials):
        # N개 랜덤 샘플링
        if n == len(answers):
            # 모든 path 사용
            sampled_indices = list(range(len(answers)))
        else:
            sampled_indices = random.sample(range(len(answers)), n)
        
        sampled_answers = [answers[i] for i in sampled_indices]
        sampled_tokens = [tokens[i] for i in sampled_indices]
        
        # SC accuracy 계산
        is_correct = calculate_sc_accuracy_sampled(sampled_answers, ground_truth)
        accuracies.append(is_correct)
        
        # 토큰 수 계산: input_tokens + sum of sampled output tokens
        total_tokens = input_tokens + sum(sampled_tokens)
        token_counts.append(total_tokens)
    
    return {
        'accuracies': accuracies,
        'avg_accuracy': float(np.mean(accuracies)),
        'token_counts': token_counts,
        'avg_tokens': float(np.mean(token_counts))
    }


def get_level_value(sample: Dict[str, Any]) -> Any:
    """
    샘플에서 level 값 추출
    
    Args:
        sample: 샘플 데이터
        
    Returns:
        level 값 (없으면 None)
    """
    metadata = sample.get('metadata', {})
    level = metadata.get('level', None)
    if level is not None:
        return level
    
    # 최상위 레벨에서도 확인
    level = sample.get('level', None)
    return level


def calculate_accuracy_curve(json_path: str, num_trials: int = 10, seed: int = 42, pool_size: int = None, subset_n: int = 40) -> Dict[int, float]:
    """
    N=1~40까지의 정확도 곡선 F(N) 계산 (부트스트래핑 적용)
    항상 N=1부터 N=40까지 고정으로 계산
    
    Args:
        json_path: JSON 파일 경로
        num_trials: 부트스트래핑 반복 횟수 (기본값: 10)
        seed: 랜덤 시드 (기본값: 42)
        pool_size: 사용할 reasoning chain 풀 크기 (None이면 JSON 내 전체 answers 사용)
        subset_n: pool_size 내에서 먼저 랜덤 샘플링할 체인 수 (기본값: 40, metrics.py와 동일)
        
    Returns:
        Dict[int, float]: {N: accuracy} 딕셔너리
    """
    # JSON 파일 읽기
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    # prepare_samples로 변환
    samples = prepare_samples(json_data)
    
    # 각 N에 대해 정확도 계산 (N=1부터 N=40까지 고정)
    accuracy_curve = {}
    min_n = 1
    max_n = 40
    
    for n in range(min_n, max_n + 1):
        accuracy = calculate_sc_accuracy_for_n(samples, n, num_trials=num_trials, seed=seed, pool_size=pool_size, subset_n=subset_n)
        accuracy_curve[n] = accuracy
    
    return accuracy_curve


def find_best_n(accuracy_curve: Dict[int, float]) -> Tuple[Optional[int], float]:
    """
    정확도 곡선에서 Best N* 찾기 (전체 N 중 최고 정확도 지점).
    최고 정확도 동률이면 더 작은 N을 선택.
    """
    if not accuracy_curve:
        return None, 0.0
    best_accuracy = max(accuracy_curve.values())
    best_n = min([n for n, acc in accuracy_curve.items() if acc == best_accuracy])
    return best_n, best_accuracy


def find_optimal_n(accuracy_curve: Dict[int, float]) -> Optional[int]:
    """
    정확도 곡선에서 Optimal N* 찾기:
    N이 증가하며 성능이 처음 떨어지는 지점의 직전 N.
    (즉, first drop 직전 지점)

    - 정확도 값이 끝까지 떨어지지 않으면 Best N*를 사용.
    """
    if not accuracy_curve:
        return None

    sorted_ns = sorted(accuracy_curve.keys())
    if len(sorted_ns) == 1:
        return sorted_ns[0]

    for i in range(len(sorted_ns) - 1):
        n_cur = sorted_ns[i]
        n_next = sorted_ns[i + 1]
        if accuracy_curve[n_next] < accuracy_curve[n_cur]:
            return n_cur

    best_n, _ = find_best_n(accuracy_curve)
    return best_n


def count_accuracy_decreases(accuracy_curve: Dict[int, float], threshold: float) -> int:
    """
    N이 증가함에 따라 accuracy가 하락하는 횟수 계산
    (감소량이 threshold보다 큰 경우만 카운트)
    
    Args:
        accuracy_curve: {N: accuracy} 딕셔너리
        threshold: 하락 임계값 (예: 0.001 = 0.1%p, 0.015 = 1.5%p)
    
    Returns:
        num_decrease: 하락 횟수
    """
    if not accuracy_curve:
        return 0
    
    sorted_ns = sorted(accuracy_curve.keys())
    num_decrease = 0
    
    for i in range(len(sorted_ns) - 1):
        n_cur = sorted_ns[i]
        n_next = sorted_ns[i + 1]
        acc_cur = accuracy_curve.get(n_cur, 0.0)
        acc_next = accuracy_curve.get(n_next, 0.0)
        diff = acc_next - acc_cur
        # 감소량이 threshold보다 큰 경우만 카운트
        if diff < -threshold:
            num_decrease += 1
    
    return num_decrease


def calculate_sc_accuracy_for_n(samples: List[Dict[str, Any]], n: int, num_trials: int = 10, seed: int = 42,
                                pool_size: int = None, subset_n: int = 40) -> float:
    """
    N개의 reasoning paths를 사용하여 SC 정확도 계산 (부트스트래핑 적용)
    evaluation/metrics.py의 calculate_sc와 동일한 샘플링/시드 로직을 N에 대해 일반화한 버전.
    
    Args:
        samples: prepare_samples로 변환된 샘플 리스트
        n: 사용할 reasoning path 개수 (1~40)
        num_trials: 부트스트래핑 반복 횟수 (기본값: 10)
        seed: 랜덤 시드 (기본값: 42)
        pool_size: 사용할 reasoning chain 풀 크기 (None이면 JSON 내 전체 answers 사용)
        subset_n: pool_size 내에서 먼저 랜덤 샘플링할 체인 수 (기본값: 40, metrics.py와 동일)
        
    Returns:
        float: SC 정확도 (0.0 ~ 1.0, 여러 trial의 평균)
    """
    trial_accuracies = []
    
    for trial in range(num_trials):
        correct_flags = []
        
        for sample in samples:
            answers = sample['cot_answers']
            tokens = sample.get('cot_tokens', [])
            correct = sample['correct_answer']
            
            if not answers:
                flag = 0
            else:
                # 풀 크기 설정 (metrics.py의 calculate_sc와 동일)
                if pool_size is None:
                    available_answers = answers
                    available_tokens = tokens
                else:
                    available_answers = answers[:pool_size]
                    available_tokens = tokens[:pool_size] if tokens else []
                
                # AC/ESC와 동일한 방식: pool에서 subset_n(40)개를 먼저 샘플링
                # metrics.py의 calculate_sc와 동일한 seed 규칙 (seed + trial)
                subset_seed = seed + trial
                subset_answers, subset_tokens = _sample_subset_for_trial(
                    available_answers, available_tokens, subset_n,
                    seed=subset_seed
                )
                
                # subset 내에서 앞에서부터 n개를 순서대로 사용 (SC 알고리즘)
                # metrics.py의 calculate_sc와 동일: subset만 랜덤 샘플링하고, subset 내에서는 순서대로 사용
                if len(subset_answers) <= n:
                    # subset 크기가 n 이하면 모두 사용
                    sampled_answers = subset_answers
                else:
                    # subset에서 앞에서부터 n개를 순서대로 사용 (랜덤 샘플링 없음)
                    sampled_answers = subset_answers[:n]
                
                # 정규화된 답변으로 Counter를 만들어 majority voting
                normalized_answers = [normalize_answer_eval(ans) for ans in sampled_answers]
                if not normalized_answers:
                    flag = 0
                else:
                    most_common_normalized = Counter(normalized_answers).most_common(1)[0][0]
                    flag = 1 if most_common_normalized == normalize_answer_eval(correct) else 0
            
            correct_flags.append(flag)
        
        # 이번 trial의 정확도
        trial_accuracy = sum(correct_flags) / len(samples) if samples else 0.0
        trial_accuracies.append(trial_accuracy)
    
    # 모든 trial의 평균
    accuracy = sum(trial_accuracies) / len(trial_accuracies) if trial_accuracies else 0.0
    return accuracy


def analyze_dataset_scaling(json_path: str, dataset_name: str, 
                           min_n: int = 1, max_n: int = 39, 
                           num_trials: int = 10,
                           output_dir: str = None,
                           level: int = None,
                           seed: int = 42) -> Dict[str, Any]:
    """
    데이터셋 전체에 대해 N=1부터 N=39까지 SC 성능 분석
    find_optimal_n.py의 부트스트래핑 방식 사용
    
    Args:
        json_path: Stage1 JSON 파일 경로
        dataset_name: 데이터셋 이름
        min_n: 최소 N값
        max_n: 최대 N값
        num_trials: 각 N에 대한 반복 횟수 (부트스트래핑)
        output_dir: 결과 저장 디렉토리
        level: 특정 level만 분석 (None이면 전체)
        seed: 랜덤 시드
        
    Returns:
        Dict: 분석 결과
    """
    print(f"\n{'='*60}")
    if level is not None:
        print(f"Analyzing SC scaling: {dataset_name} (Level {level})")
    else:
        print(f"Analyzing SC scaling: {dataset_name}")
    print(f"JSON file: {json_path}")
    print(f"N range: {min_n} ~ {max_n}, Trials per N: {num_trials}")
    print(f"{'='*60}\n")
    
    # JSON 파일 읽기
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    # prepare_samples로 변환
    all_samples = prepare_samples(json_data)
    
    # Level 필터링 (level이 있는 경우)
    if level is not None:
        # 원본 데이터에서 level 필터링
        filtered_data = [s for s in json_data if get_level_value(s) == level]
        if not filtered_data:
            print(f"Warning: No samples found for level {level}")
            return None
        # 필터링된 데이터를 다시 prepare_samples로 변환
        samples = prepare_samples(filtered_data)
    else:
        samples = all_samples
    
    print(f"Total samples: {len(samples)}")
    
    # 각 N값에 대해 정확도 계산 (find_optimal_n.py 방식)
    n_values = list(range(min_n, max_n + 1))
    summary = {}
    
    for n in n_values:
        if (n - min_n + 1) % 5 == 0 or n == min_n or n == max_n:
            print(f"Calculating accuracy for N={n}...")
        
        # 부트스트래핑을 통한 정확도 계산
        accuracy = calculate_sc_accuracy_for_n(samples, n, num_trials=num_trials, seed=seed)
        
        # 토큰 수 계산은 기존 방식 유지 (평균 토큰 수 추정)
        # 실제 토큰 수는 샘플별로 다를 수 있으므로 근사값 사용
        avg_tokens = 0.0
        if samples:
            # 첫 번째 샘플의 input_tokens와 평균 output_tokens 추정
            sample = json_data[0] if level is None else filtered_data[0]
            input_tokens = sample.get('input_tokens', 0)
            reasoning_paths = sample.get('Reasoning Paths', [])
            if reasoning_paths:
                _, tokens = extract_cot_data(reasoning_paths)
                avg_output_tokens = np.mean(tokens) if tokens else 0
                avg_tokens = input_tokens + avg_output_tokens * n
            else:
                avg_tokens = input_tokens
        
        summary[n] = {
            'accuracy': accuracy,
            'accuracy_std': 0.0,  # 부트스트래핑 평균이므로 std는 별도 계산 필요
            'tokens': avg_tokens,
            'tokens_std': 0.0
        }
    
    # 시각화 (output_dir가 None이면 저장하지 않음)
    if output_dir is None:
        # 그래프 생성하지 않고 통계만 반환
        return {
            'dataset_name': dataset_name,
            'total_samples': len(samples),
            'summary': summary
        }
    
    # 파일명에 level 정보 추가
    if level is not None:
        output_file = os.path.join(output_dir, f'{dataset_name}_level{level}_sc_scaling.png')
        text_output_file = os.path.join(output_dir, f'{dataset_name}_level{level}_sc_scaling.txt')
        title_suffix = f' (Level {level})'
    else:
        output_file = os.path.join(output_dir, f'{dataset_name}_sc_scaling.png')
        text_output_file = os.path.join(output_dir, f'{dataset_name}_sc_scaling.txt')
        title_suffix = ''
    
    # 그래프 생성
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    n_list = list(n_values)
    accuracies = [summary[n]['accuracy'] for n in n_list]
    accuracy_stds = [summary[n]['accuracy_std'] for n in n_list]
    tokens = [summary[n]['tokens'] for n in n_list]
    token_stds = [summary[n]['tokens_std'] for n in n_list]
    
    # Accuracy 그래프
    ax1.plot(n_list, accuracies, 'o-', color='#2E86AB', linewidth=2, markersize=6, label='SC Accuracy')
    ax1.fill_between(n_list, 
                     [a - s for a, s in zip(accuracies, accuracy_stds)],
                     [a + s for a, s in zip(accuracies, accuracy_stds)],
                     alpha=0.2, color='#2E86AB')
    ax1.set_xlabel('N (Number of Reasoning Chains)', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title(f'SC Accuracy vs N: {dataset_name}{title_suffix}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_ylim([0, 1.05])
    
    # Token 그래프
    ax2.plot(n_list, tokens, 's-', color='#A23B72', linewidth=2, markersize=6, label='Total Tokens')
    ax2.fill_between(n_list,
                     [t - s for t, s in zip(tokens, token_stds)],
                     [t + s for t, s in zip(tokens, token_stds)],
                     alpha=0.2, color='#A23B72')
    ax2.set_xlabel('N (Number of Reasoning Chains)', fontsize=12)
    ax2.set_ylabel('Total Tokens', fontsize=12)
    ax2.set_title(f'Token Usage vs N: {dataset_name}{title_suffix}', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nVisualization saved to: {output_file}")
    
    # 텍스트 결과 저장
    with open(text_output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        if level is not None:
            f.write(f"SC Scaling Analysis: {dataset_name} (Level {level})\n")
        else:
            f.write(f"SC Scaling Analysis: {dataset_name}\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total samples: {len(samples)}\n")
        f.write(f"N range: {min_n} ~ {max_n}\n")
        f.write(f"Trials per N: {num_trials}\n\n")
        f.write("-"*80 + "\n")
        f.write(f"{'N':<5} {'Accuracy':<12} {'Accuracy (std)':<15} {'Tokens':<15} {'Tokens (std)':<15}\n")
        f.write("-"*80 + "\n")
        
        for n in n_list:
            acc = summary[n]['accuracy']
            acc_std = summary[n]['accuracy_std']
            tok = summary[n]['tokens']
            tok_std = summary[n]['tokens_std']
            f.write(f"{n:<5} {acc:<12.4f} {acc_std:<15.2f} {tok:<15.0f} {tok_std:<15.0f}\n")
    
    print(f"Text results saved to: {text_output_file}")
    
    return {
        'dataset_name': dataset_name,
        'total_samples': len(samples),
        'summary': summary
    }


def analyze_multiple_tasks(stage1_dir: str,
                           task_names: List[str] = None,
                           min_n: int = 1, max_n: int = 39,
                           num_trials: int = 10,
                           output_dir: str = None,
                           seed: int = 42) -> Dict[str, Dict[str, Any]]:
    """
    여러 task에 대해 SC scaling 분석을 수행하고 하나의 그래프에 표시
    
    Args:
        stage1_dir: Stage1 결과 디렉토리 경로
        task_names: 분석할 task 이름 리스트 (None이면 모든 JSON 파일)
        min_n: 최소 N값
        max_n: 최대 N값
        num_trials: 각 N에 대한 반복 횟수
        output_dir: 결과 저장 디렉토리
        
    Returns:
        Dict: task별 분석 결과
    """
    # JSON 파일 찾기
    if task_names is None:
        # VISUALIZATION_DATASETS가 지정되어 있으면 그것만 사용, 없으면 모든 파일
        if VISUALIZATION_DATASETS is not None and len(VISUALIZATION_DATASETS) > 0:
            json_files = [f"{task}.json" for task in VISUALIZATION_DATASETS]
            # 실제 존재하는 파일만 필터링
            json_files = [f for f in json_files if os.path.exists(os.path.join(stage1_dir, f))]
        else:
            json_files = [f for f in os.listdir(stage1_dir) if f.endswith('.json')]
            # stages-로 시작하는 파일 제외 (중간 단계 파일)
            json_files = [f for f in json_files if not os.path.basename(f).startswith('DEEPMATH_2600_stages-')]
    else:
        json_files = [f"{task}.json" for task in task_names]
    
    print(f"\n{'='*60}")
    print(f"Analyzing multiple tasks: {len(json_files)} tasks")
    print(f"Tasks: {', '.join([os.path.splitext(f)[0] for f in json_files])}")
    print(f"N range: {min_n} ~ {max_n}, Trials per N: {num_trials}")
    print(f"{'='*60}\n")
    
    # 각 task별로 분석
    results_by_task = {}
    
    # 각 데이터셋별로 명확하게 구분되는 색상 정의
    task_colors = {
        'AIME_2024': '#E63946',      # 빨간색 (Red)
        'AIME_2025': '#F77F00',      # 주황색 (Orange)
        'AMC23': '#FFB703',          # 노란색 (Yellow)
        'GPQA-Diamond': '#2A9D8F',    # 청록색 (Teal)
        'GSM8K': '#7209B7',          # 보라색 (Purple)
        'MATH500_test': '#06FFA5',    # 민트색 (Mint)
        'MATH2500': '#8338EC',       # 보라빨강 (Violet)
        'MMLU-Pro': '#219EBC',        # 파란색 (Blue)
    }
    
    # 기본 색상 팔레트 (task_colors에 없는 경우 사용)
    default_colors = ['#E63946', '#2A9D8F', '#F77F00', '#7209B7', '#FFB703', 
                      '#06FFA5', '#8338EC', '#219EBC', '#FB8500', '#023047']
    
    for i, json_file in enumerate(json_files):
        task_name = os.path.splitext(json_file)[0]
        json_path = os.path.join(stage1_dir, json_file)
        
        if not os.path.exists(json_path):
            print(f"Warning: File not found: {json_path}, skipping...")
            continue
        
        print(f"\nProcessing {task_name} ({i+1}/{len(json_files)})...")
        
        try:
            # 각 task에 대해 분석 (요약만 계산)
            result = analyze_dataset_scaling(
                json_path=json_path,
                dataset_name=task_name,
                min_n=min_n,
                max_n=max_n,
                num_trials=num_trials,
                output_dir=None,  # 개별 파일 저장 안 함
                level=None,
                seed=seed
            )
            
            if result:
                results_by_task[task_name] = result
                print(f"  ✓ {task_name}: {result['total_samples']} samples")
        except Exception as e:
            print(f"  ✗ Error analyzing {task_name}: {e}")
            continue
    
    if not results_by_task:
        print("No valid results found!")
        return {}
    
    # 통합 시각화 생성
    if output_dir is None:
        output_dir = stage1_dir
    
    output_file = os.path.join(output_dir, 'all_tasks_sc_scaling.png')
    
    # 그래프 생성
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    n_list = list(range(min_n, max_n + 1))
    
    # VISUALIZATION_DATASETS 순서에 따라 task 정렬
    if VISUALIZATION_DATASETS is not None and len(VISUALIZATION_DATASETS) > 0:
        # VISUALIZATION_DATASETS 순서대로 정렬된 task 리스트 생성
        ordered_tasks = []
        for dataset in VISUALIZATION_DATASETS:
            if dataset in results_by_task:
                ordered_tasks.append(dataset)
        # VISUALIZATION_DATASETS에 없는 나머지 task 추가
        for task_name in results_by_task.keys():
            if task_name not in ordered_tasks:
                ordered_tasks.append(task_name)
    else:
        # VISUALIZATION_DATASETS가 없으면 기존 순서 유지
        ordered_tasks = list(results_by_task.keys())
    
    # 각 task별로 accuracy 플롯 (VISUALIZATION_DATASETS 순서대로)
    for i, task_name in enumerate(ordered_tasks):
        if task_name not in results_by_task:
            continue
        result = results_by_task[task_name]
        summary = result['summary']
        accuracies = [summary[n]['accuracy'] for n in n_list]
        
        # task_colors에 정의된 색상 사용, 없으면 default_colors 사용
        color = task_colors.get(task_name, default_colors[i % len(default_colors)])
        
        ax.plot(n_list, accuracies, 'o-', color=color, linewidth=2, 
                markersize=5, label=task_name, alpha=0.8)
    
    ax.set_xlabel('Number of Chains (N)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('SC Accuracy vs Number of Chains (All Tasks)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='best')
    ax.set_ylim([0, 1.05])
    ax.set_xlim([min_n - 1, max_n + 1])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n{'='*60}")
    print(f"Combined visualization saved to: {output_file}")
    print(f"{'='*60}\n")
    
    return results_by_task


def _create_visualization_from_results(results_by_task: Dict[str, Dict[str, Any]], 
                                       output_dir: str, output_file: str = None):
    """
    계산된 results_by_task를 사용하여 시각화 생성
    항상 N=1부터 N=40까지 고정으로 시각화
    
    Args:
        results_by_task: task별 분석 결과
        output_dir: 결과 저장 디렉토리
        output_file: 출력 파일 경로 (None이면 기본값 사용)
    """
    if not results_by_task:
        return
    
    if output_file is None:
        output_file = os.path.join(output_dir, 'all_tasks_sc_scaling.png')
    
    # 그래프 생성
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # N=1부터 N=40까지 고정
    min_n = 1
    max_n = 40
    n_list = list(range(min_n, max_n + 1))
    
    # VISUALIZATION_DATASETS 순서에 따라 task 정렬
    if VISUALIZATION_DATASETS is not None and len(VISUALIZATION_DATASETS) > 0:
        # VISUALIZATION_DATASETS 순서대로 정렬된 task 리스트 생성
        ordered_tasks = []
        for dataset in VISUALIZATION_DATASETS:
            if dataset in results_by_task:
                ordered_tasks.append(dataset)
        # VISUALIZATION_DATASETS에 없는 나머지 task 추가
        for task_name in results_by_task.keys():
            if task_name not in ordered_tasks:
                ordered_tasks.append(task_name)
    else:
        # VISUALIZATION_DATASETS가 없으면 기존 순서 유지
        ordered_tasks = list(results_by_task.keys())
    
    # 각 task별로 accuracy 플롯 (VISUALIZATION_DATASETS 순서대로)
    task_colors = {
        'AIME_2024': '#E63946',
        'AIME_2025': '#F77F00',
        'AMC23': '#FFB703',
        'GPQA-Diamond': '#2A9D8F',
        'GSM8K': '#7209B7',
        'MATH500_test': '#06FFA5',
        'MATH_train': '#06FFA5',
        'MATH2500': '#8338EC',
        'MMLU-Pro': '#219EBC',
    }
    
    default_colors = ['#E63946', '#2A9D8F', '#F77F00', '#7209B7', '#FFB703', 
                      '#06FFA5', '#8338EC', '#219EBC', '#FB8500', '#023047']
    
    for i, task_name in enumerate(ordered_tasks):
        if task_name not in results_by_task:
            continue
        result = results_by_task[task_name]
        summary = result['summary']
        accuracies = [summary.get(n, {}).get('accuracy', 0.0) for n in n_list]
        
        # task_colors에 정의된 색상 사용, 없으면 default_colors 사용
        color = task_colors.get(task_name, default_colors[i % len(default_colors)])
        
        # 기본 라인 플롯
        line = ax.plot(n_list, accuracies, 'o-', color=color, linewidth=2, 
                markersize=5, label=task_name, alpha=0.8)
        
        # Best/Optimal N 값에 서로 다른 마커 표시
        best_n = result.get('best_n')
        optimal_n = result.get('optimal_n')
        best_accuracy = summary.get(best_n, {}).get('accuracy', 0.0) if best_n is not None else 0.0
        optimal_accuracy = summary.get(optimal_n, {}).get('accuracy', 0.0) if optimal_n is not None else 0.0

        if best_n is not None and optimal_n is not None and best_n == optimal_n and best_accuracy > 0:
            # 같은 지점이면 하나의 마커로 통합 표시
            ax.plot(best_n, best_accuracy, 'X', color=color,
                    markersize=12, markeredgewidth=2.2, markeredgecolor='black',
                    zorder=6, alpha=1.0)
            ax.annotate(
                f'B&O={best_n}',
                xy=(best_n, best_accuracy),
                xytext=(6, 6), textcoords='offset points',
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                          edgecolor='black', alpha=0.85),
                zorder=7
            )
        else:
            if best_n is not None and 1 <= best_n <= 40 and best_accuracy > 0:
                # Best: 별표
                ax.plot(best_n, best_accuracy, '*', color=color,
                        markersize=18, markeredgewidth=2.5, markeredgecolor='black',
                        zorder=6, alpha=1.0)
                ax.annotate(
                    f'B={best_n}',
                    xy=(best_n, best_accuracy),
                    xytext=(6, 8), textcoords='offset points',
                    fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                              edgecolor='black', alpha=0.85),
                    zorder=7
                )

            if optimal_n is not None and 1 <= optimal_n <= 40 and optimal_accuracy > 0:
                # Optimal: 다이아몬드
                ax.plot(optimal_n, optimal_accuracy, 'D', color=color,
                        markersize=9, markeredgewidth=2.0, markeredgecolor='black',
                        markerfacecolor='white', zorder=5, alpha=1.0)
                ax.annotate(
                    f'O={optimal_n}',
                    xy=(optimal_n, optimal_accuracy),
                    xytext=(6, -14), textcoords='offset points',
                    fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                              edgecolor='black', alpha=0.85),
                    zorder=7
                )
    
    ax.set_xlabel('Number of Chains (N)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('SC Accuracy vs Number of Chains (All Tasks)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='best')
    ax.set_ylim([0, 1.05])
    ax.set_xlim([min_n - 1, max_n + 1])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n{'='*60}")
    print(f"Combined visualization saved to: {output_file}")
    print(f"{'='*60}\n")


def parse_report_file(report_path: str) -> Optional[Dict[str, Any]]:
    """
    리포트 파일을 파싱하여 결과 딕셔너리 반환
    
    Args:
        report_path: 리포트 파일 경로
        
    Returns:
        Dict: 파싱된 결과 (None이면 파싱 실패)
    """
    if not os.path.exists(report_path):
        return None
    
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 파일 내용 파싱
        model_path = None
        dataset_name = None
        total_samples = 0
        optimal_n = None
        best_n = None
        best_accuracy = 0.0
        max_accuracy = 0.0  # 하위 호환
        n_1_accuracy = 0.0
        n_40_accuracy = 0.0
        improvement = 0.0
        accuracy_curve = {}
        
        in_accuracy_curve = False
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Model 경로 추출
            if line.startswith('## Model:'):
                model_path = line.replace('## Model:', '').strip()
            
            # Dataset 이름 추출
            elif line.startswith('## Dataset:'):
                dataset_name = line.replace('## Dataset:', '').strip()
            
            # Total Samples 추출
            elif line.startswith('Total Samples:'):
                total_samples = int(line.replace('Total Samples:', '').strip())
            
            # Optimal N* 추출
            elif line.startswith('Optimal N*:'):
                optimal_n = int(line.replace('Optimal N*:', '').strip())

            # Best N* 추출
            elif line.startswith('Best N*:'):
                best_n = int(line.replace('Best N*:', '').strip())
            
            # Best Accuracy 추출 (신규 포맷)
            elif line.startswith('Best Accuracy:'):
                best_accuracy = float(line.replace('Best Accuracy:', '').strip())
                max_accuracy = best_accuracy

            # Max Accuracy 추출 (구버전 포맷 호환)
            elif line.startswith('Max Accuracy:'):
                max_accuracy = float(line.replace('Max Accuracy:', '').strip())
                if best_accuracy == 0.0:
                    best_accuracy = max_accuracy
            
            # N=1 Accuracy 추출
            elif line.startswith('N=1 Accuracy:'):
                n_1_accuracy = float(line.replace('N=1 Accuracy:', '').strip())
            
            # N=40 Accuracy 추출 (또는 N={max_n} 형식)
            elif line.startswith('N=') and 'Accuracy:' in line and not line.startswith('N=1'):
                parts = line.split('Accuracy:')
                if len(parts) == 2:
                    n_40_accuracy = float(parts[1].strip())
            
            # Improvement 추출
            elif line.startswith('Improvement'):
                improvement = float(line.split(':')[-1].strip())
            
            # Accuracy Curve 섹션 시작
            elif 'Accuracy Curve F(N):' in line:
                in_accuracy_curve = True
                continue
            
            # Accuracy Curve 데이터 파싱
            elif in_accuracy_curve:
                # 헤더 라인 스킵
                if 'N' in line and 'Accuracy' in line:
                    continue
                # 구분선 스킵
                elif line.startswith('-') or not line:
                    if line.startswith('-') and len(line) > 10:
                        # 구분선이 나오면 섹션 종료
                        break
                    continue
                else:
                    # N과 Accuracy 값 파싱
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            n = int(parts[0])
                            acc_str = parts[1]
                            # "<-- Optimal" 같은 마커 제거
                            if '<--' in acc_str:
                                acc_str = acc_str.split('<--')[0].strip()
                            accuracy = float(acc_str)
                            accuracy_curve[n] = accuracy
                        except (ValueError, IndexError):
                            continue
        
        if not accuracy_curve:
            return None

        # 누락된 값 보정 (구버전 리포트 호환)
        if best_n is None:
            best_n, best_accuracy_from_curve = find_best_n(accuracy_curve)
            if best_accuracy == 0.0:
                best_accuracy = best_accuracy_from_curve
        if optimal_n is None:
            optimal_n = find_optimal_n(accuracy_curve)
        
        return {
            'model_path': model_path,
            'dataset_name': dataset_name,
            'json_path': None,  # 리포트에서 읽었으므로 json_path는 None
            'accuracy_curve': accuracy_curve,
            'optimal_n': optimal_n,
            'best_n': best_n,
            'best_accuracy': best_accuracy,
            'max_accuracy': max_accuracy if max_accuracy != 0.0 else best_accuracy,  # 하위 호환
            'n_1_accuracy': n_1_accuracy,
            'n_40_accuracy': n_40_accuracy,
            'improvement': improvement,
            'total_samples': total_samples
        }
        
    except Exception as e:
        print(f"    리포트 파일 파싱 오류: {e}")
        return None


def process_single_model_dataset(model_path: str, dataset_name: str, json_path: str,
                                 num_trials: int = 10, seed: int = 42,
                                 results_optimal_dir: str = None, stages_config: str = 'stages-40') -> Dict[str, Any]:
    """
    단일 모델/데이터셋에 대해 정확도 곡선 및 최적 N* 계산 (부트스트래핑 적용)
    기존 리포트 파일이 존재하면 읽어서 사용, 없으면 계산 수행
    항상 N=1부터 N=40까지 고정으로 계산
    
    Args:
        model_path: 모델 경로
        dataset_name: 데이터셋 이름
        json_path: JSON 파일 경로
        num_trials: 부트스트래핑 반복 횟수 (기본값: 10)
        seed: 랜덤 시드 (기본값: 42)
        results_optimal_dir: results_optimal 디렉토리 경로 (기존 리포트 확인용)
        stages_config: stages 설정 (기존 리포트 확인용)
        
    Returns:
        Dict: 계산 결과
    """
    print(f"  처리 중: {model_path} / {dataset_name}")
    
    # 기존 리포트 파일 확인
    if results_optimal_dir:
        # 저장 구조: results_optimal/<model_path>/<stages_config>/<dataset>_optimal_n_report_<num_trials>.txt
        model_output_dir = os.path.join(results_optimal_dir, model_path, stages_config)
        report_path = os.path.join(model_output_dir, f'{dataset_name}_optimal_n_report_{num_trials}.txt')
        
        if os.path.exists(report_path):
            print(f"    기존 리포트 파일 발견: {report_path}")
            print(f"    리포트 파일에서 데이터 읽는 중...")
            parsed_result = parse_report_file(report_path)
            if parsed_result:
                print(f"    ✓ 리포트 파일에서 데이터 로드 완료")
                print(
                    f"    Optimal N*: {parsed_result['optimal_n']}, "
                    f"Best N*: {parsed_result.get('best_n', parsed_result['optimal_n'])}, "
                    f"Best Accuracy: {parsed_result.get('best_accuracy', parsed_result.get('max_accuracy', 0.0)):.4f}"
                )
                print(f"    N=1 정확도: {parsed_result['n_1_accuracy']:.4f}, N=40 정확도: {parsed_result['n_40_accuracy']:.4f}")
                return parsed_result
            else:
                print(f"    ⚠ 리포트 파일 파싱 실패, 재계산 수행...")
    
    # 리포트 파일이 없거나 파싱 실패 시 계산 수행
    try:
        print(f"    정확도 곡선 계산 중 (부트스트래핑 적용, N=1~40)...")
        # 정확도 곡선 계산 (부트스트래핑 적용, N=1~40 고정)
        pool_size = extract_pool_size_from_stages_config(stages_config)
        # metrics.py와 동일하게 subset_n=40 사용
        accuracy_curve = calculate_accuracy_curve(json_path, num_trials=num_trials, seed=seed, pool_size=pool_size, subset_n=40)
        
        # Best/Optimal N* 찾기
        best_n, best_accuracy = find_best_n(accuracy_curve)
        optimal_n = find_optimal_n(accuracy_curve)
        
        # 통계 정보
        max_n = 40
        n_40_accuracy = accuracy_curve.get(max_n, 0.0)
        improvement = best_accuracy - accuracy_curve.get(1, 0.0)
        
        result = {
            'model_path': model_path,
            'dataset_name': dataset_name,
            'json_path': json_path,
            'accuracy_curve': accuracy_curve,
            'optimal_n': optimal_n,
            'best_n': best_n,
            'best_accuracy': best_accuracy,
            'max_accuracy': best_accuracy,  # 하위 호환
            'n_1_accuracy': accuracy_curve.get(1, 0.0),
            'n_40_accuracy': n_40_accuracy,
            'improvement': improvement,
            'total_samples': 0
        }

        # total_samples 계산 (파일 핸들 안전하게 닫기)
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            result['total_samples'] = len(prepare_samples(raw))
        except Exception:
            result['total_samples'] = 0
        
        print(f"    ✓ 계산 완료")
        print(f"    Optimal N*: {optimal_n}, Best N*: {best_n}, Best Accuracy: {best_accuracy:.4f}")
        print(f"    N=1 정확도: {result['n_1_accuracy']:.4f}, N=40 정확도: {n_40_accuracy:.4f}")
        
        return result
        
    except Exception as e:
        print(f"    ✗ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_results(results: List[Dict[str, Any]], output_dir: str, stages_config: str = 'stages-40', num_trials: int = 10):
    """
    결과를 results_optimal 폴더에 저장
    
    Args:
        results: 계산 결과 리스트
        output_dir: 출력 디렉토리
        stages_config: stages 설정 (파일명에 포함)
        num_trials: 부트스트래핑 반복 횟수 (파일명에 포함)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 모델별로 결과 그룹화
    model_results = {}
    for result in results:
        if result is None:
            continue
        
        model_path = result['model_path']
        if model_path not in model_results:
            model_results[model_path] = []
        model_results[model_path].append(result)
    
    # 각 모델별로 처리
    for model_path, model_result_list in model_results.items():
        # 모델 경로 + stages_config에 따라 디렉토리 생성
        # 예: results_optimal/Qwen/Qwen2.5-3B-Instruct/stages-100/
        model_output_dir = os.path.join(output_dir, model_path, stages_config)
        os.makedirs(model_output_dir, exist_ok=True)
        
        # 각 데이터셋별 텍스트 리포트 저장
        for result in model_result_list:
            dataset_name = result['dataset_name']
            
            # 텍스트 리포트 생성
            report_path = os.path.join(model_output_dir, f'{dataset_name}_optimal_n_report_{num_trials}.txt')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=" * 70 + "\n")
                f.write(f"## Model: {model_path}\n")
                f.write(f"## Dataset: {dataset_name}\n")
                f.write("=" * 70 + "\n\n")
                f.write(f"Total Samples: {result['total_samples']}\n\n")
                
                # Accuracy 하락 횟수 통계 (0.1%부터 1.5%까지 모든 threshold별)
                f.write("Num Decreases (Accuracy(N+1) < Accuracy(N)) by Threshold:\n")
                thresholds = [round(0.001 * i, 3) for i in range(1, 21)]  # 0.1%, 0.2%, ..., 2.0% (0.001, 0.002, ..., 0.020)
                for threshold in thresholds:
                    num_decrease = count_accuracy_decreases(result['accuracy_curve'], threshold)
                    threshold_percent = threshold * 100  # 퍼센트로 변환 (0.001 -> 0.1)
                    f.write(f"  Threshold {threshold_percent:.1f}%: {num_decrease}\n")
                f.write("\n")
                
                f.write(f"Optimal N*: {result['optimal_n']}\n")
                f.write(f"Best N*: {result.get('best_n', result['optimal_n'])}\n")
                f.write(f"Best Accuracy: {result.get('best_accuracy', result.get('max_accuracy', 0.0)):.4f}\n\n")
                f.write(f"N=1 Accuracy: {result['n_1_accuracy']:.4f}\n")
                max_n = max(result['accuracy_curve'].keys())
                f.write(f"N={max_n} Accuracy: {result['n_40_accuracy']:.4f}\n")
                f.write(f"Improvement (N=1 -> Best): {result['improvement']:.4f}\n\n")
                f.write("Accuracy Curve F(N):\n")
                f.write("-" * 70 + "\n")
                f.write(f"{'N':<5} {'Accuracy':<10}\n")
                f.write("-" * 70 + "\n")
                for n in sorted(result['accuracy_curve'].keys()):
                    acc = result['accuracy_curve'][n]
                    markers = []
                    if n == result.get('best_n'):
                        markers.append("Best")
                    if n == result['optimal_n']:
                        markers.append("Optimal")
                    marker = f" <-- {' & '.join(markers)}" if markers else ""
                    f.write(f"{n:<5} {acc:<10.4f}{marker}\n")
                f.write("-" * 70 + "\n")
        
        # 모델별 요약 파일 생성
        summary_path = os.path.join(model_output_dir, f'optimal_n_summary_{num_trials}.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write(f"## Model: {model_path}\n")
            f.write("Optimal N* Summary\n")
            f.write("=" * 70 + "\n\n")
            
            # 데이터셋 순서 정의
            dataset_order = [
                'GSM8K',
                'MATH500_test',
                'MATH_train',
                'AMC23',
                'AIME_2024',
                'AIME_2025',
                'MMLU-Pro',
                'GPQA-Diamond',
                'MATH2500'
            ]
            
            # 데이터셋별로 지정된 순서로 정렬
            def get_dataset_sort_key(result):
                dataset_name = result['dataset_name']
                try:
                    return dataset_order.index(dataset_name)
                except ValueError:
                    # 순서에 없는 데이터셋은 뒤에 배치
                    return len(dataset_order) + hash(dataset_name)
            
            sorted_results = sorted(model_result_list, key=get_dataset_sort_key)
            
            # 헤더 작성
            f.write(f"{'Dataset':<25} {'Optimal N*':<12} {'Best N*':<10} {'Best Accuracy':<15} {'N=1 Acc':<12} {'N=40 Acc':<12} {'Improvement':<12}\n")
            f.write("-" * 70 + "\n")
            
            # 각 데이터셋별 정보 작성
            for result in sorted_results:
                dataset_name = result['dataset_name']
                optimal_n = result['optimal_n']
                best_n = result.get('best_n', optimal_n)
                best_accuracy = result.get('best_accuracy', result.get('max_accuracy', 0.0))
                n_1_accuracy = result['n_1_accuracy']
                n_40_accuracy = result['n_40_accuracy']
                improvement = result['improvement']
                
                f.write(f"{dataset_name:<25} {optimal_n:<12} {best_n:<10} {best_accuracy:<15.4f} {n_1_accuracy:<12.4f} {n_40_accuracy:<12.4f} {improvement:<12.4f}\n")
            
            f.write("-" * 70 + "\n\n")
            
            # 통계 정보
            optimal_ns = [r['optimal_n'] for r in sorted_results]
            best_accuracies = [r.get('best_accuracy', r.get('max_accuracy', 0.0)) for r in sorted_results]
            improvements = [r['improvement'] for r in sorted_results]
            
            f.write("Statistics:\n")
            f.write(f"  Total Datasets: {len(sorted_results)}\n")
            f.write(f"  Average Optimal N*: {sum(optimal_ns) / len(optimal_ns):.2f}\n")
            f.write(f"  Optimal N* Range: {min(optimal_ns)} ~ {max(optimal_ns)}\n")
            f.write(f"  Average Best Accuracy: {sum(best_accuracies) / len(best_accuracies):.4f}\n")
            f.write(f"  Average Improvement: {sum(improvements) / len(improvements):.4f}\n")
    
    print(f"\n결과 저장 완료: {output_dir}")


def find_report_files(results_optimal_dir: str,
                      stages_config: str,
                      num_trials: int,
                      model_filters: List[str] = None,
                      dataset_filters: List[str] = None) -> List[str]:
    """
    results_optimal 디렉토리에서 stages_config에 해당하는 리포트 txt 파일을 찾아 반환.

    기본 저장 구조(이 스크립트 기준):
      results_optimal/<family>/<model>/<stages_config>/*_optimal_n_report_<num_trials>.txt
    """
    report_files: List[str] = []
    if not os.path.exists(results_optimal_dir):
        return report_files

    for root, _, files in os.walk(results_optimal_dir):
        # stages_config 폴더만 대상으로
        if os.path.basename(root) != stages_config:
            continue

        # 모델 필터: root 경로에 토큰 포함 여부(상단 VISUALIZATION_MODELS 정책과 동일하게 부분 문자열 매칭)
        root_norm = os.path.normpath(root)
        if model_filters is not None and len(model_filters) > 0:
            if not any(token in root_norm for token in model_filters):
                continue

        for fn in files:
            if not fn.endswith(f"_optimal_n_report_{num_trials}.txt"):
                continue
            # 데이터셋 필터: 파일명에서 dataset_name 추출 후 필터링
            dataset_name = fn.replace(f"_optimal_n_report_{num_trials}.txt", "")
            if dataset_filters is not None and len(dataset_filters) > 0:
                if dataset_name not in dataset_filters:
                    continue
            report_files.append(os.path.join(root, fn))

    return report_files


def find_json_files(results_dir: str, 
                    model_filters: List[str] = None,
                    dataset_filters: List[str] = None,
                    stages_config: str = 'stages-40') -> List[Tuple[str, str]]:
    """
    results 디렉토리에서 모델/데이터셋 필터에 맞는 JSON 파일들을 찾아 반환
    
    Args:
        results_dir: results 디렉토리 경로
        model_filters: 모델 필터 리스트 (None이면 모든 모델)
        dataset_filters: 데이터셋 필터 리스트 (None이면 모든 데이터셋)
        stages_config: stages 설정 (기본값: 'stages-40')
        
    Returns:
        List[Tuple[str, str]]: (모델_경로, JSON_파일_경로) 튜플 리스트
    """
    json_files = []
    
    if not os.path.exists(results_dir):
        return json_files
    
    # results_dir 하위를 재귀적으로 탐색하여 Stage1 디렉토리와 JSON 파일 찾기
    for root, dirs, files in os.walk(results_dir):
        if 'Stage1' in dirs:
            stage1_path = os.path.join(root, 'Stage1')
            if os.path.exists(stage1_path):
                # 부모 디렉토리가 stages_config인지 확인
                parent_dir = os.path.dirname(stage1_path)
                if os.path.basename(parent_dir) != stages_config:
                    continue
                
                # 모델 필터링: 경로에 모델 필터가 포함되어 있는지 확인
                stage1_path_normalized = os.path.normpath(stage1_path)
                if model_filters is not None and len(model_filters) > 0:
                    if not any(token in stage1_path_normalized for token in model_filters):
                        continue
                
                # Stage1 디렉토리 내의 JSON 파일 찾기
                for json_file in os.listdir(stage1_path):
                    if json_file.endswith('.json'):
                        dataset_name = os.path.splitext(json_file)[0]
                        
                        # 데이터셋 필터링
                        if dataset_filters is not None and len(dataset_filters) > 0:
                            if dataset_name not in dataset_filters:
                                continue
                        
                        json_path = os.path.join(stage1_path, json_file)
                        # 모델 경로 추출 (results_dir 기준 상대 경로)
                        model_path = os.path.relpath(os.path.dirname(parent_dir), results_dir)
                        json_files.append((model_path, json_path))
    
    return json_files


def main():
    parser = argparse.ArgumentParser(description='SC(N) 성능 변화 시각화')
    parser.add_argument('--num_trials', type=int, default=10,
                       help='각 N에 대한 반복 횟수 (기본값: 10)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='결과 저장 디렉토리 (None이면 visualization 폴더)')
    parser.add_argument('--seed', type=int, default=42,
                       help='랜덤 시드 (기본값: 42)')
    parser.add_argument('--stages_config', type=str, default='stages-40',
                       help='stages 설정 (기본값: stages-40). 예: 40, 100, stages-40, stages-100')
    parser.add_argument('--results_optimal_dir', type=str, default='experiments/results_optimal',
                       help='results_optimal 출력 디렉토리 (기본값: experiments/results_optimal)')
    parser.add_argument('--save_optimal_reports', action='store_true', default=True,
                       help='results_optimal 폴더에 리포트 저장 (기본값: True)')
    parser.add_argument('--reports_only', action='store_true', default=False,
                       help='results_optimal의 리포트(txt)만 사용하여 시각화. (리포트가 없으면 에러)')
    
    args = parser.parse_args()

    def normalize_stages_config(value: str) -> str:
        """
        --stages_config 입력을 stages-XXX 폴더명으로 정규화.
        허용 입력 예: "100", "stages-100", " 100 "
        """
        if value is None:
            return value
        v = str(value).strip()
        if not v:
            return v
        if v.isdigit():
            return f"stages-{v}"
        return v

    args.stages_config = normalize_stages_config(args.stages_config)
    
    # 랜덤 시드 설정
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # rsc_project 디렉토리 찾기
    script_dir = os.path.dirname(os.path.abspath(__file__))
    current_dir = script_dir
    rsc_project_dir = None
    for _ in range(10):
        if os.path.basename(current_dir) == 'rsc_project' and os.path.exists(os.path.join(current_dir, 'experiments')):
            rsc_project_dir = current_dir
            break
        parent = os.path.dirname(current_dir)
        if parent == current_dir:
            break
        current_dir = parent
    
    if not rsc_project_dir:
        print("Error: rsc_project 디렉토리를 찾을 수 없습니다.")
        return
    
    results_dir = os.path.join(rsc_project_dir, 'experiments', 'results')
    if not os.path.exists(results_dir):
        print(f"Error: Results 디렉토리를 찾을 수 없습니다: {results_dir}")
        return
    
    # VISUALIZATION_MODELS와 VISUALIZATION_DATASETS 사용하여 JSON 파일 찾기
    print(f"\n{'='*60}")
    print("Results 디렉토리/리포트에서 입력 데이터 검색 중...")
    print(f"Stages 설정: {args.stages_config}")
    if VISUALIZATION_MODELS is not None and len(VISUALIZATION_MODELS) > 0:
        print(f"모델 필터: {', '.join(VISUALIZATION_MODELS)}")
    else:
        print("모델 필터: 모든 모델")
    
    if VISUALIZATION_DATASETS is not None and len(VISUALIZATION_DATASETS) > 0:
        print(f"데이터셋 필터: {', '.join(VISUALIZATION_DATASETS)}")
    else:
        print("데이터셋 필터: 모든 데이터셋")
    print(f"{'='*60}\n")

    # results_optimal_dir 절대 경로 설정
    if not os.path.isabs(args.results_optimal_dir):
        results_optimal_dir = os.path.join(rsc_project_dir, args.results_optimal_dir)
    else:
        results_optimal_dir = args.results_optimal_dir

    # 1) (우선) results_optimal의 리포트(txt)에서 읽기
    report_files = find_report_files(
        results_optimal_dir=results_optimal_dir,
        stages_config=args.stages_config,
        num_trials=args.num_trials,
        model_filters=VISUALIZATION_MODELS if VISUALIZATION_MODELS is not None and len(VISUALIZATION_MODELS) > 0 else None,
        dataset_filters=VISUALIZATION_DATASETS if VISUALIZATION_DATASETS is not None and len(VISUALIZATION_DATASETS) > 0 else None,
    )

    optimal_results: List[Dict[str, Any]] = []
    if report_files:
        print(f"발견된 리포트(txt): {len(report_files)}개")
        for rp in report_files:
            parsed = parse_report_file(rp)
            if parsed:
                optimal_results.append(parsed)
            else:
                print(f"Warning: 리포트 파싱 실패: {rp}")
    else:
        print("리포트(txt)를 찾지 못했습니다. (results_optimal)")

    # output_dir 설정
    # 기본은 rsc_project/visualization 이고,
    # 실제 PNG는 그 아래 visualize_sc_scaling/<stages_config>/ 하위에 저장
    if args.output_dir is None:
        visualization_dir = os.path.join(rsc_project_dir, 'visualization')
        os.makedirs(visualization_dir, exist_ok=True)
        base_output_dir = visualization_dir
    else:
        base_output_dir = args.output_dir

    args.output_dir = os.path.join(base_output_dir, 'visualize_sc_scaling', args.stages_config)
    os.makedirs(args.output_dir, exist_ok=True)

    # 2) 리포트만 사용 모드면 여기서 종료 조건 체크
    if args.reports_only and not optimal_results:
        print("Error: reports_only 모드인데, 조건에 맞는 리포트(txt)가 없습니다.")
        print(f"  results_optimal_dir: {results_optimal_dir}")
        print(f"  stages_config: {args.stages_config}, num_trials: {args.num_trials}")
        print(f"  (PNG 저장 경로는 미리 생성됨): {args.output_dir}")
        return

    # 3) 리포트가 없거나 일부만 있는 경우, 부족분은 results(Stage1 JSON)에서 계산 후 리포트 저장
    #    (효율을 위해: 이미 리포트가 있는 (model_path, dataset_name)은 재계산하지 않음)
    existing_keys = set()
    for r in optimal_results:
        if r and r.get('model_path') and r.get('dataset_name'):
            existing_keys.add((r['model_path'], r['dataset_name']))

    json_files = find_json_files(
        results_dir=results_dir,
        model_filters=VISUALIZATION_MODELS if VISUALIZATION_MODELS is not None and len(VISUALIZATION_MODELS) > 0 else None,
        dataset_filters=VISUALIZATION_DATASETS if VISUALIZATION_DATASETS is not None and len(VISUALIZATION_DATASETS) > 0 else None,
        stages_config=args.stages_config
    )

    if not json_files and not optimal_results:
        print("Error: 필터 조건에 맞는 JSON/리포트를 찾을 수 없습니다.")
        return

    
    try:
        # 3) 부족한 (model,dataset)만 계산해서 optimal_results에 추가
        if json_files:
            missing = [(mp, jp) for (mp, jp) in json_files
                       if (mp, os.path.splitext(os.path.basename(jp))[0]) not in existing_keys]
        else:
            missing = []

        if missing and not args.reports_only:
            print(f"\n{'='*60}")
            print("리포트가 없는 항목 계산 중 (JSON → 리포트 생성)...")
            print(f"Missing pairs: {len(missing)}")
            print(f"{'='*60}\n")

            for model_path, json_path in missing:
                dataset_name = os.path.splitext(os.path.basename(json_path))[0]
                result = process_single_model_dataset(
                    model_path=model_path,
                    dataset_name=dataset_name,
                    json_path=json_path,
                    num_trials=args.num_trials,
                    seed=args.seed,
                    results_optimal_dir=results_optimal_dir,
                    stages_config=args.stages_config
                )
                if result:
                    optimal_results.append(result)
                    existing_keys.add((model_path, dataset_name))

            # results_optimal 폴더에 리포트 저장 (요청된 디렉토리 구조 적용됨)
            if args.save_optimal_reports and optimal_results:
                save_results(optimal_results, results_optimal_dir, stages_config=args.stages_config, num_trials=args.num_trials)

        # 4) 시각화는 (가능하면) results_optimal 리포트에서 읽은 값(=optimal_results)로만 수행
        if not optimal_results:
            print("Error: 시각화할 결과가 없습니다. (리포트/계산 결과 모두 없음)")
            return

        # 모델별로 accuracy_curve 저장: {model_path: {dataset_name: accuracy_curve}}
        accuracy_curves_by_model: Dict[str, Dict[str, Dict[int, float]]] = {}
        best_n_by_model_dataset: Dict[Tuple[str, str], Optional[int]] = {}
        optimal_n_by_model_dataset: Dict[Tuple[str, str], Optional[int]] = {}
        total_samples_by_model_dataset: Dict[Tuple[str, str], int] = {}

        for r in optimal_results:
            if not r:
                continue
            mp = r.get('model_path')
            ds = r.get('dataset_name')
            curve = r.get('accuracy_curve')
            opt_n = r.get('optimal_n')
            if not mp or not ds or not curve:
                continue
            if mp not in accuracy_curves_by_model:
                accuracy_curves_by_model[mp] = {}
            accuracy_curves_by_model[mp][ds] = curve
            best_n_by_model_dataset[(mp, ds)] = r.get('best_n')
            optimal_n_by_model_dataset[(mp, ds)] = opt_n
            total_samples_by_model_dataset[(mp, ds)] = int(r.get('total_samples', 0) or 0)

        print(f"\n{'='*60}")
        print("리포트 기반 시각화 생성 중...")
        print(f"모델 수: {len(accuracy_curves_by_model)}")
        print(f"{'='*60}\n")

        for model_path, dataset_curves in accuracy_curves_by_model.items():
            print(f"\n{'='*60}")
            print(f"모델 시각화: {model_path}")
            print(f"{'='*60}")

            results_by_task = {}
            for dataset_name, accuracy_curve in dataset_curves.items():
                # summary 형식으로 변환 (N=1~40 고정)
                summary = {}
                for n in range(1, 41):
                    if n in accuracy_curve:
                        summary[n] = {
                            'accuracy': accuracy_curve[n],
                            'accuracy_std': 0.0,
                            'tokens': 0.0,
                            'tokens_std': 0.0
                        }

                results_by_task[dataset_name] = {
                    'dataset_name': dataset_name,
                    'total_samples': total_samples_by_model_dataset.get((model_path, dataset_name), 0),
                    'summary': summary,
                    'best_n': best_n_by_model_dataset.get((model_path, dataset_name)),
                    'optimal_n': optimal_n_by_model_dataset.get((model_path, dataset_name))
                }

            if results_by_task:
                model_safe_name = model_path.replace('/', '_').replace('\\', '_')
                output_file = os.path.join(
                    args.output_dir,
                    f'all_tasks_sc_scaling_{args.stages_config}_{args.num_trials}_{model_safe_name}.png'
                )
                _create_visualization_from_results(results_by_task, args.output_dir, output_file=output_file)

        # 5) 전체 통계 출력
        print(f"\n{'='*70}")
        print("전체 통계")
        print(f"{'='*70}")

        valid_results = [r for r in optimal_results if r is not None and r.get('optimal_n') is not None]
        if valid_results:
            optimal_ns = [r['optimal_n'] for r in valid_results]
            best_accuracies = [r.get('best_accuracy', r.get('max_accuracy', 0.0)) for r in valid_results]
            improvements = [r.get('improvement', 0.0) for r in valid_results]

            print(f"처리된 모델/데이터셋 조합: {len(valid_results)}개")
            print(f"평균 최적 N*: {sum(optimal_ns) / len(optimal_ns):.2f}")
            print(f"최적 N* 범위: {min(optimal_ns)} ~ {max(optimal_ns)}")
            print(f"평균 Best 정확도: {sum(best_accuracies) / len(best_accuracies):.4f}")
            print(f"평균 개선도: {sum(improvements) / len(improvements):.4f}")

        print(f"{'='*70}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
