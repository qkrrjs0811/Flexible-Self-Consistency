"""
답안 분포 다양성 분석 결과 시각화

맞춘 문제 vs 틀린 문제에서 SC 답변의 고유 답 개수(다양성)를 비교한다.
"""

import json
import os
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any
from collections import Counter

# 프로젝트 루트 경로 추가
script_dir = os.path.dirname(os.path.abspath(__file__))
# rsc_project 디렉토리 찾기
current_dir = script_dir
rsc_project_dir = None
for _ in range(10):
    if os.path.basename(current_dir) == 'rsc_project' and os.path.exists(os.path.join(current_dir, 'evaluation')):
        rsc_project_dir = current_dir
        break
    parent = os.path.dirname(current_dir)
    if parent == current_dir:
        break
    current_dir = parent

if rsc_project_dir is None:
    rsc_project_dir = '/home/albert0811/dev_env/git/repos/RASC/rsc_project'
    if not os.path.exists(rsc_project_dir):
        rsc_project_dir = os.path.join(os.path.dirname(script_dir), '../../rsc_project')
        rsc_project_dir = os.path.abspath(rsc_project_dir)

if rsc_project_dir and os.path.exists(rsc_project_dir):
    if rsc_project_dir not in sys.path:
        sys.path.insert(0, rsc_project_dir)

from evaluation.metrics import extract_cot_data, normalize_answer_eval

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 시각화에 사용할 데이터셋 리스트 (원하는 데이터셋만 추가/제거)
# ============================================================================
# None이면 모든 데이터셋 사용, 리스트가 비어있으면 모든 데이터셋 사용
# 예시: ['AMC23', 'GSM8K', 'MATH500_test'] - 이 데이터셋들만 시각화
VISUALIZATION_DATASETS = ['GSM8K', 'MATH500_test', 'AMC23', 'AIME_2024', 'AIME_2025']  # None이면 모든 데이터셋, 또는 ['AMC23', 'GSM8K', ...] 형태로 지정

# ============================================================================
# (자동 탐색 시) 시각화에 사용할 모델 리스트
# ============================================================================
# Stage1 자동 탐색(args.stage1_dir=None)에서만 적용됩니다.
# - None이면 모든 모델 후보 사용
# - 리스트가 비어있으면 모든 모델 후보 사용
# 예시: ['Qwen2.5-7B-Instruct', 'gemma-3-4b-it']
VISUALIZATION_MODELS = ['Qwen2.5-7B-Instruct']

# 데이터셋 이름 매핑 (시각화용 표시 이름)
DATASET_DISPLAY_NAMES = {
    'MATH500_test': 'MATH500',
    'AIME_2024': 'AIME2024',
    'AIME_2025': 'AIME2025',
    # 다른 데이터셋 이름도 필요시 여기에 추가
}
# ============================================================================


def calculate_diversity_metrics(answer_distribution: Dict[str, int], total_chains: int) -> Dict[str, float]:
    """
    답안 분포의 다양성 지표 계산 (정규화 전 원본 답 문자열 기준 고유 개수).
    """
    if not answer_distribution or total_chains == 0:
        return {'num_unique_answers': 0}

    return {'num_unique_answers': float(len(answer_distribution))}


def analyze_sample(sample: Dict, dataset_name: str) -> Dict:
    """
    단일 샘플 분석
    
    Args:
        sample: 샘플 데이터 (JSON에서 읽은 딕셔너리)
        dataset_name: 데이터셋 이름
        
    Returns:
        Dict: 분석 결과
    """
    # Reasoning Paths에서 답변 추출
    reasoning_paths = sample.get('Reasoning Paths', [])
    answers, _ = extract_cot_data(reasoning_paths)
    
    if not answers:
        return {
            'id': sample.get('id', 'unknown'),
            'ground_truth': str(sample.get('answer', '')),
            'answer_distribution': {},
            'majority_answer': '',
            'is_correct': False,
            'num_chains': 0,
            'diversity_metrics': calculate_diversity_metrics({}, 0)
        }
    
    # 답변 분포 계산 (정규화 전 원본 답변 기준)
    answer_counter = Counter(answers)
    answer_distribution = dict(answer_counter)
    
    # Majority vote로 최종 답변 결정 (정규화 후 비교)
    normalized_answers = [normalize_answer_eval(ans) for ans in answers]
    normalized_counter = Counter(normalized_answers)
    majority_normalized = normalized_counter.most_common(1)[0][0] if normalized_counter else ''
    
    # 정규화된 majority answer에 해당하는 원본 답변 찾기
    majority_answer = ''
    for ans in answers:
        if normalize_answer_eval(ans) == majority_normalized:
            majority_answer = ans
            break
    if not majority_answer:
        majority_answer = majority_normalized
    
    # 실제 정답과 비교
    ground_truth = str(sample.get('answer', '')).strip()
    is_correct = normalize_answer_eval(majority_answer) == normalize_answer_eval(ground_truth)
    
    # 다양성 지표 계산
    diversity_metrics = calculate_diversity_metrics(answer_distribution, len(answers))
    
    return {
        'id': sample.get('id', 'unknown'),
        'ground_truth': ground_truth,
        'answer_distribution': answer_distribution,
        'majority_answer': majority_answer,
        'is_correct': is_correct,
        'num_chains': len(answers),
        'diversity_metrics': diversity_metrics
    }


def calculate_statistics(diversity_metrics_list: List[Dict[str, float]]) -> Dict[str, Any]:
    """
    다양성 지표들의 통계 계산
    
    Args:
        diversity_metrics_list: 다양성 지표 딕셔너리 리스트
        
    Returns:
        Dict: 통계 정보
    """
    if not diversity_metrics_list:
        return {
            'count': 0,
            'num_unique_answers': {'mean': 0, 'min': 0, 'max': 0},
        }

    num_unique = [m['num_unique_answers'] for m in diversity_metrics_list]

    def calc_stats(values):
        return {
            'mean': float(np.mean(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values))
        }

    return {
        'count': len(diversity_metrics_list),
        'num_unique_answers': calc_stats(num_unique),
    }


def analyze_dataset_from_json(json_path: str, dataset_name: str) -> Dict[str, Dict[str, float]]:
    """
    JSON 파일에서 직접 데이터를 읽어서 분석
    
    Args:
        json_path: Stage1 JSON 파일 경로
        dataset_name: 데이터셋 이름
        
    Returns:
        Dict: 데이터셋별 통계 (correct / incorrect 에 평균 고유 답 개수)
    """
    print(f"  분석 중: {dataset_name}")
    
    # JSON 파일 읽기
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 각 샘플 분석
    correct_samples = []
    incorrect_samples = []
    
    for i, sample in enumerate(data):
        result = analyze_sample(sample, dataset_name)
        
        if result['is_correct']:
            correct_samples.append(result)
        else:
            incorrect_samples.append(result)
        
        # 진행 상황 출력
        if (i + 1) % 100 == 0:
            print(f"    Processed {i + 1}/{len(data)} samples...")
    
    # 통계 계산
    correct_diversity_metrics = [r['diversity_metrics'] for r in correct_samples]
    incorrect_diversity_metrics = [r['diversity_metrics'] for r in incorrect_samples]
    
    correct_stats = calculate_statistics(correct_diversity_metrics)
    incorrect_stats = calculate_statistics(incorrect_diversity_metrics)
    
    return {
        'correct': {
            'num_unique_answers': correct_stats['num_unique_answers']['mean'],
        },
        'incorrect': {
            'num_unique_answers': incorrect_stats['num_unique_answers']['mean'],
        }
    }


def analyze_stage1_directory(stage1_dir: str, include_ours: bool = False, 
                            target_datasets: List[str] = None) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Stage1 디렉토리에서 JSON 파일들을 읽어서 분석
    
    Args:
        stage1_dir: Stage1 결과 디렉토리 경로
        include_ours: Ours 파일도 포함할지 여부
        target_datasets: 분석할 데이터셋 리스트 (None이면 모든 데이터셋)
        
    Returns:
        Dict: 데이터셋별 통계
    """
    results = {}
    
    json_files = [f for f in os.listdir(stage1_dir) if f.endswith('.json')]
    
    if not json_files:
        print(f"Error: No JSON files found in {stage1_dir}")
        return results
    
    # 일반 파일과 Ours 파일 분리
    normal_files = []
    ours_files = []
    
    for json_file in json_files:
        dataset_name = os.path.splitext(json_file)[0]
        
        # target_datasets가 지정된 경우 필터링
        if target_datasets is not None and len(target_datasets) > 0:
            if dataset_name not in target_datasets:
                continue
        
        # Ours 파일인지 확인 (_Ours가 포함되어 있는지)
        if '_Ours' in dataset_name:
            ours_files.append(json_file)
        else:
            normal_files.append(json_file)
    
    # 일반 파일 분석
    print(f"\n일반 파일 분석 시작 ({len(normal_files)}개 파일)")
    for json_file in normal_files:
        dataset_name = os.path.splitext(json_file)[0]
        json_path = os.path.join(stage1_dir, json_file)
        
        try:
            dataset_data = analyze_dataset_from_json(json_path, dataset_name)
            results[dataset_name] = dataset_data
        except Exception as e:
            print(f"  Error analyzing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Ours 파일 분석 (include_ours가 True인 경우)
    if include_ours and ours_files:
        print(f"\nOurs 파일 분석 시작 ({len(ours_files)}개 파일)")
        for json_file in ours_files:
            dataset_name = os.path.splitext(json_file)[0]
            json_path = os.path.join(stage1_dir, json_file)
            
            try:
                dataset_data = analyze_dataset_from_json(json_path, dataset_name)
                results[dataset_name] = dataset_data
            except Exception as e:
                print(f"  Error analyzing {dataset_name}: {e}")
                import traceback
                traceback.print_exc()
    
    return results


def create_visualization(data: Dict[str, Dict[str, Dict[str, float]]], 
                         output_path: str,
                         dataset_order: List[str] = None):
    """
    연구 가설을 뒷받침하는 시각화 생성
    
    Args:
        data: 파싱된 데이터
        output_path: 출력 파일 경로
        dataset_order: 데이터셋 표시 순서 (None이면 data.keys() 순서 사용)
    """
    # dataset_order가 지정된 경우 그 순서대로 정렬
    if dataset_order is not None and len(dataset_order) > 0:
        # dataset_order에 있는 데이터셋만 필터링하고 순서 유지
        datasets = [d for d in dataset_order if d in data.keys()]
        # dataset_order에 없지만 data에 있는 데이터셋은 뒤에 추가
        remaining = [d for d in data.keys() if d not in datasets]
        datasets = datasets + remaining
    else:
        datasets = list(data.keys())
    
    n_datasets = len(datasets)
    
    correct_unique = [data[d]['correct']['num_unique_answers'] for d in datasets]
    incorrect_unique = [data[d]['incorrect']['num_unique_answers'] for d in datasets]

    display_names = [DATASET_DISPLAY_NAMES.get(d, d) for d in datasets]

    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

    x = np.arange(n_datasets)
    width = 0.35

    bars1_correct = ax1.bar(x - width/2, correct_unique, width,
                           label='Correct', color='#3498db', alpha=0.8, edgecolor='black')
    bars1_incorrect = ax1.bar(x + width/2, incorrect_unique, width,
                             label='Incorrect', color='#e74c3c', alpha=0.8, edgecolor='black')

    ax1.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Unique Answers', fontsize=12, fontweight='bold')
    ax1.set_title('Number of Unique Answers (Correct vs Incorrect)',
                 fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(display_names, rotation=45, ha='right')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    ymax = max(incorrect_unique + correct_unique) if (incorrect_unique + correct_unique) else 1.0
    for bars in [bars1_correct, bars1_incorrect]:
        for bar in bars:
            height = bar.get_height()
            if height > 0.1:
                ax1.text(bar.get_x() + bar.get_width()/2., height + ymax * 0.02,
                        f'{height:.1f}',
                        ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"시각화 저장: {output_path}")
    plt.close()


def create_summary_statistics(data: Dict[str, Dict[str, Dict[str, float]]]):
    """
    전체 요약 통계 출력
    
    Args:
        data: 파싱된 데이터
    """
    print("\n" + "="*80)
    print("고유 답 개수 요약 (Correct vs Incorrect)")
    print("="*80)

    all_unique_diff = []

    for dataset_name, dataset_data in data.items():
        correct_unique = dataset_data['correct']['num_unique_answers']
        incorrect_unique = dataset_data['incorrect']['num_unique_answers']
        unique_diff = incorrect_unique - correct_unique
        all_unique_diff.append(unique_diff)

        print(f"{dataset_name}:")
        print(f"  고유 답변 개수 차이 (Incorrect − Correct): {unique_diff:+.2f} "
              f"(틀린 문제가 {'더 다양함' if unique_diff > 0 else '덜 다양함' if unique_diff < 0 else '동일'})")
        print()

    print("-"*80)
    print("전체 평균:")
    print(f"  평균 고유 답변 개수 차이: {np.mean(all_unique_diff):+.2f}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='답안 분포 다양성 분석 결과 시각화')
    parser.add_argument('--stage1_dir', type=str, default=None,
                       help='Stage1 결과 디렉토리 경로 (None이면 자동 탐색)')
    parser.add_argument('--stages_config', type=str, default='stages-40',
                       help='stages 설정 (예: 40, 100, stages-40, stages-100). 자동 탐색 시 이 stages 폴더만 사용')
    parser.add_argument('--model', type=str, default=None,
                       help='(자동 탐색 시) 특정 모델만 필터링 (예: Qwen2.5-7B-Instruct). 지정하면 상단 VISUALIZATION_MODELS보다 우선')
    parser.add_argument('--output_file', type=str, default=None,
                       help='출력 이미지 파일 경로 (None이면 stage1_dir과 같은 디렉토리)')
    parser.add_argument('--include_ours', action='store_true',
                       help='Ours 파일도 포함하여 분석 (기본값: False)')
    
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
        # "stages-100" 같은 케이스는 그대로 사용
        return v

    args.stages_config = normalize_stages_config(args.stages_config)

    # 자동 탐색 시 사용할 모델 필터 결정
    # - args.model이 있으면 단일 모델로 강제
    # - 없으면 상단 VISUALIZATION_MODELS 사용
    target_models = [args.model] if args.model else VISUALIZATION_MODELS
    
    # Stage1 디렉토리 찾기
    if args.stage1_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # rsc_project 디렉토리 찾기
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
        
        # rsc_project 디렉토리에서 experiments/results 하위의 Stage1 찾기
        stage1_dir = None
        if rsc_project_dir:
            experiments_dir = os.path.join(rsc_project_dir, 'experiments', 'results')
            if os.path.exists(experiments_dir):
                # stages_config에 맞는 Stage1 디렉토리만 후보로 모으기
                stage1_dirs = []
                for root, dirs, files in os.walk(experiments_dir):
                    if 'Stage1' in dirs:
                        stage1_path = os.path.join(root, 'Stage1')
                        if os.path.exists(stage1_path):
                            # Stage1의 부모는 stages-XX 여야 함
                            parent_dir = os.path.dirname(stage1_path)
                            if os.path.basename(parent_dir) != args.stages_config:
                                continue
                            # (선택) 모델 필터 적용: .../results/<family>/<model>/<stages_config>/Stage1
                            if target_models is not None and len(target_models) > 0:
                                model_dir = os.path.basename(os.path.dirname(parent_dir))
                                if model_dir not in target_models:
                                    continue
                            stage1_dirs.append(stage1_path)
                
                if stage1_dirs:
                    # 가장 최근에 수정된 디렉토리 선택
                    stage1_dir = max(stage1_dirs, key=lambda p: os.path.getmtime(p))
        
        if not stage1_dir:
            print("Error: Stage1 디렉토리를 찾을 수 없습니다. --stage1_dir을 지정해주세요.")
            print("예시: --stage1_dir experiments/results/Qwen/Qwen2.5-7B-Instruct/stages-40/Stage1")
            print("또는: --stages_config 100  (또는 stages-100)")
            return
    else:
        stage1_dir = args.stage1_dir
        # 상대 경로인 경우 rsc_project 기준으로 변환
        if not os.path.isabs(stage1_dir):
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
            
            if rsc_project_dir:
                stage1_dir = os.path.join(rsc_project_dir, stage1_dir)
    
    if not os.path.exists(stage1_dir):
        print(f"Error: Directory not found: {stage1_dir}")
        return
    
    print(f"Stage1 디렉토리: {stage1_dir}")
    
    # 시각화할 데이터셋 리스트 (코드 상단의 VISUALIZATION_DATASETS 사용)
    target_datasets = VISUALIZATION_DATASETS
    if target_datasets is not None and len(target_datasets) > 0:
        print(f"\n시각화할 데이터셋: {', '.join(target_datasets)}")
    else:
        print(f"\n모든 데이터셋 사용")
    
    # JSON 파일에서 직접 분석
    print(f"\nJSON 파일에서 직접 분석 중...")
    data = analyze_stage1_directory(stage1_dir, include_ours=args.include_ours, 
                                   target_datasets=target_datasets)
    
    if not data:
        print("Error: 데이터를 추출할 수 없습니다.")
        return
    
    print(f"\n추출된 데이터셋 수: {len(data)}")
    print(f"데이터셋: {', '.join(data.keys())}")
    
    # 요약 통계 출력
    create_summary_statistics(data)
    
    # 출력 경로 설정: 기본은 rsc_project/visualization 디렉토리
    if args.output_file is None:
        # rsc_project 루트 탐색
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
            rsc_project_dir = os.path.join(os.path.dirname(script_dir), '..')
            rsc_project_dir = os.path.abspath(rsc_project_dir)
        visualization_dir = os.path.join(rsc_project_dir, 'visualization')
        os.makedirs(visualization_dir, exist_ok=True)
        output_file = os.path.join(visualization_dir, 'answer_diversity_visualization.png')
    else:
        output_file = args.output_file
    
    # 시각화 생성
    print(f"\n시각화 생성 중...")
    create_visualization(data, output_file, dataset_order=target_datasets)
    
    print("\n완료!")


if __name__ == '__main__':
    main()
