"""
η_adj 기반 동적 SC 샘플링 및 평가

probe 폴더의 JSON 파일에서 number_of_paths 값을 읽어서:
1. 모델/데이터셋별로 자동으로 results 폴더에서 Stage1(pool) JSON 파일 찾기 (stages_config 기반)
2. 각 샘플에 대해 N_SC = number_of_paths 값 사용
3. 각 trial마다 전체 pool(stages_config)에서 40개 subset을 랜덤 샘플링한 뒤, 그 subset에서 N_SC개를 추가로 랜덤 샘플링하여 평가 (부트스트래핑)
4. JSON 파일 생성 없이 평가 수행하여 eval_result.txt 형식으로 출력
"""

import json
import os
import sys
import random
import argparse
import math
import re
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter

# 프로젝트 루트 경로 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluation.metrics import (
    extract_cot_data, normalize_answer_eval,
    extract_pool_size_from_stages_config,
    prepare_samples
)

# ============================================================================
# 실험 설정: 사용할 모델 목록
# ============================================================================
# None이면 probe 폴더의 모든 모델을 처리합니다.
# 특정 모델만 지정하려면 리스트에 모델 이름을 추가하세요.
# 예: MODELS = ["Qwen2.5-7B-Inst", "gemma-3-4b-it"]
MODELS = ["Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct", "google/gemma-3-4b-it"]
# MODELS = ["meta-llama/Llama-3.1-8B-Instruct"]

# ============================================================================


def _is_qwen3_model(model_name: str) -> bool:
    """Check if the model is a Qwen3 series model"""
    model_name_lower = model_name.lower()
    return 'qwen3' in model_name_lower or 'qwen-3' in model_name_lower


def _is_nemotron_model(model_name: str) -> bool:
    """Check if the model is a Nemotron series model"""
    model_name_lower = model_name.lower()
    return 'nemotron' in model_name_lower or 'nvidia-nemotron' in model_name_lower


def load_probe_data(probe_dir: str) -> Dict[str, Dict[str, Dict[str, Dict[str, int]]]]:
    """
    probe 폴더에서 데이터 로드
    
    폴더 구조:
    - probe/.../{model_name}/{p_value}/{dataset_name}.json
      예: probe/math2500/gemma-3-4b-it/p_1.0/GSM8K.json
    
    각 JSON 파일은 리스트 형태: [{"id": "...", "number_of_paths": N, ...}, ...]
    
    Args:
        probe_dir: probe 디렉토리 경로
        
    Returns:
        데이터 딕셔너리:
        {model_name: {p_value: {dataset_name: {sample_id: number_of_paths, ...}, ...}, ...}, ...}
    """
    result: Dict[str, Dict[str, Dict[str, Dict[str, int]]]] = {}
    
    if not os.path.exists(probe_dir):
        raise FileNotFoundError(f"probe 디렉토리를 찾을 수 없습니다: {probe_dir}")

    def _load_dataset_json(json_path: str) -> Optional[Dict[str, int]]:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"  경고: {json_path} 로드 실패: {e}")
            return None

        if not isinstance(data, list):
            return None

        sample_dict: Dict[str, int] = {}
        for item in data:
            sample_id = str(item.get('id', ''))
            number_of_paths = item.get('number_of_paths')
            if sample_id and number_of_paths is not None:
                sample_dict[sample_id] = int(number_of_paths)
        return sample_dict if sample_dict else None

    # probe_dir 하위 어디서든 .../{model_name}/{p_value}/*.json 구조를 탐색
    for root, dirs, files in os.walk(probe_dir):
        p_value = os.path.basename(root)
        if not p_value.startswith("p_"):
            continue

        model_dir = os.path.dirname(root)
        model_name = os.path.basename(model_dir)
        if not model_name:
            continue

        for filename in files:
            if not filename.endswith(".json"):
                continue
            dataset_name = os.path.splitext(filename)[0]
            json_path = os.path.join(root, filename)
            sample_dict = _load_dataset_json(json_path)
            if not sample_dict:
                continue

            result.setdefault(model_name, {}).setdefault(p_value, {})[dataset_name] = sample_dict

    # 로드 결과 출력(요약)
    for model_name, p_map in result.items():
        for p_value, ds_map in p_map.items():
            for dataset_name, sample_dict in ds_map.items():
                print(f"  로드: {model_name}/{p_value}/{dataset_name} ({len(sample_dict)}개 샘플)")

    return result


def find_sc40_json(results_dir: str, model_name: str, dataset_name: str,
                   stages_config: str = "stages-40") -> Optional[str]:
    """
    results 폴더에서 SC(n=40) JSON 파일 찾기
    
    Args:
        results_dir: results 디렉토리 경로
        model_name: 모델 이름 (probe 폴더의 이름)
        dataset_name: 데이터셋 이름
        stages_config: stages 설정 (기본값: "stages-40")
        
    Returns:
        Optional[str]: JSON 파일 경로 (없으면 None)
    """
    # 데이터셋 이름 매핑 (probe 폴더 이름 -> results 폴더 이름)
    dataset_name_map = {
        'MATH500': 'MATH500_test'
    }
    
    # 매핑된 데이터셋 이름 사용
    actual_dataset_name = dataset_name_map.get(dataset_name, dataset_name)
    
    # 가능한 경로들 (여러 패턴 시도)
    possible_paths = []
    
    # 1. 직접 매칭: results_dir/model_name/stages-40/Stage1/dataset_name.json
    base_dir = os.path.join(results_dir, model_name)
    stage1_path = os.path.join(base_dir, stages_config, "Stage1", f"{actual_dataset_name}.json")
    possible_paths.append(stage1_path)
    
    # 2. "Inst" -> "Instruct" 변환 시도
    if model_name.endswith("Inst"):
        model_name_instruct = model_name.replace("Inst", "Instruct")
        base_dir_instruct = os.path.join(results_dir, model_name_instruct)
        stage1_path_instruct = os.path.join(base_dir_instruct, stages_config, "Stage1", f"{actual_dataset_name}.json")
        possible_paths.append(stage1_path_instruct)
    
    # 3. Qwen 계열 모델의 경우 "Qwen/" 접두사 추가 시도
    if "qwen" in model_name.lower():
        # Qwen2.5-7B-Inst -> Qwen/Qwen2.5-7B-Instruct
        if model_name.endswith("Inst"):
            model_name_full = model_name.replace("Inst", "Instruct")
            if not model_name_full.startswith("Qwen/"):
                model_name_full = f"Qwen/{model_name_full}"
        else:
            if not model_name.startswith("Qwen/"):
                model_name_full = f"Qwen/{model_name}"
        base_dir_qwen = os.path.join(results_dir, model_name_full)
        stage1_path_qwen = os.path.join(base_dir_qwen, stages_config, "Stage1", f"{actual_dataset_name}.json")
        possible_paths.append(stage1_path_qwen)
    
    # 4. Gemma 계열 모델의 경우 "google/" 접두사 추가 시도
    if "gemma" in model_name.lower():
        # 예: gemma-3-4b-it -> google/gemma-3-4b-it
        if not model_name.startswith("google/"):
            model_name_full = f"google/{model_name}"
        else:
            model_name_full = model_name
        base_dir_gemma = os.path.join(results_dir, model_name_full)
        stage1_path_gemma = os.path.join(base_dir_gemma, stages_config, "Stage1", f"{actual_dataset_name}.json")
        possible_paths.append(stage1_path_gemma)
        
    # 4.5. Llama 계열 모델의 경우 "meta-llama/" 접두사 추가 시도
    if "llama" in model_name.lower():
        if not model_name.startswith("meta-llama/"):
            model_name_full = f"meta-llama/{model_name}"
        else:
            model_name_full = model_name
        base_dir_llama = os.path.join(results_dir, model_name_full)
        stage1_path_llama = os.path.join(base_dir_llama, stages_config, "Stage1", f"{actual_dataset_name}.json")
        possible_paths.append(stage1_path_llama)
    
    # 5. results 디렉토리 전체 탐색 (유연한 매칭, 단 동일 모델 계열에 한정)
    if not any(os.path.exists(p) for p in possible_paths):
        # 모델 이름과 그 변형들 중 하나라도 포함된 경로만 대상으로,
        # results_dir 내의 모든 하위 디렉토리 탐색
        name_candidates = set()
        name_candidates.add(model_name)
        # Inst/Instruct 변형
        if model_name.endswith("Inst"):
            name_candidates.add(model_name.replace("Inst", "Instruct"))
        if model_name.endswith("Instruct"):
            name_candidates.add(model_name.replace("Instruct", "Inst"))
        # 벤더 접두사 포함 변형들
        if "qwen" in model_name.lower():
            if not model_name.startswith("Qwen/"):
                name_candidates.add(f"Qwen/{model_name}")
        if "gemma" in model_name.lower():
            if not model_name.startswith("google/"):
                name_candidates.add(f"google/{model_name}")
        if "llama" in model_name.lower():
            if not model_name.startswith("meta-llama/"):
                name_candidates.add(f"meta-llama/{model_name}")
        # 소문자 버전도 함께 사용
        name_candidates_lower = {n.lower() for n in name_candidates}

        for root, dirs, files in os.walk(results_dir):
            # stages-40/Stage1 경로인지 확인
            if stages_config in root and "Stage1" in root:
                # 모델 이름 계열이 경로에 포함되지 않으면 스킵 (다른 모델로 잘못 매칭되는 것 방지)
                root_lower = root.lower()
                if not any(n in root_lower for n in name_candidates_lower):
                    continue
                target_file = os.path.join(root, f"{actual_dataset_name}.json")
                if os.path.exists(target_file):
                    possible_paths.append(target_file)
                    break  # 첫 번째 매칭만 사용
    
    # 파일 존재 확인
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None




def calculate_n_sc(number_of_paths: int, max_n: int = 40) -> int:
    """
    number_of_paths 값을 N_SC로 사용 (이미 계산된 값)
    
    Args:
        number_of_paths: probe에서 읽은 number_of_paths 값
        max_n: 최대 샘플 수 (기본값: 40, 범위 체크용)
        
    Returns:
        int: N_SC = number_of_paths (1~max_n 범위로 클리핑)
    """
    return max(1, min(max_n, int(number_of_paths)))  # 1~max_n 범위로 클리핑


def sample_reasoning_paths(sample: Dict[str, Any], n_sc: int, seed: int = 42) -> Dict[str, Any]:
    """
    샘플에서 N_SC개의 reasoning paths를 랜덤 샘플링
    
    Args:
        sample: 원본 샘플 데이터
        n_sc: 샘플링할 reasoning path 개수
        seed: 랜덤 시드
        
    Returns:
        Dict: 샘플링된 reasoning paths를 포함한 새로운 샘플
    """
    # 원본 샘플 복사
    new_sample = sample.copy()
    
    # Reasoning Paths 추출
    reasoning_paths = sample.get('Reasoning Paths', [])
    
    if not reasoning_paths:
        # 기존 형식 (CoT_0, CoT_1, ...) 처리
        reasoning_paths = []
        cot_count = 0
        while f'CoT_{cot_count}' in sample:
            path_entry = {}
            path_entry[f'CoT_{cot_count}'] = sample.get(f'CoT_{cot_count}', '')
            path_entry[f'Final Answer_{cot_count}'] = sample.get(f'Final Answer_{cot_count}', '')
            if f'Tokens_{cot_count}' in sample:
                path_entry[f'Tokens_{cot_count}'] = sample.get(f'Tokens_{cot_count}')
            if f'input_tokens_{cot_count}' in sample:
                path_entry[f'input_tokens_{cot_count}'] = sample.get(f'input_tokens_{cot_count}')
            reasoning_paths.append(path_entry)
            cot_count += 1
    
    if not reasoning_paths:
        return new_sample
    
    # N_SC개 샘플링
    available_n = len(reasoning_paths)
    if available_n == 0:
        return new_sample
    
    # N_SC가 available_n보다 크면 모두 사용
    if n_sc >= available_n:
        sampled_paths = reasoning_paths
    else:
        # 랜덤 샘플링
        random.seed(seed)
        sampled_indices = random.sample(range(available_n), n_sc)
        sampled_paths = [reasoning_paths[i] for i in sorted(sampled_indices)]
    
    # Reasoning Paths 재인덱싱 (CoT_0, CoT_1, ...)
    reindexed_paths = []
    for new_idx, path_entry in enumerate(sampled_paths):
        new_path_entry = {}
        for key, value in path_entry.items():
            # CoT_, Tokens_, Final Answer_, input_tokens_ 키 재인덱싱
            if key.startswith('CoT_'):
                new_path_entry[f'CoT_{new_idx}'] = value
            elif key.startswith('Tokens_'):
                new_path_entry[f'Tokens_{new_idx}'] = value
            elif key.startswith('Final Answer_'):
                new_path_entry[f'Final Answer_{new_idx}'] = value
            elif key.startswith('input_tokens_'):
                new_path_entry[f'input_tokens_{new_idx}'] = value
            else:
                new_path_entry[key] = value
        reindexed_paths.append(new_path_entry)
    
    # 새로운 샘플에 업데이트
    new_sample['Reasoning Paths'] = reindexed_paths
    
    return new_sample


def create_ours_json(input_json_path: str, number_of_paths_dict: Dict[str, int],
                    output_json_path: str, seed: int = 42, max_n: int = 40) -> Dict[str, Any]:
    """
    number_of_paths 기반으로 동적 샘플링하여 새로운 JSON 파일 생성
    
    Args:
        input_json_path: 입력 JSON 파일 경로 (SC N=40)
        number_of_paths_dict: 샘플 ID -> number_of_paths 값 매핑
        output_json_path: 출력 JSON 파일 경로
        seed: 랜덤 시드
        max_n: 최대 샘플 수 (기본값: 40)
        
    Returns:
        Dict: 통계 정보 (avg_n_sc, total_samples, etc.)
    """
    # 입력 JSON 파일 읽기
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    new_data = []
    n_sc_list = []
    missing_count = 0
    
    for sample in data:
        sample_id = str(sample.get('id', ''))
        
        # number_of_paths 값 가져오기
        if sample_id not in number_of_paths_dict:
            # ID가 없으면 기본값 사용 (max_n = 최대 샘플 수 사용)
            number_of_paths = max_n
            missing_count += 1
        else:
            number_of_paths = number_of_paths_dict[sample_id]
        
        # N_SC 계산 (number_of_paths를 그대로 사용)
        n_sc = calculate_n_sc(number_of_paths, max_n=max_n)
        n_sc_list.append(n_sc)
        
        # Reasoning paths 샘플링
        new_sample = sample_reasoning_paths(sample, n_sc, seed=seed)
        new_data.append(new_sample)
    
    # 출력 JSON 파일 저장
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)
    
    stats = {
        'total_samples': len(data),
        'avg_n_sc': sum(n_sc_list) / len(n_sc_list) if n_sc_list else 0.0,
        'min_n_sc': min(n_sc_list) if n_sc_list else 0,
        'max_n_sc': max(n_sc_list) if n_sc_list else 0,
        'missing_count': missing_count
    }
    
    return stats


def evaluate_ours_json(input_json_path: str,
                       dataset_name: str,
                       number_of_paths_dict: Dict[str, int],
                       *,
                       pool_size: Optional[int] = None,
                       subset_n: int = 40,
                       num_trials: int = 10,
                       max_n: int = 40) -> Dict[str, Any]:
    """
    입력 JSON 파일에서 동적으로 샘플링하여 Ours 메트릭 계산 (부트스트래핑 적용)
    JSON 파일을 생성하지 않고 평가 시점에 동적으로 샘플링
    
    모든 메트릭(SC, AC, ESC, DSC, Ours)이 동일한 방식으로 시작:
    1. pool에서 subset_n(40)개를 먼저 샘플링
    2. 그 40개 안에서 각 방법론의 알고리즘을 적용:
       - SC: 40개 모두 사용
       - AC: 40개에서 시작하여 동적으로 중단
       - ESC: 40개에서 시작하여 동적으로 중단
       - DSC: 40개에서 시작하여 동적으로 중단
       - Ours: 40개에서 N_SC개 샘플링 (number_of_paths 기반)
    
    Args:
        input_json_path: 입력 JSON 파일 경로 (Stage1 pool JSON)
        dataset_name: 데이터셋 이름
        number_of_paths_dict: 샘플 ID -> number_of_paths 값 매핑
        pool_size: 사용할 reasoning chain 풀 크기 (None이면 JSON에 있는 전체 사용)
        subset_n: 각 trial마다 pool_size 내에서 먼저 샘플링할 subset 크기 (기본값: 40, 모든 메트릭과 동일)
        num_trials: 부트스트래핑 반복 횟수 (기본값: 10)
        max_n: 최대 샘플 수 (기본값: 40)
        
    Returns:
        Dict: 평가 메트릭 (Ours만)
    """
    from collections import Counter
    
    # 입력 JSON 파일 읽기
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 원본 데이터를 prepare_samples 형식으로 변환하기 전에 reasoning paths 추출
    original_samples = []
    for sample in data:
        sample_id = str(sample.get('id', ''))
        
        # Reasoning Paths 추출
        reasoning_paths = sample.get('Reasoning Paths', [])
        if not reasoning_paths:
            # 기존 형식 (CoT_0, CoT_1, ...) 처리
            reasoning_paths = []
            cot_count = 0
            while f'CoT_{cot_count}' in sample:
                path_entry = {}
                path_entry[f'CoT_{cot_count}'] = sample.get(f'CoT_{cot_count}', '')
                path_entry[f'Final Answer_{cot_count}'] = sample.get(f'Final Answer_{cot_count}', '')
                if f'Tokens_{cot_count}' in sample:
                    path_entry[f'Tokens_{cot_count}'] = sample.get(f'Tokens_{cot_count}')
                if f'input_tokens_{cot_count}' in sample:
                    path_entry[f'input_tokens_{cot_count}'] = sample.get(f'input_tokens_{cot_count}')
                reasoning_paths.append(path_entry)
                cot_count += 1
        
        # number_of_paths 값 가져오기
        if sample_id not in number_of_paths_dict:
            n_sc = max_n
        else:
            n_sc = max(1, min(max_n, int(number_of_paths_dict[sample_id])))
        
        original_samples.append({
            'id': sample_id,
            'reasoning_paths': reasoning_paths,
            'n_sc': n_sc,
            'correct_answer': str(sample.get('answer', '')).strip(),  # prepare_samples와 동일하게 'answer' 필드 사용
            'input_tokens': sample.get('input_tokens', sample.get('input_tokens_0', 0))
        })
    
    trial_accuracies = []
    trial_avg_chains = []
    trial_avg_input_tokens = []
    trial_avg_output_tokens = []
    
    for trial in range(num_trials):
        correct_flags = []
        total_chains = 0
        total_input_tokens = 0
        total_output_tokens = 0
        
        for orig_sample in original_samples:
            reasoning_paths = orig_sample['reasoning_paths']
            n_sc = orig_sample['n_sc']
            correct = orig_sample['correct_answer']
            input_tokens = orig_sample['input_tokens']
            
            if not reasoning_paths:
                flag = 0
                chains_used = 0
                sampled_input_tokens = 0
                sampled_output_tokens = 0
            else:
                # stages_config(pool_size)에 따라 풀 크기 제한
                if pool_size is not None:
                    reasoning_paths = reasoning_paths[:pool_size]

                available_n = len(reasoning_paths)
                if available_n == 0:
                    flag = 0
                    chains_used = 0
                    sampled_input_tokens = 0
                    sampled_output_tokens = 0
                    correct_flags.append(flag)
                    total_chains += chains_used
                    total_input_tokens += sampled_input_tokens
                    total_output_tokens += sampled_output_tokens
                    continue

                # 모든 메트릭(SC, AC, ESC, DSC, Ours)이 동일한 방식으로 시작:
                # pool에서 40개를 먼저 샘플링하고, 그 안에서 각 방법론의 알고리즘을 적용
                # 1) (stage-40 고정 실험 기준) subset은 "앞에서부터" 순서대로 사용
                subset_k = min(max(0, int(subset_n)), available_n)
                if subset_k <= 0:
                    flag = 0
                    chains_used = 0
                    sampled_input_tokens = 0
                    sampled_output_tokens = 0
                    correct_flags.append(flag)
                    total_chains += chains_used
                    total_input_tokens += sampled_input_tokens
                    total_output_tokens += sampled_output_tokens
                    continue

                # 랜덤 샘플링 제거: stage-40에서는 pool 크기와 subset 크기가 같으므로
                # 그대로 사용해도 동일하고, tie-break도 경로 등장 순서(insertion order)로 고정됩니다.
                subset_paths = reasoning_paths[:subset_k]

                # 2) subset(40개) 내에서 Ours 알고리즘 적용: 앞에서부터 N_SC개를 순서대로 가져오기
                # 다른 메트릭들도 이 40개 안에서 각자의 알고리즘을 적용함:
                # - SC: 40개 모두 사용
                # - AC: 40개에서 시작하여 동적으로 중단
                # - ESC: 40개에서 시작하여 동적으로 중단
                # - DSC: 40개에서 시작하여 동적으로 중단
                # - Ours: 40개에서 앞에서부터 N_SC개를 순서대로 사용 (number_of_paths 기반)
                n_sc_effective = max(1, min(int(n_sc), len(subset_paths)))
                if n_sc_effective >= len(subset_paths):
                    sampled_paths = subset_paths
                else:
                    # 랜덤 샘플링 대신 앞에서부터 순서대로 가져오기
                    sampled_paths = subset_paths[:n_sc_effective]
                
                # 답변과 토큰 추출 (extract_cot_data와 동일한 로직)
                pattern_answer = re.compile(r'^Final Answer_(\d+)$')
                pattern_tokens = re.compile(r'^Tokens_(\d+)$')
                answers = []
                tokens = []
                for path_entry in sampled_paths:
                    answer_val = None
                    token_val = 0
                    for key, value in path_entry.items():
                        match_answer = pattern_answer.match(key)
                        if match_answer:
                            answer_val = str(value) if value is not None else None
                            continue
                        match_tokens = pattern_tokens.match(key)
                        if match_tokens:
                            token_val = int(value) if value is not None else 0
                    if answer_val is not None:
                        answers.append(answer_val)
                        tokens.append(token_val)
                
                if answers:
                    # SC와 동일: majority voting (정규화된 답변 기준)
                    normalized_answers = [normalize_answer_eval(ans) for ans in answers]
                    most_common_normalized = Counter(normalized_answers).most_common(1)[0][0]
                    flag = 1 if most_common_normalized == normalize_answer_eval(correct) else 0

                    chains_used = len(answers)
                    sampled_input_tokens = input_tokens
                    sampled_output_tokens = sum(tokens) if tokens else 0
                else:
                    flag = 0
                    chains_used = 0
                    sampled_input_tokens = 0
                    sampled_output_tokens = 0
            
            correct_flags.append(flag)
            total_chains += chains_used
            total_input_tokens += sampled_input_tokens
            total_output_tokens += sampled_output_tokens
        
        # 이번 trial의 메트릭
        trial_accuracy = sum(correct_flags) / len(original_samples) if original_samples else 0.0
        trial_avg_chain = total_chains / len(original_samples) if original_samples else 0.0
        trial_avg_input = total_input_tokens / len(original_samples) if original_samples else 0.0
        trial_avg_output = total_output_tokens / len(original_samples) if original_samples else 0.0
        
        trial_accuracies.append(trial_accuracy)
        trial_avg_chains.append(trial_avg_chain)
        trial_avg_input_tokens.append(trial_avg_input)
        trial_avg_output_tokens.append(trial_avg_output)
    
    # 모든 trial의 평균 (Ours)
    ours_accuracy = sum(trial_accuracies) / len(trial_accuracies) if trial_accuracies else 0.0
    ours_avg_chains = sum(trial_avg_chains) / len(trial_avg_chains) if trial_avg_chains else 0.0
    ours_avg_input_tokens = sum(trial_avg_input_tokens) / len(trial_avg_input_tokens) if trial_avg_input_tokens else 0.0
    ours_avg_output_tokens = sum(trial_avg_output_tokens) / len(trial_avg_output_tokens) if trial_avg_output_tokens else 0.0
    
    return {
        # Ours: 부트스트래핑 적용
        'ours_accuracy': ours_accuracy,
        'ours_avg_chains': ours_avg_chains,
        'ours_avg_input_tokens': ours_avg_input_tokens,
        'ours_avg_output_tokens': ours_avg_output_tokens,
        'total_samples': len(original_samples),
        'trial_accuracies': trial_accuracies,
        'trial_avg_chains': trial_avg_chains,
        'trial_avg_output_tokens': trial_avg_output_tokens,
    }


def evaluate_mmlu_pro_ours_by_category_from_data(
    json_data: List[Dict[str, Any]],
    dataset_name: str,
    number_of_paths_dict: Dict[str, int],
    *,
    pool_size: Optional[int],
    subset_n: int = 40,
    num_trials: int = 1,
    max_n: int = 40,
) -> Dict[str, Any]:
    """MMLU-Pro Stage1 리스트를 metadata.category별로 나누어 Ours 메트릭 집계 (집계 JSON용)."""
    from collections import defaultdict
    import tempfile

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in json_data:
        meta = item.get("metadata") or {}
        raw = meta.get("category")
        cat = str(raw).strip() if raw is not None and str(raw).strip() else "unknown"
        groups[cat].append(item)

    out: Dict[str, Any] = {}
    for cat in sorted(groups.keys()):
        chunk = groups[cat]
        if not chunk:
            continue
        fd, tmp_path = tempfile.mkstemp(suffix=".json", text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as wf:
                json.dump(chunk, wf, ensure_ascii=False)
            m = evaluate_ours_json(
                tmp_path,
                dataset_name,
                number_of_paths_dict,
                pool_size=pool_size,
                subset_n=subset_n,
                num_trials=num_trials,
                max_n=max_n,
            )
            out[cat] = {
                "sample_count": int(m.get("total_samples", 0)),
                "Ours": {
                    "accuracy": round(float(m["ours_accuracy"]) * 100.0, 4),
                    "avg_input_tokens": round(float(m["ours_avg_input_tokens"]), 4),
                    "avg_output_tokens": round(float(m["ours_avg_output_tokens"]), 4),
                    "avg_total_tokens": round(
                        float(m["ours_avg_input_tokens"]) + float(m["ours_avg_output_tokens"]), 4
                    ),
                    "avg_chains": round(float(m["ours_avg_chains"]), 4),
                },
            }
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    return out


def print_eval_result(dataset_name: str, json_path: str, metrics: Dict[str, Any]):
    """
    eval_result.txt 형식으로 결과 출력 (Ours만)
    
    Args:
        dataset_name: 데이터셋 이름
        json_path: JSON 파일 경로
        metrics: 평가 메트릭
    """
    print("")
    print("=" * 70)
    print(f"## Dataset: {dataset_name}")
    print("=" * 70)
    print(f"Loading JSON file: {json_path}")
    print(f"Loaded {metrics.get('total_samples', 0)} samples")
    print("")
    
    # Ours 결과 출력
    if 'ours_accuracy' in metrics:
        print(f"Ours Accuracy: {metrics.get('ours_accuracy', 0.0):.4f} "
              f"(avg_chains={metrics.get('ours_avg_chains', 0.0):.2f}, "
              f"avg_input_tokens={metrics.get('ours_avg_input_tokens', 0.0):.2f}, "
              f"avg_output_tokens={metrics.get('ours_avg_output_tokens', 0.0):.2f})")
    print("")


def save_eval_result_all(eval_result_all_path: str, dataset_name: str, trials: list, metric_name: str, fmt: str = ".4f"):
    if not trials:
        return

    if os.path.exists(eval_result_all_path):
        with open(eval_result_all_path, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
    else:
        all_lines = []

    dataset_header = f"## Dataset: {dataset_name}"
    section_start_idx = -1
    section_end_idx = -1

    for i, line in enumerate(all_lines):
        if dataset_header in line:
            section_start_idx = i
            break

    if section_start_idx >= 0:
        for i in range(section_start_idx + 1, len(all_lines)):
            if all_lines[i].startswith("## Dataset:"):
                section_end_idx = i
                break
        if section_end_idx < 0:
            section_end_idx = len(all_lines)

        method_prefix = f"{metric_name} Trials:"
        line_indices = []
        for i in range(section_start_idx, section_end_idx):
            if all_lines[i].startswith(method_prefix):
                line_indices.append(i)

        for idx in sorted(line_indices, reverse=True):
            all_lines.pop(idx)
            section_end_idx -= 1

        insert_idx = section_end_idx
        while insert_idx > section_start_idx and all_lines[insert_idx - 1].strip() == "":
            insert_idx -= 1

        trials_str = f"{metric_name} Trials: " + "[" + ", ".join(f"{x:{fmt}}" for x in trials) + "]\n"
        all_lines.insert(insert_idx, trials_str)
    else:
        new_section = f"======================================================================\n"
        new_section += f"## Dataset: {dataset_name}\n"
        new_section += f"======================================================================\n"
        new_section += f"{metric_name} Trials: " + "[" + ", ".join(f"{x:{fmt}}" for x in trials) + "]\n\n"
        all_lines.extend(new_section.splitlines(keepends=True))

    with open(eval_result_all_path, 'w', encoding='utf-8') as f:
        f.writelines(all_lines)

def save_eval_result(eval_result_path: str, dataset_name: str, json_path: str, metrics: Dict[str, Any], 
                     metric_name: str = "Ours"):
    """
    eval_result.txt 파일에 결과 추가 (기존 데이터셋 섹션에 추가)
    
    Args:
        eval_result_path: eval_result.txt 파일 경로
        dataset_name: 데이터셋 이름
        json_path: JSON 파일 경로
        metrics: 평가 메트릭
        metric_name: 메트릭 이름 (예: "Ours_math(p_0.5)")
    """
    # 기존 내용 읽기
    if os.path.exists(eval_result_path):
        with open(eval_result_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    else:
        lines = []
    
    # Ours 메트릭 라인 생성
    metric_lines = []
    
    # Ours (메트릭 이름 동적 사용)
    if 'ours_accuracy' in metrics:
        metric_lines.append(
            f"{metric_name}: {metrics.get('ours_accuracy', 0.0):.4f} "
            f"(avg_chains={metrics.get('ours_avg_chains', 0.0):.2f}, "
            f"avg_input_tokens={metrics.get('ours_avg_input_tokens', 0.0):.2f}, "
            f"avg_output_tokens={metrics.get('ours_avg_output_tokens', 0.0):.2f})\n"
        )
    
    # 해당 데이터셋 섹션 찾기
    dataset_header = f"## Dataset: {dataset_name}"
    section_start_idx = -1
    section_end_idx = -1
    
    for i, line in enumerate(lines):
        if dataset_header in line:
            section_start_idx = i
            break
    
    if section_start_idx >= 0:
        # 섹션 끝 찾기 (다음 데이터셋 섹션 시작 전까지)
        for i in range(section_start_idx + 1, len(lines)):
            if lines[i].startswith("## Dataset:"):
                section_end_idx = i
                break
        
        if section_end_idx < 0:
            # 마지막 섹션인 경우
            section_end_idx = len(lines)
        
        # 섹션 내에서 Ours_* 메트릭 라인들 찾기 (p별 여러 줄을 관리)
        existing_ours_lines = []
        ours_line_indices = []
        for i in range(section_start_idx, section_end_idx):
            line = lines[i]
            if re.match(r'^\s*Ours_[^(]+\((p_[0-9.]+)\):', line) and "avg_chains=" in line:
                existing_ours_lines.append(line)
                ours_line_indices.append(i)
        
        # 섹션 내에서 다른 메트릭 라인들(CoT, SC, ESC, AC 등)의 마지막 위치 찾기
        # Ours 라인들을 제거하기 전에 삽입 위치를 먼저 계산
        last_non_ours_metric_idx = section_start_idx
        for i in range(section_start_idx, section_end_idx):
            line_stripped = lines[i].strip()
            if (line_stripped and not line_stripped.startswith("=") 
                and not line_stripped.startswith("##") 
                and not line_stripped.startswith("Loading")
                and not re.match(r'^\s*Ours_[^(]+\((p_[0-9.]+)\):', lines[i])):
                last_non_ours_metric_idx = i
        
        # 기존 Ours 메트릭 라인들 제거 (나중에 정렬하여 재삽입)
        # 제거된 라인 수를 세어서 insert_idx 조정
        num_removed = 0
        for idx in sorted(ours_line_indices, reverse=True):
            lines.pop(idx)
            num_removed += 1
            # 제거된 라인 이후의 인덱스 조정
            if idx <= last_non_ours_metric_idx:
                last_non_ours_metric_idx -= 1
        
        # 새로운 메트릭 라인 추가 또는 기존 라인 교체
        # 같은 p 라벨이면 교체, 없으면 추가
        for metric_line in metric_lines:
            # metric_line은 "Ours_<probe>(p_x): ..." 형식
            pattern = metric_line.split(":", 1)[0] + ":"
            replaced = False
            for i, orig_line in enumerate(existing_ours_lines):
                if orig_line.startswith(pattern):
                    existing_ours_lines[i] = metric_line
                    replaced = True
                    break
            if not replaced:
                existing_ours_lines.append(metric_line)

        # Ours 라인 정렬: probe 그룹명 -> p 값 순서
        def _sort_key(line: str):
            m = re.search(r'^(Ours_[^(]+)\\(p_([0-9.]+)\\):', line.strip())
            if m:
                try:
                    return (m.group(1), float(m.group(2)))
                except Exception:
                    return (line, 1e9)
            return (line, 1e9)

        existing_ours_lines.sort(key=_sort_key)
        
        # 삽입 위치: 다른 메트릭 라인들 다음
        insert_idx = last_non_ours_metric_idx + 1
        
        # 정렬된 Ours 메트릭 라인들을 삽입 위치에 삽입
        for i, metric_line in enumerate(existing_ours_lines):
            lines.insert(insert_idx + i, metric_line)
        
        # 빈 줄 추가 (섹션 끝 유지)
        if insert_idx < len(lines) and lines[insert_idx].strip() != "":
            lines.insert(insert_idx, "\n")
    else:
        # 데이터셋 섹션이 없으면 새로 추가
        metric_lines_str = "".join(metric_lines) if metric_lines else ""
        new_section = f"""
======================================================================
## Dataset: {dataset_name}
======================================================================
Loading JSON file: {json_path}
Loaded {metrics.get('total_samples', 0)} samples

{metric_lines_str}
"""
        lines.extend(new_section.splitlines(keepends=True))
        if not lines[-1].endswith('\n'):
            lines.append('\n')
    
    # 파일에 저장
    os.makedirs(os.path.dirname(eval_result_path), exist_ok=True)
    with open(eval_result_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    if os.path.basename(eval_result_path).startswith("eval_single_") and eval_result_path.endswith(".txt"):
        print(f"\n[eval_single] 저장 (eval_result.txt 미수정): {eval_result_path}")
    else:
        print(f"\n[eval_result.txt] 저장: {eval_result_path}")


def process_single_dataset(results_dir: str, model_name: str, dataset_name: str,
                          number_of_paths_dict: Dict[str, int], stages_config: str = "stages-40",
                          metric_name: str = "Ours",
                          run_single: bool = False) -> Optional[Dict[str, Any]]:
    """
    단일 모델/데이터셋에 대해 처리
    
    Args:
        results_dir: results 디렉토리 경로
        model_name: 모델 이름
        dataset_name: 데이터셋 이름
        number_of_paths_dict: 샘플 ID -> number_of_paths 값 매핑
        stages_config: stages 설정
        
    Returns:
        Dict: 처리 결과 (성공 시), None (실패 시)
    """
    # SC(n=40) 파일 찾기
    input_json = find_sc40_json(results_dir, model_name, dataset_name, stages_config)
    if input_json is None:
        print(f"  경고: SC(n=40) 파일을 찾을 수 없습니다 (모델: {model_name}, 데이터셋: {dataset_name})")
        return None
    
    print(f"  입력 파일: {input_json}")
    
    # 통계 정보 계산 (출력용)
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    n_sc_list = []
    missing_count = 0
    for sample in data:
        sample_id = str(sample.get('id', ''))
        if sample_id not in number_of_paths_dict:
            n_sc_list.append(40)
            missing_count += 1
        else:
            n_sc = max(1, min(40, int(number_of_paths_dict[sample_id])))
            n_sc_list.append(n_sc)
    
    stats = {
        'total_samples': len(data),
        'avg_n_sc': sum(n_sc_list) / len(n_sc_list) if n_sc_list else 0.0,
        'min_n_sc': min(n_sc_list) if n_sc_list else 0,
        'max_n_sc': max(n_sc_list) if n_sc_list else 0,
        'missing_count': missing_count
    }
    
    print(f"    총 샘플 수: {stats['total_samples']}")
    print(f"    평균 N_SC: {stats['avg_n_sc']:.2f}")
    print(f"    N_SC 범위: {stats['min_n_sc']} ~ {stats['max_n_sc']}")
    if stats['missing_count'] > 0:
        print(f"    경고: number_of_paths 값이 없는 샘플 {stats['missing_count']}개 (기본값 40 사용)")
    
    # 평가 수행 (부트스트래핑 적용, Ours만 측정, JSON 파일 생성 없이 동적 샘플링)
    pool_size = extract_pool_size_from_stages_config(stages_config)
    metrics = evaluate_ours_json(
        input_json,
        dataset_name,
        number_of_paths_dict,
        pool_size=pool_size,
        subset_n=40,
        num_trials=1,
        max_n=40,
    )
    
    # eval_result.txt 경로 찾기
    # run_eval.py와 동일한 경로 구조: results_dir/model_name/[mode_subdir]/stages_config/eval_result.txt
    # 입력 JSON 파일 경로에서 실제 모델 경로 추출
    input_parts = input_json.split(os.sep)
    
    # results_dir 다음 경로부터 추출
    try:
        results_dir_basename = os.path.basename(results_dir.rstrip(os.sep))
        if results_dir_basename in input_parts:
            results_idx = input_parts.index(results_dir_basename)
        else:
            # results_dir이 절대 경로인 경우
            results_idx = -1
            for i, part in enumerate(input_parts):
                if 'results' in part.lower():
                    results_idx = i
                    break
        
        if results_idx >= 0:
            # results_dir 다음이 모델 이름 (예: "Qwen")
            # 그 다음이 실제 모델 이름 (예: "Qwen2.5-7B-Instruct") 또는 mode_subdir
            model_base_parts = []
            idx = results_idx + 1
            while idx < len(input_parts) and input_parts[idx] not in [stages_config, 'Stage1', 'Stage2']:
                model_base_parts.append(input_parts[idx])
                idx += 1
            
            # model_base_parts에서 mode_subdir 제거 (thinking/non-thinking)
            filtered_parts = [p for p in model_base_parts if p not in ['thinking', 'non-thinking']]
            if filtered_parts:
                # 실제 모델 경로 구성 (예: "Qwen/Qwen2.5-7B-Instruct")
                actual_model_path = os.path.join(*filtered_parts)
                model_base_dir = os.path.join(results_dir, actual_model_path)
            else:
                # 추출 실패 시 원본 방식 사용
                model_base_dir = os.path.join(results_dir, model_name)
        else:
            model_base_dir = os.path.join(results_dir, model_name)
    except:
        # 오류 발생 시 원본 방식 사용
        model_base_dir = os.path.join(results_dir, model_name)
    
    # thinking/non-thinking 모드 확인
    if 'thinking' in input_parts or 'non-thinking' in input_parts:
        mode_subdir = 'thinking' if 'thinking' in input_parts else 'non-thinking'
        model_base_dir = os.path.join(model_base_dir, mode_subdir)
    
    # eval_result.txt 경로: model_base_dir/stages_config/eval_result.txt
    eval_result_path = os.path.join(model_base_dir, stages_config, 'eval_result.txt')
    eval_result_all_path = os.path.join(model_base_dir, stages_config, 'eval_result_all.txt')

    # ------------------------------------------------------------------
    # 단일 실험 모드: eval_result.txt는 수정하지 않고, eval_single_{seed}.txt에만 기록
    # - eval_single_{seed}.txt는 사전에 존재해야 함
    # - 같은 디렉터리에 다른 seed의 eval_single_*.txt가 있으면 seed 불일치로 간주하고 중단
    # ------------------------------------------------------------------
    # B 방향(결정론)에서는 Ours 계산용 seed/num_trials가 필요 없고,
    # 단지 --run_single 모드에서 eval_single 파일명을 고르기 위해 고정 seed만 사용합니다.
    single_filename_seed = 137

    if run_single:
        import glob
        eval_dir = os.path.dirname(eval_result_path)
        existing_single_files = sorted(glob.glob(os.path.join(eval_dir, "eval_single_*.txt")))
        target_single = os.path.join(eval_dir, f"eval_single_{single_filename_seed}.txt")

        if not os.path.exists(target_single):
            print(f"[ERROR] --run_single: 사전에 저장된 eval_single 파일이 없습니다: {target_single}")
            if existing_single_files:
                print("[ERROR] 현재 디렉터리의 eval_single 후보:")
                for p in existing_single_files:
                    print(f"  - {p}")
            raise SystemExit(1)

        eval_result_path = target_single
    
    # 이미 해당 데이터셋과 metric_name이 둘 다 작성되어 있는지 확인
    def _has_metric_in_file(filepath: str, ds_name: str, m_name: str) -> bool:
        if not os.path.exists(filepath):
            return False
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 특정 데이터셋 섹션 찾기
        ds_pattern = rf"## Dataset:\s*{re.escape(ds_name)}\s*\n"
        ds_match = re.search(ds_pattern, content)
        if not ds_match:
            return False
        
        # 데이터셋 섹션 이후부터 다음 데이터셋 섹션 전까지에서 metric_name 찾기
        start_idx = ds_match.end()
        next_ds_pattern = r"## Dataset:"
        next_ds_match = re.search(next_ds_pattern, content[start_idx:])
        
        end_idx = start_idx + next_ds_match.start() if next_ds_match else len(content)
        section_content = content[start_idx:end_idx]
        
        # 해당 metric 이름이 섹션 안에 존재하는지 확인
        return re.search(rf"^\s*{re.escape(m_name)}", section_content, flags=re.MULTILINE) is not None

    # 스킵 로직
    if run_single:
        if _has_metric_in_file(eval_result_path, dataset_name, metric_name):
            print(f"      [스킵] {metric_name}의 {dataset_name} 결과가 이미 작성되어 있습니다. (eval_single)")
            return {
                'model_name': model_name,
                'dataset_name': dataset_name,
                'metrics': {},
                'stats': stats,
                'skipped': True
            }
    else:
        # eval_result_all.txt가 아예 없거나, 그 안에 기록이 없는 경우에만 실행 (스킵 방지)
        if _has_metric_in_file(eval_result_path, dataset_name, metric_name) and os.path.exists(eval_result_all_path):
            if _has_metric_in_file(eval_result_all_path, dataset_name, metric_name):
                print(f"      [스킵] {metric_name}의 {dataset_name} 결과가 이미 작성되어 있습니다.")
                return {
                    'model_name': model_name,
                    'dataset_name': dataset_name,
                    'metrics': {},
                    'stats': stats,
                    'skipped': True
                }

    # 평가 수행 (부트스트래핑 적용, Ours만 측정, JSON 파일 생성 없이 동적 샘플링)
        pool_size = extract_pool_size_from_stages_config(stages_config)
        metrics = evaluate_ours_json(
            input_json,
            dataset_name,
            number_of_paths_dict,
            pool_size=pool_size,
            subset_n=40,
            max_n=40,
        )
        
        # 결과 저장 (input_json은 참고용으로만 사용)
    save_eval_result(eval_result_path, dataset_name, input_json, metrics, metric_name=metric_name)
    if not run_single:
        if 'trial_accuracies' in metrics:
            save_eval_result_all(eval_result_all_path, dataset_name, metrics['trial_accuracies'], metric_name=metric_name)
            if 'trial_avg_chains' in metrics:
                save_eval_result_all(eval_result_all_path, dataset_name, metrics['trial_avg_chains'], metric_name=f"{metric_name} Chains", fmt=".2f")
            if 'trial_avg_input_tokens' in metrics:
                save_eval_result_all(eval_result_all_path, dataset_name, metrics['trial_avg_input_tokens'], metric_name=f"{metric_name} Input Tokens", fmt=".2f")
            if 'trial_avg_output_tokens' in metrics:
                save_eval_result_all(eval_result_all_path, dataset_name, metrics['trial_avg_output_tokens'], metric_name=f"{metric_name} Output Tokens", fmt=".2f")

    if dataset_name in ("MMLU-Pro", "MMLU-Pro_full") and metrics.get("ours_accuracy") is not None:
        try:
            from evaluation.mmlu_pro_category_aggregate import merge_mmlu_pro_category_section

            _nt = len(metrics.get("trial_accuracies") or [])
            if _nt < 1:
                _nt = 1
            _cat_ours = evaluate_mmlu_pro_ours_by_category_from_data(
                data,
                dataset_name,
                number_of_paths_dict,
                pool_size=pool_size,
                subset_n=40,
                num_trials=_nt,
                max_n=40,
            )
            _agg_path = merge_mmlu_pro_category_section(
                os.path.dirname(eval_result_path),
                "ours",
                _cat_ours,
                dataset_name=dataset_name,
                extra_meta={"source": "evaluate_ours_method", "stages_config": stages_config},
            )
            print(f"[{dataset_name}] 카테고리별 Ours 집계 저장: {_agg_path}")
        except Exception as _e:
            print(f"[경고] {dataset_name} 카테고리별 Ours 집계 저장 실패: {_e}")
    
    return {
        'model_name': model_name,
        'dataset_name': dataset_name,
        'metrics': metrics,
        'stats': stats
    }


def main():
    parser = argparse.ArgumentParser(description='η_adj 기반 동적 SC 샘플링 및 평가 (probe 폴더 사용)')
    parser.add_argument('--probe_dir', type=str, default='probe',
                       help='probe 디렉토리 경로 (기본값: probe)')
    parser.add_argument('--results_dir', type=str, default='experiments/results',
                       help='results 디렉토리 경로 (기본값: rsc_project/experiments/results)')
    parser.add_argument('--stages_config', type=str, default='stages-40',
                       help='stages 설정 (예: 40, stages-40, 100, stages-100). 기본값: stages-40')
    parser.add_argument('--run_single', action='store_true',
                       help='단일 실험 모드: eval_result.txt/eval_result_all.txt는 수정하지 않고, '
                            '사전에 존재하는 eval_single_137.txt에만 Ours를 기록합니다. '
                            '같은 폴더에 다른 seed의 eval_single_*.txt가 있으면 에러로 중단합니다.')
    parser.add_argument('--max_n', type=int, default=40,
                       help='최대 샘플 수 (기본값: 40)')
    parser.add_argument('--model', type=str, default=None,
                       help='특정 모델만 처리 (기본값: None, 모든 모델 처리)')
    parser.add_argument('--dataset', type=str, default=None,
                       help='특정 데이터셋만 처리 (기본값: None, 모든 데이터셋 처리)')
    
    args = parser.parse_args()

    def normalize_stages_config(value: str) -> str:
        if value is None:
            return value
        v = str(value).strip()
        if not v:
            return v
        if v.isdigit():
            return f"stages-{v}"
        return v

    args.stages_config = normalize_stages_config(args.stages_config)
    
    # 프로젝트 루트 (rsc_project) 기준 경로 설정
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # probe 디렉토리 경로
    # - 절대 경로면 그대로 사용
    # - 상대 경로면 기본적으로 rsc_project/probe/{probe_dir}를 사용
    #   (단, probe/... 형태로 이미 넘긴 경우는 rsc_project/{probe_dir} 사용)
    if not os.path.isabs(args.probe_dir):
        normalized_probe_arg = args.probe_dir.replace("\\", "/").strip("/")
        if normalized_probe_arg == "probe" or normalized_probe_arg.startswith("probe/"):
            probe_dir = os.path.join(project_root, args.probe_dir)
        else:
            probe_dir = os.path.join(project_root, "probe", args.probe_dir)
    else:
        probe_dir = args.probe_dir

    # 결과 라벨 prefix용 그룹명 (예: math, math2500)
    probe_group_name = os.path.basename(os.path.normpath(probe_dir))
    
    # results 디렉토리 경로 (상대 경로인 경우 rsc_project 기준)
    if not os.path.isabs(args.results_dir):
        results_dir = os.path.join(project_root, args.results_dir)
    else:
        results_dir = args.results_dir
    
    # probe 데이터 로드
    print(f"probe 디렉토리에서 데이터 로드: {probe_dir}")
    probe_data = load_probe_data(probe_dir)
    
    if not probe_data:
        print("  Error: probe 데이터를 찾을 수 없습니다")
        return
    
    print(f"  로드된 모델 수 (probe 기준): {len(probe_data)}")
    
    # results 디렉토리에서 실제 존재하는 모델 경로 탐색 (정보용)
    if os.path.isdir(results_dir):
        results_models = set()
        for root, dirs, files in os.walk(results_dir):
            # stages-40/Stage1/*.json 구조를 가진 디렉토리만 모델로 간주
            if args.stages_config in root and "Stage1" in root:
                rel_path = os.path.relpath(root, results_dir)
                # 모델 경로 부분만 추출 (stages-40/Stage1 이전까지)
                parts = rel_path.split(os.sep)
                if args.stages_config in parts:
                    idx = parts.index(args.stages_config)
                    model_path_parts = parts[:idx]
                    if model_path_parts:
                        results_models.add(os.path.join(*model_path_parts))
        if results_models:
            print(f"\nresults 디렉토리에서 발견된 모델 경로 수: {len(results_models)}")
            for mp in sorted(results_models):
                print(f"  - {mp}")
        else:
            print(f"\n경고: results 디렉토리({results_dir})에서 stages 결과를 가진 모델을 찾지 못했습니다.")
    
    total_processed = 0
    total_failed = 0
    
    # 각 모델/데이터셋 조합에 대해 처리
    # 모델 필터링: --model 인자가 있으면 우선 사용, 없으면 상단의 MODELS 리스트 사용
    selected_models = None
    if args.model is not None:
        selected_models = [args.model]
    elif MODELS is not None:
        selected_models = MODELS
    
    for model_name, datasets in probe_data.items():
        # 모델 필터링 적용 (probe 모델 이름과 MODELS 리스트의 모델 이름 매칭)
        if selected_models is not None:
            # probe 모델 이름과 MODELS 리스트의 모델 이름을 비교
            # 예: "gemma-3-4b-it" <-> "google/gemma-3-4b-it"
            matched = False
            for selected_model in selected_models:
                # 정확히 일치하거나, 벤더 접두사를 제거/추가한 버전과 일치하는지 확인
                if model_name == selected_model:
                    matched = True
                    break
                # "gemma-3-4b-it" <-> "google/gemma-3-4b-it" 매칭
                if "gemma" in model_name.lower() and "gemma" in selected_model.lower():
                    model_base = model_name.replace("google/", "").replace("google\\", "")
                    selected_base = selected_model.replace("google/", "").replace("google\\", "")
                    if model_base == selected_base:
                        matched = True
                        break
                # "Qwen2.5-7B-Instruct" <-> "Qwen/Qwen2.5-7B-Instruct" 매칭
                if "qwen" in model_name.lower() and "qwen" in selected_model.lower():
                    model_base = model_name.replace("Qwen/", "").replace("Qwen\\", "")
                    selected_base = selected_model.replace("Qwen/", "").replace("Qwen\\", "")
                    if model_base == selected_base:
                        matched = True
                        break
                # Llama 매칭
                if "llama" in model_name.lower() and "llama" in selected_model.lower():
                    model_base = model_name.replace("meta-llama/", "").replace("meta-llama\\", "")
                    selected_base = selected_model.replace("meta-llama/", "").replace("meta-llama\\", "")
                    if model_base == selected_base:
                        matched = True
                        break
            if not matched:
                continue
        
        print(f"\n{'='*70}")
        print(f"모델: {model_name}")
        print(f"{'='*70}")
        
        print(f"  데이터셋 수: {len(datasets)}")

        # p_value별로 평가하고 eval_result.txt에 p별 라인으로 기록
        for p_value, ds_map in datasets.items():
            for dataset_name, number_of_paths_dict in ds_map.items():
                if args.dataset is not None and dataset_name != args.dataset:
                    continue

                metric_name = f"Ours_{probe_group_name}({p_value})"
                print(f"\n    p 값: {p_value}")
                print(f"    데이터셋: {dataset_name}")
                print(f"      number_of_paths 값: {len(number_of_paths_dict)}개")

                result = process_single_dataset(
                    results_dir, model_name, dataset_name, number_of_paths_dict,
                    stages_config=args.stages_config,
                    metric_name=metric_name,
                    run_single=args.run_single,
                )

                if result:
                    total_processed += 1
                    print(f"      ✓ 완료")
                else:
                    total_failed += 1
                    print(f"      ✗ 실패")
    
    print(f"\n{'='*70}")
    print(f"전체 처리 완료!")
    print(f"  성공: {total_processed}개")
    print(f"  실패: {total_failed}개")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
