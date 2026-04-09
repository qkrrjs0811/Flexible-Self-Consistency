"""
캐시 유틸리티 모듈

Stage 결과를 캐시하고 재사용하는 기능을 제공합니다.
동일한 Stage 경로(n, mode 조합)의 결과를 재사용하여 계산 비용을 절약합니다.
"""

import os
import json
import re
from typing import List, Dict, Any, Optional, Tuple


def _extract_n_from_slug(slug: str) -> Optional[int]:
    """
    Slug에서 생성 개수(n) 추출
    
    Args:
        slug: Stage slug (예: "Stage1_n40", "Stage2_collective_10_from_Stage1_n40")
        
    Returns:
        int: 추출된 개수 (없으면 None)
    """
    # Stage1_n40 형식
    match = re.search(r'Stage1_n(\d+)', slug)
    if match:
        return int(match.group(1))
    
    # Stage2_collective_10_from_... 또는 Stage2_individual_10_from_... 형식
    # 첫 번째 모드_숫자 패턴 추출 (실제 생성 개수)
    match = re.search(r'Stage\d+_(collective|individual)_(\d+)', slug)
    if match:
        return int(match.group(2))
    
    return None


def generate_stage_slug(stage_num: int, stage_config: Dict[str, Any], 
                        previous_stage_slug: Optional[str] = None) -> str:
    """
    Stage별 고유 식별자(slug) 생성
    
    Individual 모드의 경우, 설정의 n 값은 무시하고 이전 Stage의 실제 개수를 사용합니다.
    이는 Individual 모드가 항상 이전 Stage 개수만큼만 생성하기 때문입니다.
    
    Args:
        stage_num: Stage 번호
        stage_config: Stage 설정 딕셔너리
        previous_stage_slug: 이전 Stage의 slug (None이면 Stage1)
        
    Returns:
        str: Stage slug (예: "Stage1_n40", "Stage2_collective_10_from_Stage1_n40", 
             "Stage2_individual_40_from_Stage1_n40" - Individual은 이전 Stage 개수 반영)
    """
    if stage_num == 1:
        n = stage_config['n']
        return f"Stage1_n{n}"
    else:
        mode = stage_config.get('mode', 'collective')
        collective_num_paths = stage_config.get('collective_num_paths')
        
        if previous_stage_slug is None:
            raise ValueError("Stage 2 이상은 previous_stage_slug가 필요합니다.")
        
        # Individual 모드: 설정의 n 값 무시, 이전 Stage의 실제 개수 사용
        if mode == 'individual':
            previous_n = _extract_n_from_slug(previous_stage_slug)
            if previous_n is None:
                raise ValueError(f"이전 Stage slug에서 개수를 추출할 수 없습니다: {previous_stage_slug}")
            n = previous_n  # 실제 생성 개수 (이전 Stage 개수)
        else:
            # Collective 모드: 설정의 n 값 사용
            n = stage_config.get('n', 'default')
        
        # collective_num_paths가 있으면 slug에 포함
        if collective_num_paths is not None:
            slug = f"Stage{stage_num}_{mode}_{n}_from_{previous_stage_slug}_sub{collective_num_paths}"
        else:
            slug = f"Stage{stage_num}_{mode}_{n}_from_{previous_stage_slug}"
        
        return slug


def find_larger_stage1_cache(model_name: str, mode_subdir: str, dataset_name: str,
                             target_n: int, cache_dir: str = 'experiments/results_cache') -> Optional[Tuple[str, int]]:
    """
    Stage 1에서 target_n보다 큰 n 값을 가진 캐시를 찾아 반환
    
    Args:
        model_name: 모델 이름
        mode_subdir: 모드 서브디렉토리 (thinking/non-thinking, None이면 Qwen3가 아닌 모델)
        dataset_name: 데이터셋 이름
        target_n: 목표 n 값 (이보다 큰 n의 캐시를 찾음)
        cache_dir: 캐시 디렉토리 경로
        
    Returns:
        Optional[Tuple[str, int]]: (캐시 파일 경로, n 값) 또는 None
    """
    # 캐시 디렉토리 경로 구성
    base_dir = os.path.join(cache_dir, model_name)
    if mode_subdir:
        base_dir = os.path.join(base_dir, mode_subdir)
    
    if not os.path.exists(base_dir):
        return None
    
    best_cache_path = None
    best_n = None
    
    # 기존 구조 탐색: base_dir/dataset_name/Stage1_n*/{dataset_name}.json
    cache_base = os.path.join(base_dir, dataset_name)
    if os.path.exists(cache_base):
        for item in os.listdir(cache_base):
            if item.startswith('Stage1_n'):
                # n 값 추출
                match = re.search(r'Stage1_n(\d+)', item)
                if match:
                    n_value = int(match.group(1))
                    if n_value >= target_n:
                        cache_path = os.path.join(cache_base, item, f'{dataset_name}.json')
                        if os.path.exists(cache_path):
                            # 더 큰 n 값을 우선 선택
                            if best_n is None or n_value > best_n:
                                best_n = n_value
                                best_cache_path = cache_path
    
    if best_cache_path and best_n:
        return (best_cache_path, best_n)
    return None


def get_cache_path(model_name: str, mode_subdir: str, dataset_name: str, 
                   stage_slug: str, cache_dir: str = 'experiments/results_cache') -> str:
    """
    캐시 파일 경로 생성 (기존 구조)
    
    Args:
        model_name: 모델 이름
        mode_subdir: 모드 서브디렉토리 (thinking/non-thinking, None이면 Qwen3가 아닌 모델)
        dataset_name: 데이터셋 이름
        stage_slug: Stage slug
        cache_dir: 캐시 디렉토리 경로
        
    Returns:
        str: 캐시 파일 경로
    """
    # 기존 구조: cache_dir/model_name/[mode_subdir]/dataset_name/stage_slug/{dataset_name}.json
    if mode_subdir:
        cache_base = os.path.join(cache_dir, model_name, mode_subdir, dataset_name, stage_slug)
    else:
        cache_base = os.path.join(cache_dir, model_name, dataset_name, stage_slug)
    
    os.makedirs(cache_base, exist_ok=True)
    return os.path.join(cache_base, f'{dataset_name}.json')


def load_from_cache(cache_path: str) -> Optional[List[Dict[str, Any]]]:
    """
    캐시에서 데이터 로드
    
    Args:
        cache_path: 캐시 파일 경로
        
    Returns:
        Optional[List[Dict[str, Any]]]: 캐시된 데이터 (없으면 None)
    """
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data
        except Exception as e:
            print(f"[경고] 캐시 파일 읽기 실패: {cache_path}, 오류: {e}")
            return None
    return None


def save_to_cache(data: List[Dict[str, Any]], cache_path: str):
    """
    데이터를 캐시에 저장
    
    Args:
        data: 저장할 데이터 리스트
        cache_path: 캐시 파일 경로
    """
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def generate_stages_config_slug(stages: List[Dict[str, Any]], stage_slugs: Optional[List[str]] = None) -> str:
    """
    Stage 1 구성에 대한 slug 생성 (최종 결과 폴더명용)
    
    Args:
        stages: stages 설정 리스트 (Stage 1만 사용)
        stage_slugs: 사용하지 않음 (하위 호환성을 위해 유지)
        
    Returns:
        str: 구성 slug (예: "stages-40")
    """
    # Stage 1만 사용
    return f"stages-{stages[0]['n']}"


def copy_cache_to_result_dir(cache_path: str, result_path: str):
    """
    캐시 파일을 결과 디렉토리로 복사 또는 심볼릭 링크 생성
    
    Args:
        cache_path: 캐시 파일 경로
        result_path: 결과 파일 경로
    """
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"캐시 파일을 찾을 수 없습니다: {cache_path}")
    
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    
    # 파일 복사 (심볼릭 링크 대신 복사 사용 - 더 호환성 좋음)
    import shutil
    shutil.copy2(cache_path, result_path)

