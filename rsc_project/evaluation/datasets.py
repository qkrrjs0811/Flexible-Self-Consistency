"""
Dataset Loader 모듈

이 모듈은 다양한 수학 문제 데이터셋을 로드하는 기능을 제공합니다.
기존 src/SC_generator_ours.py의 load_jsonl 함수를 활용합니다.
"""

from typing import List, Dict, Any, Optional
import json
import os
import sys

# 기존 코드 경로 추가 (src 디렉토리)
# 주석 처리: evaluation 패키지와 src/evaluation.py 파일 이름 충돌 방지
# 필요시 나중에 활성화 가능
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))


class DatasetLoader:
    """
    데이터셋을 로드하는 클래스
    
    기존 코드의 JSONL 로딩 로직을 활용합니다.
    지원 데이터셋: AIME_2024, AIME_2025, MATH500_test, MATH_train, MATH2500, AMC23, MMLU-Pro, MMLU-Pro_full, GPQA-Diamond
    """
    
    def __init__(self, dataset_name: str, dataset_path: str = None, eval_dir: str = '../eval'):
        """
        Args:
            dataset_name: 데이터셋 이름 ('AIME_2024', 'AIME_2025', 'MATH500_test', 'MATH_train', 'MATH2500', 'AMC23', 'MMLU-Pro', 'MMLU-Pro_full', 'GPQA-Diamond')
            dataset_path: 데이터셋 파일 경로 (None이면 eval_dir에서 자동 찾기)
            eval_dir: 평가 데이터 디렉토리 (기본값: '../eval')
        """
        self.dataset_name = dataset_name
        self.eval_dir = eval_dir
        self.dataset_path = dataset_path or self._get_default_path()
        self.data = []
    
    def _get_default_path(self) -> str:
        """
        기본 데이터셋 경로 반환 (기존 코드 로직)
        
        Returns:
            str: 데이터셋 파일 경로
        """
        jsonl_filename = f"{self.dataset_name}.jsonl"
        return os.path.join(self.eval_dir, jsonl_filename)
    
    def load(self, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        데이터셋 로드 (기존 load_jsonl 함수 활용)
        
        Args:
            max_samples: 최대 샘플 수 (None이면 전체)
            
        Returns:
            List[Dict[str, Any]]: 데이터셋 샘플 리스트
                각 샘플은 {'question': str, 'answer': str, 'id': str, ...} 형식
        """
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"데이터셋 파일을 찾을 수 없습니다: {self.dataset_path}")
        
        # 기존 load_jsonl 함수 활용
        data_list = self.load_jsonl(self.dataset_path)
        
        if max_samples:
            data_list = data_list[:max_samples]
        
        return data_list
    
    def load_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        """
        JSONL 파일 로드 (기존 SC_generator_ours.py의 load_jsonl 함수)
        
        Args:
            file_path: JSONL 파일 경로
            
        Returns:
            List[Dict[str, Any]]: 로드된 데이터 리스트
        """
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    
    def format_options(self, options: Optional[List[str]]) -> str:
        """
        선택지 포맷팅 (기존 코드의 format_options 함수 활용)
        
        Args:
            options: 선택지 리스트
            
        Returns:
            str: 포맷팅된 선택지 문자열
        """
        if not options:
            return ''
        option_lines = []
        for idx, opt in enumerate(options):
            option_label = chr(65 + idx)  # A, B, C, D, ...
            option_lines.append(f"{option_label}. {opt}")
        return "Options:\n" + "\n".join(option_lines)

