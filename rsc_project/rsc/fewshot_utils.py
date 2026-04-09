"""
Few-shot Utilities 모듈

이 모듈은 Few-shot 예제를 구성하는 기능을 제공합니다.
기존 src/SC_generator_ours.py의 build_fewshot_context 함수를 활용합니다.
"""

from typing import List, Dict, Any, Optional
import sys
import os

# 기존 코드 경로 추가 (src 디렉토리)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

try:
    from fewshot import (
        AIME_fewshot,
        MATH_fewshot,
        MMLU_fewshot,
        GPQA_fewshot,
    )
    
    DATASET_FEWSHOT_MAP: Dict[str, List[Dict[str, Any]]] = {
        'AIME_2024': AIME_fewshot,
        'AIME_2025': AIME_fewshot,
        'MATH500_test': MATH_fewshot,
        'MATH_train': MATH_fewshot,
        'MATH2500': MATH_fewshot,
        'AMC23': AIME_fewshot,
        'MMLU-Pro': MMLU_fewshot,
        'MMLU-Pro_full': MMLU_fewshot,
        'GPQA-Diamond': GPQA_fewshot,
    }
except ImportError:
    # fewshot 모듈이 없는 경우 빈 딕셔너리 사용
    DATASET_FEWSHOT_MAP: Dict[str, List[Dict[str, Any]]] = {}


def format_options(options: Optional[List[str]]) -> str:
    """
    선택지 포맷팅 (기존 코드의 format_options 함수)
    
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


def build_fewshot_context(dataset_key: str, disabled: bool = False) -> str:
    """
    Few-shot 컨텍스트 구성 (기존 SC_generator_ours.py의 build_fewshot_context 함수)
    
    Args:
        dataset_key: 데이터셋 키 ('AIME_2024', 'AIME_2025', 'MATH500_test', 'MATH_train', 'MATH2500', 'AMC23', 'MMLU-Pro', 'GPQA-Diamond')
        disabled: Few-shot 비활성화 여부
        
    Returns:
        str: 구성된 Few-shot 컨텍스트
    """
    if disabled:
        return ''
    examples = DATASET_FEWSHOT_MAP.get(dataset_key, [])
    formatted_examples = []
    for idx, example in enumerate(examples, start=1):
        parts = [f"### Example {idx}"]
        question_block = example.get('question', '')
        
        # MMLU-Pro 계열의 경우 question 필드에 이미 "Question:"이 포함되어 있을 수 있음
        if dataset_key in ('MMLU-Pro', 'MMLU-Pro_full') and question_block.startswith('Question:'):
            # 이미 "Question:"이 있으면 그대로 사용
            pass
        else:
            # 다른 데이터셋의 경우 "Question:" 접두사 추가
            question_block = f"Question: {question_block}"
        
        options = example.get('options')
        if options:
            question_block = f"{question_block}\n{format_options(options)}"
        parts.append(question_block.strip())
        solution = example.get('solution')
        if solution:
            parts.append(f"Answer: {solution.strip()}")
        answer = example.get('answer')
        if answer:
            parts.append(f"Final Answer: {answer}")
        formatted_examples.append("\n".join(parts).strip())
    return "\n\n".join(formatted_examples)

