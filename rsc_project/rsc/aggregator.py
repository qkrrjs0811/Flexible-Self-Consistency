"""
Self-Consistency Aggregator 모듈

이 모듈은 여러 reasoning path로부터 majority vote를 통해 최종 답변을 산출합니다.
기존 src/SC_generator_ours.py의 extract 함수들을 활용합니다.
"""

from typing import List, Dict, Any, Optional
from collections import Counter
import sys
import os
import re

# 기존 코드 경로 추가 (src 디렉토리) - 주석 처리: evaluation 패키지와 충돌 방지
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

# normalize_answer는 evaluation.metrics에서 가져오기
try:
    from evaluation.metrics import normalize_answer_eval as normalize_answer
except ImportError:
    # evaluation.metrics가 없는 경우 기본 구현 사용
    def normalize_answer(answer):
        """답변을 정규화하여 비교 가능하게 만듦 (공백 제거)."""
        if answer is None:
            return ""
        answer_str = str(answer).strip()
        answer_str = answer_str.replace(" ", "")
        return answer_str.lower()


def _basic_strip(s: str) -> str:
    """Extract 단계에서만 쓰는 '최소 정제' (정규화는 normalize_answer_eval에 맡김)."""
    if s is None:
        return ""
    s = str(s).strip()
    # 흔한 출력 포맷: "$...$" 또는 "$..." / "...$" 형태를 최소로 제거
    # (closing $가 누락된 경우도 존재: \\boxed{$125,000})
    if s.startswith("$"):
        s = s[1:].strip()
    if s.endswith("$"):
        s = s[:-1].strip()
    return s


class SCAggregator:
    """
    Self-Consistency를 위한 majority vote aggregator
    
    여러 reasoning path에서 최종 답변을 추출하고, majority vote를 통해 결정합니다.
    기존 extract_math_answer, extract_answer_gpqa, extract_answer_mmlu_pro 함수를 활용합니다.
    """
    
    def __init__(self, dataset_name: str = None):
        """
        Args:
            dataset_name: 데이터셋 이름 (None이면 자동 감지)
                         'AIME_2024', 'AIME_2025', 'MATH500_test', 'MATH_train', 'MATH2500', 'GPQA-Diamond', 'MMLU-Pro', 'MMLU-Pro_full'
        """
        self.dataset_name = dataset_name
        self.extract_func = self._get_extract_function(dataset_name)
    
    def _get_extract_function(self, dataset_name: Optional[str]):
        """
        데이터셋에 맞는 답변 추출 함수 반환 (기존 코드 로직)
        
        Args:
            dataset_name: 데이터셋 이름
            
        Returns:
            callable: 답변 추출 함수
        """
        if dataset_name in ['AIME_2024', 'AIME_2025', 'MATH500_test', 'MATH_train', 'MATH2500']:
            return self.extract_math_answer
        elif dataset_name == 'GPQA-Diamond':
            return self.extract_answer_gpqa
        elif dataset_name in ('MMLU-Pro', 'MMLU-Pro_full'):
            return self.extract_answer_mmlu_pro
        else:
            # 기본값으로 math answer 추출 사용
            return self.extract_math_answer
    
    def extract_answer(self, path: str) -> str:
        """
        단일 reasoning path에서 답변 추출
        
        Args:
            path: reasoning path 텍스트
            
        Returns:
            str: 추출된 답변 (빈 문자열일 수 있음)
        """
        answer = self.extract_func(path)
        return str(answer) if answer else ""
    
    def aggregate(self, reasoning_paths: List[str]) -> str:
        """
        여러 reasoning path로부터 majority vote를 통해 최종 답변 산출
        
        Args:
            reasoning_paths: reasoning path 리스트 (R1 또는 R2)
            
        Returns:
            str: majority vote로 결정된 최종 답변 (a1 또는 a2)
        """
        # 각 reasoning path에서 답변 추출 (기존 extract 함수 활용)
        answers = []
        for path in reasoning_paths:
            answer = self.extract_func(path)
            if answer:
                answers.append(str(answer))
        
        if not answers:
            # 답변을 추출할 수 없는 경우, 첫 번째 path의 마지막 부분을 사용
            return reasoning_paths[0].strip() if reasoning_paths else ""
        
        # Majority vote 수행 (normalize_answer를 사용하여 정규화 후 비교)
        # SC Accuracy 계산과 동일한 방식으로 정규화하여 일관성 유지
        normalized_answers = [normalize_answer(ans) for ans in answers]
        answer_counts = Counter(normalized_answers)
        majority_normalized = answer_counts.most_common(1)[0][0]
        
        # 정규화된 majority answer에 해당하는 원본 답변 반환
        # (정규화된 값이 같은 원본 답변 중 첫 번째 것을 반환)
        for ans in answers:
            if normalize_answer(ans) == majority_normalized:
                return ans
        
        # fallback: 정규화된 majority answer를 그대로 반환
        return majority_normalized
    
    def extract_math_answer(self, pred_str: Optional[str]) -> str:
        """
        수학 문제 답변 추출 (Extract 단계에서는 '답 후보를 뽑는 것'까지만 수행)
        
        주의:
        - LaTeX/수식 정규화(_strip_string), 단위 제거 등은 여기서 하지 않습니다.
        - 비교/집계는 normalize_answer_eval(=evaluation.metrics.normalize_answer_eval)에서 일괄 처리합니다.
        
        Args:
            pred_str: 모델 출력 텍스트
            
        Returns:
            str: 추출된 답변
        """
        if pred_str is None:
            return ''
        pred_str = str(pred_str).strip()
        if not pred_str:
            return ''
        
        if 'boxed' in pred_str:
            ans = pred_str.split('boxed')[-1]
            if (len(ans) and ans[0] == '{'):
                stack = 1
                a = ''
                for c in ans[1:]:
                    if (c == '{'):
                        stack += 1
                        a += c
                    elif (c == '}'):
                        stack -= 1
                        if (stack == 0):
                            break
                        a += c
                    else:
                        a += c
            else:
                a = ans.split('$')[0].strip()
            pred = a
            
        elif ('The answer is ' in pred_str):
            # "The answer is " 뒤의 텍스트를 가져오되, 마침표(.) 또는 줄바꿈(\n)까지만 추출
            answer_part = pred_str.split('The answer is ')[-1].strip()
            # 마침표나 줄바꿈으로 분리 (첫 번째 문장만)
            if '.' in answer_part:
                pred = answer_part.split('.')[0].strip()
            elif '\n' in answer_part:
                pred = answer_part.split('\n')[0].strip()
            else:
                pred = answer_part
                
        elif ('the answer is ' in pred_str):
            # "the answer is " 뒤의 텍스트를 가져오되, 마침표(.) 또는 줄바꿈(\n)까지만 추출
            answer_part = pred_str.split('the answer is ')[-1].strip()
            # 마침표나 줄바꿈으로 분리 (첫 번째 문장만)
            if '.' in answer_part:
                pred = answer_part.split('.')[0].strip()
            elif '\n' in answer_part:
                pred = answer_part.split('\n')[0].strip()
            else:
                pred = answer_part
        else:
            pattern = r'-?\d*\.?\d+'
            pred = re.findall(pattern, pred_str)
            if len(pred) >= 1:
                pred = pred[-1]
            else:
                pred = ''
        
        if pred != "":
            if pred[-1] == ".":
                pred = pred[:-1]
            if pred != "" and pred[-1] == "/":
                pred = pred[:-1]
        return _basic_strip(pred)

    def extract_answer_gpqa(self, content: Optional[str]) -> Optional[str]:
        """
        GPQA 데이터셋 답변 추출 (기존 SC_generator_ours.py의 extract_answer_gpqa 활용)
        
        Args:
            content: 모델 출력 텍스트
            
        Returns:
            str: 추출된 답변 (A, B, C, D 중 하나)
        """
        if content is None:
            return None

        # First, check for boxed format
        if 'boxed' in content:
            ans = content.split('boxed')[-1]
            if len(ans) and ans[0] == '{':
                stack = 1
                a = ''
                for c in ans[1:]:
                    if c == '{':
                        stack += 1
                        a += c
                    elif c == '}':
                        stack -= 1
                        if stack == 0:
                            break
                        a += c
                    else:
                        a += c
            else:
                a = ans.split('$')[0].strip()
            a = _basic_strip(a)
            
            if a:
                text_match = re.search(r'\\text(?:bf|it|tt)?\{([^}]+)\}', a)
                if text_match:
                    a = text_match.group(1)
                
                boxed_match = re.search(r'([A-D])[\.]?', a.strip())
                if boxed_match:
                    return boxed_match.group(1).upper()

        # Regex to find "Answer: $LETTER"
        answer_keyword_pattern = r"(?i)(?:answer|final answer|correct answer)(?:\s+\w+)*\s*[:=\-]?\s*"
        answer_matches = list(re.finditer(answer_keyword_pattern, content))
        
        if answer_matches:
            for answer_match in reversed(answer_matches):
                text_after = content[answer_match.end():]
                letter_match = re.search(r'[\s\$]*([A-D])[\.]?(?:\s|$)', text_after)
                if letter_match:
                    return letter_match.group(1).upper()
        
        # Fallback
        ANSWER_PATTERN = r"(?i)(?:answer|final answer|correct answer)(?:\s+\w+)*\s*[:=\-]?\s*[\$]?\s*([A-D])[\$]?[\.]?(?:\s|$)"
        matches = list(re.finditer(ANSWER_PATTERN, content))
        
        if matches:
            last_match = matches[-1]
            extracted_letter = last_match.group(1).upper()
            return extracted_letter
        else:
            stripped_content = content.strip()
            if stripped_content and stripped_content[-1].upper() in ['A', 'B', 'C', 'D']:
                last_part = stripped_content[-15:].lower()
                if any(keyword in last_part for keyword in ['answer']):
                    return stripped_content[-1].upper()
        return None
    
    def extract_answer_mmlu_pro(self, content: Optional[str]) -> Optional[str]:
        """
        MMLU-Pro 데이터셋 답변 추출 (기존 SC_generator_ours.py의 extract_answer_mmlu_pro 활용)
        
        Args:
            content: 모델 출력 텍스트
            
        Returns:
            str: 추출된 답변 (A-J 중 하나)
        """
        if content is None:
            return None

        ANSWER_PATTERN = r"((?:[A-J]\b))"
        matches = list(re.finditer(ANSWER_PATTERN, content, flags=re.IGNORECASE))

        if matches:
            last = matches[-1]
            return last.group(1).upper()

        return None
    
    def get_answer_distribution(self, reasoning_paths: List[str]) -> Dict[str, int]:
        """
        reasoning path들의 답변 분포를 반환
        
        Args:
            reasoning_paths: reasoning path 리스트
            
        Returns:
            Dict[str, int]: 답변별 빈도수
        """
        answers = []
        for path in reasoning_paths:
            answer = self.extract_func(path)
            if answer:
                answers.append(str(answer))
        
        return dict(Counter(answers))

