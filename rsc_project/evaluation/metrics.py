"""
Evaluation Metrics 모듈

이 모듈은 RSC 알고리즘의 성능을 평가하기 위한 메트릭을 계산합니다.
src/evaluation.py의 평가 방식을 활용하여 CoT, SC, ESC, AC 메트릭을 계산합니다.
"""

from typing import List, Dict, Any, Optional
import re
import os
import sys
import json
import random
from collections import Counter


# ============================================================================
# 수식/LaTeX 문자열 정규화 유틸리티 (src/utils.py 320-387 라인과 동일 로직)
# ============================================================================

def extract_pool_size_from_stages_config(stages_config_slug: str) -> int:
    """
    stages_config_slug에서 풀 크기 추출
    
    Args:
        stages_config_slug: "stages-40", "stages-100" 등
    
    Returns:
        int: 풀 크기 (예: 40, 100). 기본값은 40
    """
    if not stages_config_slug:
        return 40
    
    # "stages-40" -> 40, "stages-100" -> 100
    match = re.search(r'stages-(\d+)', stages_config_slug)
    if match:
        return int(match.group(1))
    return 40  # 기본값


def _fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except Exception:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string: str) -> str:
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except Exception:
        return string


def _remove_right_units(string: str) -> str:
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        # assert len(splits) == 2
        return splits[0]
    else:
        return string


def _fix_sqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if not split:
            continue
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _strip_string(string: str) -> str:
    """
    수식/LaTeX 형식 문자열을 비교 용도로 정규화.

    src/utils.py의 `_strip_string`과 동일한 로직을 사용.
    """
    # linebreaks
    string = string.replace("\n", "")

    # remove thousands separators in numbers (e.g., "125,000" -> "125000")
    # Important for GSM8K-style currency answers inside \\boxed{$125,000}
    string = string.replace(",", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs (both escaped LaTeX "\\$" and raw "$")
    string = string.replace("\\$", "")
    string = string.replace("$", "")

    # remove units (on the right): \text{ cm} (공백 있음)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


# adaptive_consistency 모듈은 optional (없으면 직접 구현한 버전 사용)
try:
    from adaptive_consistency import AC, BetaStoppingCriteria
    HAS_AC_MODULE = True
except ImportError:
    HAS_AC_MODULE = False
    # BetaStoppingCriteria 직접 구현
    class BetaStoppingCriteria:
        """Beta 기준으로 중단 조건을 판단하는 클래스 (adaptive_consistency 모듈 없이 사용).
        
        현재까지의 답변들 중에서 가장 많이 나온 답변의 비율이 beta 이상이면 중단.
        """
        def __init__(self, beta: float):
            """
            Args:
                beta: 중단 기준 비율 (0.0 ~ 1.0, 예: 0.95)
            """
            self.beta = beta
        
        def should_stop(self, answers: List[str]) -> bool:
            """
            현재까지의 답변들로 중단 조건을 확인.
            
            Args:
                answers: 현재까지 생성된 답변 리스트
                
            Returns:
                bool: True면 중단, False면 계속
            """
            if not answers:
                return False
            
            # 정규화된 답변으로 Counter 생성
            normalized_answers = [normalize_answer_eval(ans) for ans in answers]
            counter = Counter(normalized_answers)
            
            # 가장 많이 나온 답변의 개수와 비율 계산
            if counter:
                most_common_count = counter.most_common(1)[0][1]
                ratio = most_common_count / len(answers)
                # beta 이상이면 중단
                return ratio >= self.beta
            
            return False
    
    # AC 클래스 직접 구현
    class AC:
        """Adaptive Consistency 클래스 (adaptive_consistency 모듈 없이 사용).
        
        BetaStoppingCriteria를 사용하여 적응적으로 중단하는 로직을 구현.
        """
        def __init__(self, stop_criteria: BetaStoppingCriteria, max_gens: int = 40):
            """
            Args:
                stop_criteria: 중단 조건 (BetaStoppingCriteria)
                max_gens: 최대 생성 개수 (기본값: 40)
            """
            self.stop_criteria = stop_criteria
            self.max_gens = max_gens
        
        def should_stop(self, answers: List[str]) -> bool:
            """
            중단 조건 확인.
            
            Args:
                answers: 현재까지 생성된 답변 리스트
                
            Returns:
                bool: True면 중단, False면 계속
            """
            # 최대 개수에 도달했으면 중단
            if len(answers) >= self.max_gens:
                return True
            
            # BetaStoppingCriteria로 중단 조건 확인
            return self.stop_criteria.should_stop(answers)


# ============================================================================
# src/evaluation.py의 로직을 직접 구현 (독립적으로 동작)
# ============================================================================

def normalize_answer_eval(answer):
    """답변을 정규화하여 비교 가능하게 만듦.
    
    - 복합 수식(\\sqrt, \\frac, ^, { 등)이 있으면: 숫자만 추출하지 않고 _strip_string 결과 전체 반환.
      (예: "12 \\sqrt{3}" -> "12\\sqrt{3}", "6\\sqrt{7}" -> "6\\sqrt{7}" — 마지막 숫자만 쓰면 3, 7로 잘못 됨)
    - 단순 숫자만 있는 경우: 숫자만 추출 (예: "70.0", " 70 " -> "70").
    - 숫자가 없으면: 소문자·공백 정리.
    
    Args:
        answer: 원본 답변 (str 또는 다른 타입)
        
    Returns:
        str: 정규화된 답변
    """
    if answer is None:
        return ""
    # LaTeX/수식 문자열을 먼저 정규화
    answer_str = _strip_string(str(answer).strip())
    
    # \text{Yes} 같은 단순 텍스트 케이스는 내부 문자열만 추출
    text_only_match = re.fullmatch(r'\\text\{([^}]*)\}', answer_str)
    if text_only_match:
        answer_str = text_only_match.group(1).strip()

    # Yes/No 류는 텍스트가 길어도 일관되게 판정되도록 우선 처리
    lower_str = answer_str.lower()
    if re.search(r'\byes\b', lower_str):
        return "yes"
    if re.search(r'\bno\b', lower_str):
        return "no"
    
    # 복합 수식: \sqrt, \frac, ^, \{,\}, 또는 LaTeX 명령(\\+영문, 예: \infty, \cup, \cap)이 있으면
    # 마지막 숫자만 쓰지 않고 전체 반환. "(-\\infty,-4)\\cup(-4,\\infty)" -> 숫자만 쓰면 -4로 잘못됨.
    _COMPOUND_MARKERS = ('\\sqrt', '\\frac', '^', '\\cdot', '\\times', '\\div', '\\pi')
    is_compound = (
        any(m in answer_str for m in _COMPOUND_MARKERS) or
        ('{' in answer_str and '}' in answer_str) or
        bool(re.search(r'\\[a-zA-Z]', answer_str))  # \infty, \cup, \cap, \leq, \ldots 등
    )
    if is_compound:
        return answer_str
    
    # 단순 숫자만 있는 경우: 숫자만 추출 (기존 동작)
    numbers = re.findall(r'-?\d+\.?\d*', answer_str)
    if numbers:
        return numbers[-1]
    
    # 숫자가 없으면 소문자로 변환하고 공백 제거
    return answer_str.lower().strip()


def extract_cot_data(reasoning_paths):
    """Extract ordered lists of answers and token counts (src/evaluation.py와 동일).
    
    예시:
        reasoning_paths = [
            {"CoT_0": "...", "Final Answer_0": "30", "Tokens_0": 1564},
            {"CoT_1": "...", "Final Answer_1": "23", "Tokens_1": 1344},
            ...
        ]
        반환값: (["30", "23", "33", "33", ...], [1564, 1344, 1669, 1885, ...])
    """
    if not reasoning_paths:
        return [], []

    pattern_answer = re.compile(r'^Final Answer_(\d+)$')
    pattern_tokens = re.compile(r'^Tokens_(\d+)$')
    answers_map = {}
    tokens_map = {}

    for path in reasoning_paths:
        for key, value in path.items():
            match_answer = pattern_answer.match(key)
            if match_answer:
                idx = int(match_answer.group(1))
                answers_map[idx] = str(value) if value is not None else None
                continue

            match_tokens = pattern_tokens.match(key)
            if match_tokens:
                idx = int(match_tokens.group(1))
                tokens_map[idx] = int(value) if value is not None else 0

    ordered_indices = sorted(set(tokens_map.keys()) | set(answers_map.keys()))
    ordered_answers = []
    ordered_tokens = []

    for idx in ordered_indices:
        answer = answers_map.get(idx)
        token_count = tokens_map.get(idx, 0)
        if answer is not None:
            ordered_answers.append(answer)
        else:
            continue
        ordered_tokens.append(token_count)

    return ordered_answers, ordered_tokens


def prepare_samples(json_data):
    """JSON 데이터를 샘플 형식으로 변환 (src/evaluation.py와 동일).
    
    Example:
        json_data = [{
            "id": "2024-II-4",
            "question": "Let $x,y$ and $z$ be...",
            "answer": 33,
            "input_tokens": 2312,
            "Reasoning Paths": [
                {"Final Answer_0": "30", "Tokens_0": 1564, ...},
                ...
            ]
        }]
        반환값: [{
            'id': '2024-II-4',
            'question': 'Let $x,y$ and $z$ be...',
            'correct_answer': '33',
            'input_tokens': 2312,
            'cot_answers': ['30', '23', '33', ...],
            'cot_tokens': [1564, 1344, 1669, ...]
        }]
    """
    samples = []
    for item in json_data:
        answers, tokens = extract_cot_data(item.get('Reasoning Paths', []))
        samples.append({
            'id': item.get('id', ''),
            'question': item.get('question', ''),
            'correct_answer': str(item.get('answer', '')).strip(),
            'input_tokens': item.get('input_tokens', 0),
            'cot_answers': answers,
            'cot_tokens': tokens,
        })
    return samples


def calculate_cot(samples, pool_size: int = None, num_trials: int = 10, seed: int = 42):
    """Chain of Thought (CoT) 정확도 계산 (부트스트래핑 적용).
    
    각 질문의 Reasoning Path를 부트스트래핑으로 샘플링하여 정확도를 계산합니다.
    num_trials번 반복하여 평균값을 구합니다.
    
    Args:
        samples: 평가할 샘플 리스트
        pool_size: 사용할 reasoning chain 풀 크기 (None이면 모든 answers 사용)
        num_trials: 부트스트래핑 반복 횟수 (기본값: 10)
        seed: 랜덤 시드 (기본값: 42)
    """
    trial_accuracies = []
    trial_input_tokens = []
    trial_output_tokens = []
    
    for trial in range(num_trials):
        # trial마다 시드를 고정해 재현 가능하게 함
        random.seed(seed + trial)
        sample_accuracies = []
        total_input_tokens = 0
        total_output_tokens = 0

        for sample in samples:
            answers = sample['cot_answers']
            tokens = sample.get('cot_tokens', [])
            correct = sample['correct_answer']

            # STAGES_CONFIG(pool_size)에 따라 사용할 풀 크기 제한
            if pool_size is not None:
                answers = answers[:pool_size]
                tokens = tokens[:pool_size]

            if not answers:
                sample_accuracy = 0.0
                output_tokens = 0
            else:
                # 부트스트래핑: 모든 paths에서 랜덤하게 샘플링 (복원 추출)
                # 전체 paths를 사용하되, 각 trial마다 다른 순서로 평가
                # 각 샘플에서 랜덤하게 하나의 path를 선택하여 정확도 계산
                sampled_idx = random.randint(0, len(answers) - 1)
                sampled_answer = answers[sampled_idx]
                sampled_token = tokens[sampled_idx] if sampled_idx < len(tokens) else 0
                
                # 선택된 path가 정답과 일치하는지 확인
                correct_normalized = normalize_answer_eval(correct)
                sample_accuracy = 1.0 if normalize_answer_eval(sampled_answer) == correct_normalized else 0.0
                output_tokens = sampled_token

            sample_accuracies.append(sample_accuracy)
            total_input_tokens += sample.get('input_tokens', 0)
            total_output_tokens += output_tokens

        # 이번 trial의 정확도와 토큰 수
        trial_accuracy = sum(sample_accuracies) / len(samples) if samples else 0.0
        trial_input = total_input_tokens / len(samples) if samples else 0.0
        trial_output = total_output_tokens / len(samples) if samples else 0.0
        
        trial_accuracies.append(trial_accuracy)
        trial_input_tokens.append(trial_input)
        trial_output_tokens.append(trial_output)

    # 모든 trial의 평균 (CoT는 샘플당 1 chain)
    accuracy = sum(trial_accuracies) / len(trial_accuracies) if trial_accuracies else 0.0
    avg_input_tokens = sum(trial_input_tokens) / len(trial_input_tokens) if trial_input_tokens else 0.0
    avg_output_tokens = sum(trial_output_tokens) / len(trial_output_tokens) if trial_output_tokens else 0.0
    trial_chains = [1.0] * len(trial_accuracies)  # CoT: 1 chain per sample

    return accuracy, avg_input_tokens, avg_output_tokens, trial_accuracies, trial_chains, trial_input_tokens, trial_output_tokens


def _sample_subset_for_trial(answers: List[str],
                            tokens: List[int],
                            subset_n: int,
                            *,
                            seed: int) -> (List[str], List[int]):
    """
    answers/tokens에서 subset_n개를 (복원 없이) 랜덤 샘플링.
    - subset_n이 None이거나 answers 길이 이하이면 그대로/전부 사용.
    - 시드는 외부에서 trial/sample에 따라 결정해 전달한다.
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


def calculate_sc(samples, pool_size: int = None, sc_n: int = 40,
                 subset_n: int = None,
                 num_trials: int = 10, seed: int = 42):
    """Self-Consistency (SC) 정확도 계산.
    
    모든 메트릭(SC, AC, ESC, DSC, Ours)이 동일한 방식으로 시작:
    1. pool에서 subset_n(40)개를 먼저 샘플링
    2. 그 40개 안에서 각 방법론의 알고리즘을 적용
    
    Args:
        samples: 평가할 샘플 리스트
        pool_size: 전체 풀 크기 (None이면 모든 answers 사용)
        sc_n: 사용할 reasoning chain 개수 (기본값: 40)
        subset_n: pool_size 내에서 먼저 랜덤 샘플링할 체인 수 (None이면 pool_size 전부 사용)
        num_trials: 부트스트래핑 반복 횟수 (기본값: 10)
        seed: 랜덤 시드 (기본값: 42)
    
    Returns:
        Tuple: (accuracy, avg_input_tokens, avg_output_tokens)
    """
    random.seed(seed)
    
    trial_accuracies = []
    trial_input_tokens = []
    trial_output_tokens = []
    
    for trial in range(num_trials):
        correct_flags = []
        total_input_tokens = 0
        total_output_tokens = 0
        
        for sample in samples:
            answers = sample['cot_answers']
            correct = sample['correct_answer']
            
            if not answers:
                flag = 0
                sampled_input_tokens = 0
                sampled_output_tokens = 0
            else:
                # 풀 크기 설정
                if pool_size is None:
                    # 기존 동작: 모든 answers 사용
                    available_answers = answers
                    available_tokens = sample.get('cot_tokens', [])
                else:
                    # 풀 크기만큼만 사용
                    available_answers = answers[:pool_size]
                    available_tokens = sample.get('cot_tokens', [])[:pool_size]
                
                # AC/ESC와 동일한 방식: pool에서 subset_n(40)개를 먼저 샘플링
                subset_seed = seed + trial
                subset_answers, subset_tokens = _sample_subset_for_trial(
                    available_answers, available_tokens, subset_n,
                    seed=subset_seed
                )
                
                # subset(40개) 내에서 앞에서부터 sc_n개를 순서대로 사용 (SC 알고리즘: sc_n개 모두 사용)
                # Ours와 동일하게 subset만 랜덤 샘플링하고, subset 내에서는 순서대로 사용
                if len(subset_answers) <= sc_n:
                    # subset 크기가 sc_n 이하면 모두 사용
                    sampled_answers = subset_answers
                    sampled_tokens = subset_tokens
                else:
                    # subset에서 앞에서부터 sc_n개를 순서대로 사용 (랜덤 샘플링 없음)
                    sampled_answers = subset_answers[:sc_n]
                    sampled_tokens = subset_tokens[:sc_n]
                
                # Majority voting
                normalized_answers = [normalize_answer_eval(ans) for ans in sampled_answers]
                most_common_normalized = Counter(normalized_answers).most_common(1)[0][0]
                flag = 1 if most_common_normalized == normalize_answer_eval(correct) else 0
                
                # 토큰 계산
                sampled_input_tokens = sample.get('input_tokens', 0)
                sampled_output_tokens = sum(sampled_tokens) if sampled_tokens else 0
            
            correct_flags.append(flag)
            total_input_tokens += sampled_input_tokens
            total_output_tokens += sampled_output_tokens
        
        # 이번 trial의 메트릭
        trial_accuracy = sum(correct_flags) / len(samples) if samples else 0.0
        trial_input = total_input_tokens / len(samples) if samples else 0.0
        trial_output = total_output_tokens / len(samples) if samples else 0.0
        
        trial_accuracies.append(trial_accuracy)
        trial_input_tokens.append(trial_input)
        trial_output_tokens.append(trial_output)
    
    # 모든 trial의 평균 (SC는 샘플당 sc_n chains 사용)
    accuracy = sum(trial_accuracies) / len(trial_accuracies) if trial_accuracies else 0.0
    avg_input_tokens = sum(trial_input_tokens) / len(trial_input_tokens) if trial_input_tokens else 0.0
    avg_output_tokens = sum(trial_output_tokens) / len(trial_output_tokens) if trial_output_tokens else 0.0
    trial_avg_chains = [float(sc_n)] * len(trial_accuracies)  # SC: sc_n chains per sample

    return accuracy, avg_input_tokens, avg_output_tokens, trial_accuracies, trial_avg_chains, trial_input_tokens, trial_output_tokens


def evaluate_esc_for_sample_ver_esc(answers, tokens, correct, window_size, shuffle: bool = False, seed: int = None):
    """Early Stopping (ESC) 평가 - ESC 깃허브 버전 (부트스트래핑 지원).
    
    배치 단위로 처리하고 조기 종료 후에도 전체 답변으로 예측.
    
    Args:
        answers: 답변 리스트
        tokens: 토큰 리스트
        correct: 정답
        window_size: 윈도우 크기
        shuffle: True면 answers와 tokens를 랜덤하게 섞음 (부트스트래핑용)
        seed: 랜덤 시드 (shuffle=True일 때 사용)
    """
    if not answers:
        return 0, 0, 0, 0
    
    # 부트스트래핑: 랜덤하게 섞기
    if shuffle and seed is not None:
        random.seed(seed)
        combined = list(zip(answers, tokens))
        random.shuffle(combined)
        answers, tokens = zip(*combined)
        answers = list(answers)
        tokens = list(tokens)
    
    sc_num = len(answers)
    predict_list = []
    max_iterations = sc_num // window_size if window_size > 0 else 0
    index = -1
    
    # 배치 단위로 처리
    for index in range(max_iterations):
        cur_batch = answers[index * window_size: (index + 1) * window_size]
        if not cur_batch:
            break
        
        # 현재 배치의 답변들을 predict_list에 추가
        for answer in cur_batch:
            predict_list.append(answer)
        
        # 현재 배치 내 모든 답변이 동일한지 확인 (정규화된 답변으로 비교)
        normalized_batch = [normalize_answer_eval(ans) for ans in cur_batch]
        pre_dict = {}
        for norm_ans in normalized_batch:
            pre_dict[norm_ans] = pre_dict.get(norm_ans, 0) + (1 / len(cur_batch))
        
        # 조기 종료 조건: 현재 배치 내 모든 답변이 동일
        if len(pre_dict) == 1:
            break
    
    # 최종 예측: 전체 predict_list에 대해 majority voting (정규화된 답변으로)
    if not predict_list:
        predict_list = answers
    
    # 정규화된 답변으로 majority voting
    normalized_predict_list = [normalize_answer_eval(ans) for ans in predict_list]
    final_pre_dict = {}
    for norm_ans in normalized_predict_list:
        final_pre_dict[norm_ans] = final_pre_dict.get(norm_ans, 0) + (1 / len(normalized_predict_list))
    
    # 가장 많이 나온 답변 선택
    majority_normalized = max(final_pre_dict, key=final_pre_dict.get)
    flag = 1 if majority_normalized == normalize_answer_eval(correct) else 0
    
    # 사용된 체인 수: (index + 1) * window_size
    if index >= 0:
        steps = (index + 1) * window_size
        num_windows = index + 1
    else:
        steps = len(answers)
        num_windows = 1 if len(answers) > 0 else 0
    steps = min(steps, len(answers))
    
    # 사용된 토큰 계산
    tokens_used = sum(tokens[:steps]) if len(tokens) >= steps else sum(tokens)
    
    return flag, steps, tokens_used, num_windows


def calculate_esc_ver_esc(samples, window_size, pool_size: int = None,
                         subset_n: int = None,
                         num_trials: int = 10, seed: int = 42):
    """Early Stopping (ESC) 정확도 계산 - ESC 깃허브 버전 (부트스트래핑 적용).
    
    Args:
        samples: 평가할 샘플 리스트
        window_size: 윈도우 크기
        pool_size: 전체 풀 크기 (None이면 모든 answers 사용)
        subset_n: pool_size 내에서 먼저 랜덤 샘플링할 체인 수 (None이면 pool_size 전부 사용)
        num_trials: 부트스트래핑 반복 횟수 (기본값: 10)
        seed: 랜덤 시드 (기본값: 42)
    """
    if window_size <= 0:
        return None

    random.seed(seed)
    
    trial_accuracies = []
    trial_avg_chains = []
    trial_avg_input_tokens = []
    trial_avg_output_tokens = []
    
    for trial in range(num_trials):
        correct_flags = []
        total_steps = 0
        total_input_tokens = 0
        total_output_tokens = 0

        for sample in samples:
            answers = sample['cot_answers']
            tokens = sample.get('cot_tokens', [])
            
            # 풀 크기만큼만 사용
            if pool_size is not None:
                answers = answers[:pool_size]
                tokens = tokens[:pool_size]

            # pool_size 내에서 subset_n개를 먼저 랜덤 샘플링
            subset_seed = seed + trial
            answers, tokens = _sample_subset_for_trial(
                answers, tokens, subset_n,
                seed=subset_seed
            )
            
            flag, steps, output_tokens_used, num_windows = evaluate_esc_for_sample_ver_esc(
                answers,
                tokens,
                sample['correct_answer'],
                window_size,
                shuffle=False,  # Ours와 동일하게 shuffle 없이 순서대로 사용
                seed=seed + trial  # 각 trial마다 다른 시드 사용 (shuffle=False이므로 사용 안 함)
            )

            correct_flags.append(flag)
            total_steps += steps
            # ESC는 window가 sliding/호출된 횟수만큼 input_tokens 사용
            total_input_tokens += sample.get('input_tokens', 0) * num_windows
            total_output_tokens += output_tokens_used

        # 이번 trial의 메트릭
        trial_accuracy = sum(correct_flags) / len(samples) if samples else 0.0
        trial_avg_chain = total_steps / len(samples) if samples else 0.0
        trial_avg_input = total_input_tokens / len(samples) if samples else 0.0
        trial_avg_output = total_output_tokens / len(samples) if samples else 0.0
        
        trial_accuracies.append(trial_accuracy)
        trial_avg_chains.append(trial_avg_chain)
        trial_avg_input_tokens.append(trial_avg_input)
        trial_avg_output_tokens.append(trial_avg_output)

    # 모든 trial의 평균
    accuracy = sum(trial_accuracies) / len(trial_accuracies) if trial_accuracies else 0.0
    avg_chains = sum(trial_avg_chains) / len(trial_avg_chains) if trial_avg_chains else 0.0
    avg_input_tokens = sum(trial_avg_input_tokens) / len(trial_avg_input_tokens) if trial_avg_input_tokens else 0.0
    avg_output_tokens = sum(trial_avg_output_tokens) / len(trial_avg_output_tokens) if trial_avg_output_tokens else 0.0

    return accuracy, avg_chains, avg_input_tokens, avg_output_tokens, trial_accuracies, trial_avg_chains, trial_avg_input_tokens, trial_avg_output_tokens


def evaluate_ac_for_sample(answers, tokens, correct, ac, shuffle: bool = False, seed: int = None):
    """Adaptive Self-Consistency (AC) 평가 (부트스트래핑 지원).
    
    Beta 기준으로 적응적 중단.
    
    Args:
        answers: 답변 리스트
        tokens: 토큰 리스트
        correct: 정답
        ac: AC 객체
        shuffle: True면 answers와 tokens를 랜덤하게 섞음 (부트스트래핑용)
        seed: 랜덤 시드 (shuffle=True일 때 사용)
    """
    if not answers:
        return 0, 0, 0

    # 부트스트래핑: 랜덤하게 섞기
    if shuffle and seed is not None:
        random.seed(seed)
        combined = list(zip(answers, tokens))
        random.shuffle(combined)
        answers, tokens = zip(*combined)
        answers = list(answers)
        tokens = list(tokens)

    for i, answer in enumerate(answers):
        if ac.should_stop(answers[:i+1]):
            break
    
    truncated_answers = answers[:i+1]
    truncated_tokens = tokens[:i+1]
    
    if not truncated_answers:
        return 0, 0, 0
    # 정규화된 답변으로 Counter를 만들어 majority voting
    normalized_truncated = [normalize_answer_eval(ans) for ans in truncated_answers]
    majority_normalized = Counter(normalized_truncated).most_common(1)[0][0]
    flag = 1 if majority_normalized == normalize_answer_eval(correct) else 0
    return flag, len(truncated_answers), sum(truncated_tokens)


def calculate_ac(samples, beta, max_gens: int = None,
                 pool_size: int = None,
                 subset_n: int = None,
                 num_trials: int = 10, seed: int = 42):
    """Adaptive Self-Consistency (AC) 정확도 계산 (부트스트래핑 적용).
    
    adaptive_consistency 모듈이 없어도 직접 구현한 버전으로 계산 가능.
    
    Args:
        samples: 평가할 샘플 리스트
        beta: AC beta 값
        max_gens: 최대 생성 개수 (None이면 풀 크기 자동 감지)
        pool_size: 전체 풀 크기 (None이면 모든 answers 사용)
        subset_n: pool_size 내에서 먼저 랜덤 샘플링할 체인 수 (None이면 pool_size 전부 사용)
        num_trials: 부트스트래핑 반복 횟수 (기본값: 10)
        seed: 랜덤 시드 (기본값: 42)
    """
    if beta <= 0:
        return None

    random.seed(seed)
    
    # max_gens 자동 감지: subset_n > pool_size > answers_len 순으로 우선
    if max_gens is None:
        if subset_n is not None and subset_n > 0:
            max_gens = subset_n
        elif pool_size is not None and pool_size > 0:
            max_gens = pool_size
        elif samples and samples[0].get('cot_answers'):
            max_gens = len(samples[0]['cot_answers'])
        else:
            max_gens = 40  # 기본값
    
    # 직접 구현한 AC 클래스 사용 (모듈이 없어도 동작)
    ac = AC(stop_criteria=BetaStoppingCriteria(beta), max_gens=max_gens)
    
    trial_accuracies = []
    trial_avg_chains = []
    trial_avg_input_tokens = []
    trial_avg_output_tokens = []
    
    for trial in range(num_trials):
        correct_flags = []
        total_steps = 0
        total_input_tokens = 0
        total_output_tokens = 0

        for sample in samples:
            answers = sample['cot_answers']
            tokens = sample.get('cot_tokens', [])
            if pool_size is not None:
                answers = answers[:pool_size]
                tokens = tokens[:pool_size]

            subset_seed = seed + trial
            answers, tokens = _sample_subset_for_trial(
                answers, tokens, subset_n,
                seed=subset_seed
            )

            # subset 기준으로 AC 객체 max_gens를 넘지 않게 보정
            # (AC는 내부에서 max_gens 도달 시 중단 조건을 걸기 때문에 안전하게 맞춰둔다)
            local_max_gens = min(ac.max_gens, len(answers)) if hasattr(ac, "max_gens") else len(answers)

            flag, steps, output_tokens_used = evaluate_ac_for_sample(
                answers[:local_max_gens],
                tokens[:local_max_gens],
                sample['correct_answer'],
                ac,
                shuffle=False,  # Ours와 동일하게 shuffle 없이 순서대로 사용
                seed=seed + trial  # 각 trial마다 다른 시드 사용 (shuffle=False이므로 사용 안 함)
            )

            correct_flags.append(flag)
            total_steps += steps
            # AC는 steps만큼 input_tokens 사용
            total_input_tokens += sample.get('input_tokens', 0) * steps
            total_output_tokens += output_tokens_used

        # 이번 trial의 메트릭
        trial_accuracy = sum(correct_flags) / len(samples) if samples else 0.0
        trial_avg_chain = total_steps / len(samples) if samples else 0.0
        trial_avg_input = total_input_tokens / len(samples) if samples else 0.0
        trial_avg_output = total_output_tokens / len(samples) if samples else 0.0
        
        trial_accuracies.append(trial_accuracy)
        trial_avg_chains.append(trial_avg_chain)
        trial_avg_input_tokens.append(trial_avg_input)
        trial_avg_output_tokens.append(trial_avg_output)

    # 모든 trial의 평균
    accuracy = sum(trial_accuracies) / len(trial_accuracies) if trial_accuracies else 0.0
    avg_chains = sum(trial_avg_chains) / len(trial_avg_chains) if trial_avg_chains else 0.0
    avg_input_tokens = sum(trial_avg_input_tokens) / len(trial_avg_input_tokens) if trial_avg_input_tokens else 0.0
    avg_output_tokens = sum(trial_avg_output_tokens) / len(trial_avg_output_tokens) if trial_avg_output_tokens else 0.0

    return accuracy, avg_chains, avg_input_tokens, avg_output_tokens, trial_accuracies, trial_avg_chains, trial_avg_input_tokens, trial_avg_output_tokens


def calculate_accuracy(predictions: List[str], ground_truths: List[str]) -> float:
    """
    정확도(Accuracy) 계산
    
    Args:
        predictions: 예측 답변 리스트
        ground_truths: 정답 리스트
        
    Returns:
        float: 정확도 (0.0 ~ 1.0)
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("predictions와 ground_truths의 길이가 일치하지 않습니다.")
    
    correct = 0
    for pred, gt in zip(predictions, ground_truths):
        if normalize_answer_eval(pred) == normalize_answer_eval(gt):
            correct += 1
    
    return correct / len(predictions) if predictions else 0.0


def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    전체 결과에 대한 메트릭 계산
    
    Args:
        results: 평가 결과 리스트
                각 결과는 {
                    'prediction': str,  # final_answer
                    'ground_truth': str,
                    'metadata': Dict,
                    ...
                } 형식
        
    Returns:
        Dict[str, float]: 계산된 메트릭 딕셔너리
            - final_accuracy: 최종 답변 정확도
            - total_samples: 전체 샘플 수
    """
    ground_truths = [str(r.get('ground_truth', '')) for r in results]
    
    # Final answer 정확도
    final_predictions = [r.get('prediction', '') for r in results]
    final_accuracy = calculate_accuracy(final_predictions, ground_truths)
    
    metrics = {
        'final_accuracy': final_accuracy,
        'total_samples': len(results),
        'final_correct_samples': sum(1 for p, gt in zip(final_predictions, ground_truths) 
                                   if normalize_answer_eval(p) == normalize_answer_eval(gt)),
    }
    
    return metrics


def compute_stage1_metrics_from_samples(
    samples: List[Dict[str, Any]],
    *,
    stage_name: str = "Stage1",
    pool_size: int,
    subset_n_capped: Optional[int],
    sc_n_capped: Optional[int],
    esc_window_size: int = 5,
    ac_beta: float = 0.95,
    num_trials: int = 10,
    seed: int = 42,
) -> Dict[str, Any]:
    """prepare_samples 결과 리스트에 대해 CoT/SC/ESC/AC 메트릭 계산 (evaluate_stage_from_json 핵심)."""
    metrics: Dict[str, Any] = {
        "stage": stage_name,
        "total_samples": len(samples),
        "pool_size": pool_size,
        "sc_n": sc_n_capped,
        "subset_n": subset_n_capped,
    }
    cot_acc, cot_avg_input, cot_avg_output, cot_trials, cot_trial_chains, cot_trial_input_tokens, cot_trial_output_tokens = calculate_cot(
        samples,
        pool_size=pool_size,
        num_trials=num_trials,
        seed=seed,
    )
    metrics["cot_accuracy"] = cot_acc
    metrics["cot_avg_input_tokens"] = cot_avg_input
    metrics["cot_avg_output_tokens"] = cot_avg_output
    metrics["cot_trial_accuracies"] = cot_trials
    metrics["cot_trial_chains"] = cot_trial_chains
    metrics["cot_trial_input_tokens"] = cot_trial_input_tokens
    metrics["cot_trial_output_tokens"] = cot_trial_output_tokens

    sc_acc, sc_avg_input, sc_avg_output, sc_trials, sc_trial_chains, sc_trial_input_tokens, sc_trial_output_tokens = calculate_sc(
        samples,
        pool_size=pool_size,
        sc_n=sc_n_capped,
        subset_n=subset_n_capped,
        num_trials=num_trials,
        seed=seed,
    )
    metrics["sc_accuracy"] = sc_acc
    metrics["sc_avg_input_tokens"] = sc_avg_input
    metrics["sc_avg_output_tokens"] = sc_avg_output
    metrics["sc_trial_accuracies"] = sc_trials
    metrics["sc_trial_chains"] = sc_trial_chains
    metrics["sc_trial_input_tokens"] = sc_trial_input_tokens
    metrics["sc_trial_output_tokens"] = sc_trial_output_tokens

    if esc_window_size > 0:
        esc_esc_metrics = calculate_esc_ver_esc(
            samples,
            esc_window_size,
            pool_size=pool_size,
            subset_n=subset_n_capped,
            num_trials=num_trials,
            seed=seed,
        )
        if esc_esc_metrics:
            esc_esc_acc, esc_esc_avg_chains, esc_esc_avg_input, esc_esc_avg_output, esc_trials, esc_trial_chains, esc_trial_input_tokens, esc_trial_output_tokens = esc_esc_metrics
            metrics["esc_esc_accuracy"] = esc_esc_acc
            metrics["esc_esc_avg_chains"] = esc_esc_avg_chains
            metrics["esc_esc_avg_input_tokens"] = esc_esc_avg_input
            metrics["esc_esc_avg_output_tokens"] = esc_esc_avg_output
            metrics["esc_esc_trial_accuracies"] = esc_trials
            metrics["esc_esc_trial_chains"] = esc_trial_chains
            metrics["esc_esc_trial_input_tokens"] = esc_trial_input_tokens
            metrics["esc_esc_trial_output_tokens"] = esc_trial_output_tokens

    if ac_beta > 0:
        ac_metrics = calculate_ac(
            samples,
            ac_beta,
            max_gens=subset_n_capped if subset_n_capped is not None else pool_size,
            pool_size=pool_size,
            subset_n=subset_n_capped,
            num_trials=num_trials,
            seed=seed,
        )
        if ac_metrics:
            ac_acc, ac_avg_chains, ac_avg_input, ac_avg_output, ac_trials, ac_trial_chains, ac_trial_input_tokens, ac_trial_output_tokens = ac_metrics
            metrics["ac_accuracy"] = ac_acc
            metrics["ac_avg_chains"] = ac_avg_chains
            metrics["ac_avg_input_tokens"] = ac_avg_input
            metrics["ac_avg_output_tokens"] = ac_avg_output
            metrics["ac_trial_accuracies"] = ac_trials
            metrics["ac_trial_chains"] = ac_trial_chains
            metrics["ac_trial_input_tokens"] = ac_trial_input_tokens
            metrics["ac_trial_output_tokens"] = ac_trial_output_tokens

    return metrics


def slim_stage1_metrics_for_aggregate(full: Dict[str, Any]) -> Dict[str, Any]:
    """집계 JSON용: 스칼라 요약만 유지 (trial 리스트 제외). accuracy 는 퍼센트(0~100)."""

    def _acc_block(acc_key: str, in_key: str, out_key: str, chain_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if acc_key not in full:
            return None
        block: Dict[str, Any] = {
            "accuracy": round(float(full[acc_key]) * 100.0, 4),
            "avg_input_tokens": round(float(full.get(in_key, 0)), 4),
            "avg_output_tokens": round(float(full.get(out_key, 0)), 4),
        }
        block["avg_total_tokens"] = round(block["avg_input_tokens"] + block["avg_output_tokens"], 4)
        if chain_key and chain_key in full:
            v = full[chain_key]
            if isinstance(v, list) and v:
                block["avg_chains"] = round(float(sum(v) / len(v)), 4)
            else:
                block["avg_chains"] = round(float(v), 4)
        return block

    out: Dict[str, Any] = {"sample_count": int(full.get("total_samples", 0))}
    b = _acc_block("cot_accuracy", "cot_avg_input_tokens", "cot_avg_output_tokens")
    if b:
        out["CoT"] = b
    b = _acc_block("sc_accuracy", "sc_avg_input_tokens", "sc_avg_output_tokens", "sc_trial_chains")
    if b:
        out["SC"] = b
    b = _acc_block("ac_accuracy", "ac_avg_input_tokens", "ac_avg_output_tokens", "ac_avg_chains")
    if b:
        out["AC"] = b
    b = _acc_block("esc_esc_accuracy", "esc_esc_avg_input_tokens", "esc_esc_avg_output_tokens", "esc_esc_avg_chains")
    if b:
        out["ESC"] = b
    return out


def evaluate_mmlu_pro_by_category_json_data(
    json_data: List[Dict[str, Any]],
    *,
    stage_name: str = "Stage1",
    esc_window_size: int = 5,
    ac_beta: float = 0.95,
    num_trials: int = 10,
    seed: int = 42,
    stages_config_slug: str = "stages-40",
    sc_n: int = 40,
    subset_n: int = 40,
) -> Dict[str, Dict[str, Any]]:
    """
    MMLU-Pro Stage1 JSON 리스트를 metadata.category 별로 나누어
    slim_stage1_metrics_for_aggregate 형태의 dict를 카테고리마다 반환.
    """
    from collections import defaultdict

    pool_size = extract_pool_size_from_stages_config(stages_config_slug)
    subset_n_capped = min(subset_n, pool_size) if subset_n is not None else None
    sc_n_capped = min(sc_n, pool_size) if sc_n is not None else None

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in json_data:
        meta = item.get("metadata") or {}
        raw = meta.get("category")
        cat = str(raw).strip() if raw is not None and str(raw).strip() else "unknown"
        groups[cat].append(item)

    out: Dict[str, Dict[str, Any]] = {}
    for cat in sorted(groups.keys()):
        samples = prepare_samples(groups[cat])
        if not samples:
            continue
        full = compute_stage1_metrics_from_samples(
            samples,
            stage_name=stage_name,
            pool_size=pool_size,
            subset_n_capped=subset_n_capped,
            sc_n_capped=sc_n_capped,
            esc_window_size=esc_window_size,
            ac_beta=ac_beta,
            num_trials=num_trials,
            seed=seed,
        )
        out[cat] = slim_stage1_metrics_for_aggregate(full)
    return out


def evaluate_stage_from_json(json_file_path: str, stage_name: str = "Stage1",
                            esc_window_size: int = 5, ac_beta: float = 0.95,
                            num_trials: int = 10, seed: int = 42,
                            stages_config_slug: str = "stages-40",
                            sc_n: int = 40,
                            subset_n: int = 40) -> Dict[str, Any]:
    """
    저장된 JSON 파일에서 Stage별 평가 수행 (src/evaluation.py 방식)
    
    Args:
        json_file_path: 평가할 JSON 파일 경로
        stage_name: Stage 이름 (예: "Stage1", "Stage2", ...)
        esc_window_size: ESC 평가를 위한 window size (기본값: 5)
        ac_beta: AC 평가를 위한 beta 값 (기본값: 0.95)
        num_trials: 부트스트래핑 반복 횟수 (기본값: 10, CoT/ESC/AC에 적용)
        seed: 랜덤 시드 (기본값: 42)
        stages_config_slug: stages 설정 (예: "stages-40", "stages-100")
        sc_n: SC에서 사용할 reasoning chain 개수 (기본값: 40)
        subset_n: ESC/AC/DSC류 평가에서 pool_size 내에서 먼저 랜덤 샘플링할 체인 수 (기본값: 40)
        
    Returns:
        Dict[str, Any]: 평가 결과 메트릭
            - cot_accuracy: CoT 정확도
            - sc_accuracy: SC 정확도
            - esc_esc_accuracy: ESC_esc 정확도
            - ac_accuracy: AC 정확도 (adaptive_consistency 모듈이 있을 때만)
            - 각 메트릭의 토큰 사용량 정보
    """
    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"JSON file not found: {json_file_path}")
    
    # 풀 크기 추출
    pool_size = extract_pool_size_from_stages_config(stages_config_slug)
    # subset_n과 sc_n은 풀 크기를 넘지 않게 캡
    subset_n_capped = min(subset_n, pool_size) if subset_n is not None else None
    sc_n_capped = min(sc_n, pool_size) if sc_n is not None else None
    
    # JSON 파일 로드
    with open(json_file_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    # prepare_samples로 변환 (직접 구현한 함수 사용)
    samples = prepare_samples(json_data)

    return compute_stage1_metrics_from_samples(
        samples,
        stage_name=stage_name,
        pool_size=pool_size,
        subset_n_capped=subset_n_capped,
        sc_n_capped=sc_n_capped,
        esc_window_size=esc_window_size,
        ac_beta=ac_beta,
        num_trials=num_trials,
        seed=seed,
    )

