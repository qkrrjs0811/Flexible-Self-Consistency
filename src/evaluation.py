import argparse
import json
import os
import random
import re
from collections import Counter

from adaptive_consistency import AC, BetaStoppingCriteria

EVAL_DIR = '../eval'  # 기본값 (명령줄 인자로 덮어쓸 수 있음)


def normalize_answer(answer):
    """답변을 정규화하여 비교 가능하게 만듦.
    
    숫자가 있으면 숫자만 추출하고, 없으면 공백 제거 후 소문자 변환.
    rsc_project의 normalize_answer_eval과 동일한 로직을 사용하여 일관성 유지.
    
    Args:
        answer: 원본 답변 (str 또는 다른 타입)
        
    Returns:
        str: 정규화된 답변
    """
    if answer is None:
        return ""
    answer_str = str(answer).strip()
    
    # 숫자만 추출
    numbers = re.findall(r'-?\d+\.?\d*', answer_str)
    if numbers:
        # 마지막 숫자를 답변으로 간주
        return numbers[-1]
    
    # 숫자가 없으면 소문자로 변환하고 공백 제거
    return answer_str.lower().strip()


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_cot_data(reasoning_paths):
    """Extract ordered lists of answers and token counts.
    
    예시:
        reasoning_paths = [
            {"CoT_0": "...", "Final Answer_0": "30", "Tokens_0": 1564},
            {"CoT_1": "...", "Final Answer_1": "23", "Tokens_1": 1344},
            {"CoT_2": "...", "Final Answer_2": "33", "Tokens_2": 1669},
            {"CoT_3": "...", "Final Answer_3": "33", "Tokens_3": 1885},
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
    """
    Example:
        json_data = [{
            "id": "2024-II-4",
            "question": "Let $x,y$ and $z$ be...",
            "answer": 33,
            "input_tokens": 2312,
            "Reasoning Paths": [
                {"Final Answer_0": "30", "Tokens_0": 1564, ...},
                {"Final Answer_1": "23", "Tokens_1": 1344, ...},
                {"Final Answer_2": "33", "Tokens_2": 1669, ...},
                ...
            ]
        }]
        반환값: [{
            'id': '2024-II-4',
            'question': 'Let $x,y$ and $z$ be...',
            'correct_answer': '33',
            'input_tokens': 2312,
            'cot_answers': ['30', '23', '33', '33', ...],
            'cot_tokens': [1564, 1344, 1669, 1885, ...]
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


def calculate_cot(samples):
    """Chain of Thought (CoT) 정확도 계산 - 각 질문의 모든 Reasoning Path(40개)의 정확도 평균.
    
    Example:
        sample = {
            'cot_answers': ['30', '23', '33', '33', '23', '19', '157', '33', ...],  # 40개 path
            'correct_answer': '33',
            'input_tokens': 2312,
            'cot_tokens': [1564, 1344, 1669, 1885, 1394, 1204, 1602, 1867, ...]
        }
        각 질문에 대해:
        - 40개 path 중 정답과 일치하는 path 수 계산
        - 질문의 accuracy = (정답 맞춘 path 수) / 40
        예: 40개 중 25개가 정답이면 accuracy = 25/40 = 0.625
        
        전체 CoT accuracy = 모든 질문의 accuracy 평균
    """
    sample_accuracies = []
    total_input_tokens = 0
    total_output_tokens = 0

    for sample in samples:
        answers = sample['cot_answers']
        correct = sample['correct_answer']

        if not answers:
            sample_accuracy = 0.0
            output_tokens = 0
        else:
            # 각 질문의 40개 reasoning path 중 정답과 일치하는 path 수 계산
            correct_normalized = normalize_answer(correct)
            correct_count = sum(1 for answer in answers if normalize_answer(answer) == correct_normalized)
            sample_accuracy = correct_count / len(answers)
            # 첫 번째 path의 토큰만 사용 (CoT는 단일 path를 사용하므로)
            output_tokens = sample.get('cot_tokens', [0])[0] if sample.get('cot_tokens') else 0

        sample['cot_correct'] = sample_accuracy
        sample_accuracies.append(sample_accuracy)
        total_input_tokens += sample.get('input_tokens', 0)
        total_output_tokens += output_tokens

    # 전체 CoT accuracy = 모든 질문의 accuracy 평균
    accuracy = sum(sample_accuracies) / len(samples) if samples else 0.0
    avg_input_tokens = total_input_tokens / len(samples) if samples else 0.0
    avg_output_tokens = total_output_tokens / len(samples) if samples else 0.0
    return accuracy, avg_input_tokens, avg_output_tokens


def calculate_sc(samples):
    """Self-Consistency (SC) 정확도 계산 - 모든 Reasoning Chain 사용.
    
    Example:
        sample = {
            'cot_answers': ['30', '23', '33', '33', '23', '19', '157', '33', ...],
            'correct_answer': '33',
            'input_tokens': 2312,
            'cot_tokens': [1564, 1344, 1669, 1885, 1394, 1204, 1602, 1867, ...]
        }
        Counter(['30', '23', '33', '33', '23', '19', '157', '33', ...]).most_common(1)
        -> [('33', 3)]  # '33'이 가장 많이 나옴 (3번)
        most_common = '33'
        flag = 1  # '33' == '33' (정답과 일치)
        
        만약 most_common이 '30'이었다면:
        flag = 0  # '30' != '33' (정답과 불일치)
    """
    correct_flags = []
    total_input_tokens = 0
    total_output_tokens = 0

    for sample in samples:
        answers = sample['cot_answers']
        correct = sample['correct_answer']
        num_paths = len(answers)

        if not answers:
            flag = 0
        else:
            # 정규화된 답변으로 Counter를 만들어 majority voting
            normalized_answers = [normalize_answer(ans) for ans in answers]
            most_common_normalized = Counter(normalized_answers).most_common(1)[0][0]
            flag = 1 if most_common_normalized == normalize_answer(correct) else 0

        sample['sc_correct'] = flag
        correct_flags.append(flag)
        # SC는 모든 path를 사용하므로 input_tokens
        total_input_tokens += sample.get('input_tokens', 0)
        total_output_tokens += sum(sample.get('cot_tokens', []))

    accuracy = sum(correct_flags) / len(samples) if samples else 0.0
    avg_input_tokens = total_input_tokens / len(samples) if samples else 0.0
    avg_output_tokens = total_output_tokens / len(samples) if samples else 0.0
    return accuracy, avg_input_tokens, avg_output_tokens


def calculate_sc_sampled(samples, num_chains, num_trials=10):
    """Self-Consistency (SC) 정확도 계산 - 일부 Reasoning Chain만 사용 (Monte Carlo 샘플링).
    
    모든 Reasoning Chain 중에서 랜덤하게 num_chains개를 선택하여 SC를 계산하고,
    이를 num_trials번 반복하여 평균을 구합니다.
    
    Args:
        samples: 평가할 샘플 리스트
        num_chains: 사용할 Reasoning Chain 개수 (예: 5, 10)
        num_trials: 샘플링 반복 횟수 (기본값: 10)
    
    Example:
        sample = {
            'cot_answers': ['30', '23', '33', '33', '23', '19', '157', '33', ...],  # 40개
            'correct_answer': '33',
            'input_tokens': 2312,
            'cot_tokens': [1564, 1344, 1669, 1885, 1394, 1204, 1602, 1867, ...]
        }
        num_chains = 5, num_trials = 10
        
        Trial 1: 랜덤하게 5개 선택 -> ['30', '33', '33', '23', '33'] -> majority='33' -> flag=1
        Trial 2: 랜덤하게 5개 선택 -> ['23', '19', '30', '23', '33'] -> majority='23' -> flag=0
        ...
        Trial 10: 랜덤하게 5개 선택 -> ['33', '33', '33', '23', '33'] -> majority='33' -> flag=1
        
        최종 accuracy = (Trial 1~10의 flag 평균)
        최종 토큰 수 = (num_chains * input_tokens + 선택된 chains의 output_tokens 합)의 평균
    """
    random.seed(42)  # 재현성을 위해 random seed 고정
    
    if num_chains <= 0:
        return None
    
    # 각 trial의 결과를 저장
    trial_accuracies = []
    trial_input_tokens = []
    trial_output_tokens = []
    
    for trial in range(num_trials):
        correct_flags = []
        total_input_tokens = 0
        total_output_tokens = 0
        
        for sample in samples:
            answers = sample['cot_answers']
            tokens = sample.get('cot_tokens', [])
            correct = sample['correct_answer']
            
            if not answers:
                flag = 0
                sampled_input_tokens = 0
                sampled_output_tokens = 0
            else:
                # 랜덤하게 num_chains개 선택
                available_indices = list(range(len(answers)))
                if len(available_indices) <= num_chains:
                    # 전체 chain 수가 num_chains 이하인 경우 모두 사용
                    sampled_indices = available_indices
                else:
                    sampled_indices = random.sample(available_indices, num_chains)
                
                # 선택된 chain들의 답변과 토큰 추출
                sampled_answers = [answers[i] for i in sampled_indices]
                sampled_tokens = [tokens[i] if i < len(tokens) else 0 for i in sampled_indices]
                
                # 정규화된 답변으로 Counter를 만들어 majority voting
                normalized_answers = [normalize_answer(ans) for ans in sampled_answers]
                most_common_normalized = Counter(normalized_answers).most_common(1)[0][0]
                flag = 1 if most_common_normalized == normalize_answer(correct) else 0
                
                # 토큰 계산: input_tokens는 num_chains만큼 사용, output_tokens는 선택된 chains의 합
                sampled_input_tokens = sample.get('input_tokens', 0) * len(sampled_indices)
                sampled_output_tokens = sum(sampled_tokens)
            
            correct_flags.append(flag)
            total_input_tokens += sampled_input_tokens
            total_output_tokens += sampled_output_tokens
        
        # 이번 trial의 정확도와 토큰 수
        trial_accuracy = sum(correct_flags) / len(samples) if samples else 0.0
        trial_avg_input = total_input_tokens / len(samples) if samples else 0.0
        trial_avg_output = total_output_tokens / len(samples) if samples else 0.0
        
        trial_accuracies.append(trial_accuracy)
        trial_input_tokens.append(trial_avg_input)
        trial_output_tokens.append(trial_avg_output)
    
    # 모든 trial의 평균 계산
    accuracy = sum(trial_accuracies) / num_trials if num_trials > 0 else 0.0
    avg_input_tokens = sum(trial_input_tokens) / num_trials if num_trials > 0 else 0.0
    avg_output_tokens = sum(trial_output_tokens) / num_trials if num_trials > 0 else 0.0
    
    return accuracy, avg_input_tokens, avg_output_tokens


def evaluate_esc_for_sample_ver_rasc(answers, tokens, correct, window_size):
    """Early Stopping (ESC) 평가 - window_size만큼 연속된 동일 답변을 찾으면 중단.
    
    예시 (window_size=5):
        answers = ['30', '23', '33', '33', '23', '19', '157', '33', ...]
        tokens = [1564, 1344, 1669, 1885, 1394, 1204, 1602, 1867, ...]
        correct = '33'
        
        i=0: window = ['30', '23', '33', '33', '23'] -> 모두 같지 않음
        i=1: window = ['23', '33', '33', '23', '19'] -> 모두 같지 않음
        i=2: window = ['33', '33', '23', '19', '157'] -> 모두 같지 않음
        ...
        모든 윈도우를 확인해도 연속된 5개가 같지 않음
        -> majority = Counter(answers).most_common(1)[0][0] = '33'
        -> return (1, len(answers), sum(tokens))  # '33' == '33'이므로 정답
        
        만약 answers = ['33', '33', '33', '33', '33', ...]이고 window_size=5라면:
        i=0: window = ['33', '33', '33', '33', '33'] -> 모두 같음!
        -> return (1, 5, sum(tokens[0:5]))  # 5번째에서 중단
    """
    if not answers:
        return 0, 0, 0

    steps = window_size - 1
    window_size_adjusted = min(window_size, len(answers))

    for i in range(len(answers) - window_size_adjusted + 1):
        window = answers[i:i + window_size_adjusted]
        window_tokens = tokens[i:i + window_size_adjusted]
        steps += 1
        # 정규화된 답변으로 비교
        normalized_window = [normalize_answer(ans) for ans in window]
        if normalized_window.count(normalized_window[0]) == window_size_adjusted:
            tokens_used = sum(window_tokens)
            return (1 if normalized_window[0] == normalize_answer(correct) else 0), steps, tokens_used

    # 정규화된 답변으로 Counter를 만들어 majority voting
    normalized_answers = [normalize_answer(ans) for ans in answers]
    majority_normalized = Counter(normalized_answers).most_common(1)[0][0]
    return (1 if majority_normalized == normalize_answer(correct) else 0), len(answers), sum(tokens)


def calculate_esc_ver_rasc(samples, window_size):
    """Early Stopping (ESC) 정확도 계산.
    
    예시:
        samples = [{
            'cot_answers': ['30', '23', '33', '33', '23', '19', '157', '33', ...],
            'correct_answer': '33',
            'input_tokens': 2312,
            'cot_tokens': [1564, 1344, 1669, 1885, 1394, 1204, 1602, 1867, ...]
        }, ...]
        window_size = 5
        
        각 샘플에 대해 evaluate_esc_for_sample 호출하여:
        - flag: 정답 여부 (1 또는 0)
        - steps: 사용된 체인 수
        - output_tokens_used: 사용된 출력 토큰 수
    """
    if window_size <= 0:
        return None

    correct_flags = []
    total_steps = 0
    total_input_tokens = 0
    total_output_tokens = 0

    for sample in samples:
        flag, steps, output_tokens_used = evaluate_esc_for_sample_ver_rasc(
            sample['cot_answers'],
            sample.get('cot_tokens', []),
            sample['correct_answer'],
            window_size
        )
        sample['esc_correct'] = flag
        sample['esc_steps'] = steps

        correct_flags.append(flag)
        total_steps += steps
        # ESC는 steps만큼 input_tokens 사용
        total_input_tokens += sample.get('input_tokens', 0) * steps
        total_output_tokens += output_tokens_used

    accuracy = sum(correct_flags) / len(samples) if samples else 0.0
    avg_chains = total_steps / len(samples) if samples else 0.0
    avg_input_tokens = total_input_tokens / len(samples) if samples else 0.0
    avg_output_tokens = total_output_tokens / len(samples) if samples else 0.0
    return accuracy, avg_chains, avg_input_tokens, avg_output_tokens


def evaluate_esc_for_sample_ver_esc(answers, tokens, correct, window_size):
    """Early Stopping (ESC) 평가 - ESC 깃허브 버전: 배치 단위로 처리하고 조기 종료 후에도 전체 답변으로 예측.
    
    예시 (window_size=5, sc_num=40):
        answers = ['30', '23', '33', '33', '23', '19', '33', '33', '33', '33', '33', ...]
        tokens = [1564, 1344, 1669, 1885, 1394, 1204, 1602, 1867, ...]
        correct = '33'
        
        index=0: cur_batch = ['30', '23', '33', '33', '23'] -> pre_dict에 3개 다른 값 -> 계속
        index=1: cur_batch = ['19', '33', '33', '33', '33'] -> pre_dict에 2개 다른 값 -> 계속
        index=2: cur_batch = ['33', '33', '33', '33', '33'] -> pre_dict에 1개 값 (모두 같음) -> break
        
        predict_list = ['30', '23', '33', '33', '23', '19', '33', '33', '33', '33', '33', ...]
        최종 예측: 전체 predict_list의 majority = '33'
        사용된 체인 수: (index + 1) * window_size = 3 * 5 = 15
    """
    if not answers:
        # flag, steps, tokens_used, num_windows
        return 0, 0, 0, 0
    
    sc_num = len(answers)
    predict_list = []
    max_iterations = sc_num // window_size if window_size > 0 else 0
    index = -1  # 초기화
    
    # 배치 단위로 처리
    for index in range(max_iterations):
        cur_batch = answers[index * window_size: (index + 1) * window_size]
        if not cur_batch:
            break
        
        # 현재 배치의 답변들을 predict_list에 추가
        for answer in cur_batch:
            predict_list.append(answer)
        
        # 현재 배치 내 모든 답변이 동일한지 확인 (정규화된 답변으로 비교)
        normalized_batch = [normalize_answer(ans) for ans in cur_batch]
        pre_dict = {}
        for norm_ans in normalized_batch:
            pre_dict[norm_ans] = pre_dict.get(norm_ans, 0) + (1 / len(cur_batch))
        
        # 조기 종료 조건: 현재 배치 내 모든 답변이 동일
        if len(pre_dict) == 1:
            break
    
    # 최종 예측: 전체 predict_list에 대해 majority voting (정규화된 답변으로)
    if not predict_list:
        # predict_list가 비어있으면 전체 answers 사용
        predict_list = answers
    
    # 정규화된 답변으로 majority voting
    normalized_predict_list = [normalize_answer(ans) for ans in predict_list]
    final_pre_dict = {}
    for norm_ans in normalized_predict_list:
        final_pre_dict[norm_ans] = final_pre_dict.get(norm_ans, 0) + (1 / len(normalized_predict_list))
    
    # 가장 많이 나온 답변 선택
    majority_normalized = max(final_pre_dict, key=final_pre_dict.get)
    flag = 1 if majority_normalized == normalize_answer(correct) else 0
    
    # 사용된 체인 수: (index + 1) * window_size
    if index >= 0:
        steps = (index + 1) * window_size
        # 윈도우(배치)가 sliding/호출된 횟수
        num_windows = index + 1
    else:
        steps = len(answers)
        # 루프에 한 번도 들어가지 않은 경우 (answers 수가 window_size보다 작을 때 등)
        num_windows = 1 if len(answers) > 0 else 0
    steps = min(steps, len(answers))  # 전체 답변 수를 넘지 않도록
    
    # 사용된 토큰 계산
    tokens_used = sum(tokens[:steps]) if len(tokens) >= steps else sum(tokens)
    
    return flag, steps, tokens_used, num_windows


def calculate_esc_ver_esc(samples, window_size):
    """Early Stopping (ESC) 정확도 계산 - ESC 깃허브 버전.
    
    예시:
        samples = [{
            'cot_answers': ['30', '23', '33', '33', '23', '19', '33', '33', '33', '33', '33', ...],
            'correct_answer': '33',
            'input_tokens': 2312,
            'cot_tokens': [1564, 1344, 1669, 1885, 1394, 1204, 1602, 1867, ...]
        }, ...]
        window_size = 5
        
        각 샘플에 대해 evaluate_esc_for_sample_ver_esc 호출하여:
        - flag: 정답 여부 (1 또는 0)
        - steps: 사용된 체인 수
        - output_tokens_used: 사용된 출력 토큰 수
    """
    if window_size <= 0:
        return None

    correct_flags = []
    total_steps = 0
    total_input_tokens = 0
    total_output_tokens = 0

    for sample in samples:
        flag, steps, output_tokens_used, num_windows = evaluate_esc_for_sample_ver_esc(
            sample['cot_answers'],
            sample.get('cot_tokens', []),
            sample['correct_answer'],
            window_size
        )
        sample['esc_esc_correct'] = flag
        sample['esc_esc_steps'] = steps

        correct_flags.append(flag)
        total_steps += steps
        # ESC는 window가 sliding/호출된 횟수만큼 input_tokens 사용
        total_input_tokens += sample.get('input_tokens', 0) * num_windows
        total_output_tokens += output_tokens_used

    accuracy = sum(correct_flags) / len(samples) if samples else 0.0
    avg_chains = total_steps / len(samples) if samples else 0.0
    avg_input_tokens = total_input_tokens / len(samples) if samples else 0.0
    avg_output_tokens = total_output_tokens / len(samples) if samples else 0.0
    return accuracy, avg_chains, avg_input_tokens, avg_output_tokens


def evaluate_ac_for_sample(answers, tokens, correct, ac):
    """Adaptive Self-Consistency (AC) 평가 - Beta 기준으로 적응적 중단.
    
    예시:
        answers = ['30', '23', '33', '33', '23', '19', '157', '33', ...]
        tokens = [1564, 1344, 1669, 1885, 1394, 1204, 1602, 1867, ...]
        correct = '33'
        ac = AC(stop_criteria=BetaStoppingCriteria(beta=0.95), max_gens=40)
        
        i=0: ac.should_stop(['30']) -> False, 계속
        i=1: ac.should_stop(['30', '23']) -> False, 계속
        i=2: ac.should_stop(['30', '23', '33']) -> False, 계속
        i=3: ac.should_stop(['30', '23', '33', '33']) -> True (beta=0.95 기준 충족)
        -> break
        
        truncated_answers = ['30', '23', '33', '33']
        majority = Counter(['30', '23', '33', '33']).most_common(1)[0][0] = '33'
        flag = 1  # '33' == '33' (정답)
        return (1, 4, sum([1564, 1344, 1669, 1885]))
    """
    if not answers:
        return 0, 0, 0

    for i, answer in enumerate(answers):
        if ac.should_stop(answers[:i+1]):
            break
    
    truncated_answers = answers[:i+1]
    truncated_tokens = tokens[:i+1]
    
    if not truncated_answers:
        return 0, 0, 0
    # 정규화된 답변으로 Counter를 만들어 majority voting
    normalized_truncated = [normalize_answer(ans) for ans in truncated_answers]
    majority_normalized = Counter(normalized_truncated).most_common(1)[0][0]
    flag = 1 if majority_normalized == normalize_answer(correct) else 0
    return flag, len(truncated_answers), sum(truncated_tokens)


def calculate_ac(samples, beta):
    """Adaptive Self-Consistency (AC) 정확도 계산.
    
    예시:
        samples = [{
            'cot_answers': ['30', '23', '33', '33', '23', '19', '157', '33', ...],
            'correct_answer': '33',
            'input_tokens': 2312,
            'cot_tokens': [1564, 1344, 1669, 1885, 1394, 1204, 1602, 1867, ...]
        }, ...]
        beta = 0.95
        
        각 샘플에 대해 evaluate_ac_for_sample 호출하여:
        - flag: 정답 여부 (1 또는 0)
        - steps: 사용된 체인 수 (beta 기준으로 중단된 지점)
        - output_tokens_used: 사용된 출력 토큰 수
    """
    if beta <= 0:
        return None

    ac = AC(stop_criteria=BetaStoppingCriteria(beta), max_gens=40)
    correct_flags = []
    total_steps = 0
    total_input_tokens = 0
    total_output_tokens = 0

    for sample in samples:
        flag, steps, output_tokens_used = evaluate_ac_for_sample(
            sample['cot_answers'],
            sample.get('cot_tokens', []),
            sample['correct_answer'],
            ac
        )
        sample['ac_correct'] = flag
        sample['ac_steps'] = steps

        correct_flags.append(flag)
        total_steps += steps
        # AC는 steps만큼 input_tokens 사용
        total_input_tokens += sample.get('input_tokens', 0) * steps
        total_output_tokens += output_tokens_used

    accuracy = sum(correct_flags) / len(samples) if samples else 0.0
    avg_chains = total_steps / len(samples) if samples else 0.0
    avg_input_tokens = total_input_tokens / len(samples) if samples else 0.0
    avg_output_tokens = total_output_tokens / len(samples) if samples else 0.0
    return accuracy, avg_chains, avg_input_tokens, avg_output_tokens


def evaluate_file(model_name, json_file, esc_window_size, ac_beta, sc_num_chains=None, sc_num_trials=10, eval_dir=None, eval_subdir=None):
    """
    Args:
        sc_num_chains: 리스트 형태의 chain 개수들 (예: [10, 20, 40])
    """
    if eval_dir is None:
        eval_dir = EVAL_DIR
    if eval_subdir is None:
        eval_subdir = "Evaluation_CoTs"
    json_path = os.path.join(eval_dir, f'{eval_subdir}/{model_name}/{json_file}')

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    log_lines = []

    def log(message=""):
        print(message)
        log_lines.append(message)

    log("#" * 50)
    log(f"Loading JSON file: {json_path}")
    json_data = load_json(json_path)
    samples = prepare_samples(json_data)
    total = len(samples)
    log(f"Loaded {total} samples\n")

    cot_acc, cot_avg_input, cot_avg_output = calculate_cot(samples)
    log(f"CoT Accuracy: {cot_acc:.4f} (avg_input_tokens={cot_avg_input:.2f}, avg_output_tokens={cot_avg_output:.2f})")

    # 샘플링 기반 SC 평가 (num_chains 리스트가 지정된 경우)
    if sc_num_chains is not None and len(sc_num_chains) > 0:
        for num_chains in sc_num_chains:
            if num_chains > 0:
                sc_sampled_metrics = calculate_sc_sampled(samples, num_chains, sc_num_trials)
                if sc_sampled_metrics:
                    sc_sampled_acc, sc_sampled_avg_input, sc_sampled_avg_output = sc_sampled_metrics
                    log(f"SC Accuracy(N={num_chains}, T={sc_num_trials}): {sc_sampled_acc:.4f} (avg_input_tokens={sc_sampled_avg_input:.2f}, avg_output_tokens={sc_sampled_avg_output:.2f})")

    esc_rasc_metrics = calculate_esc_ver_rasc(samples, esc_window_size)
    if esc_rasc_metrics:
        esc_rasc_acc, esc_rasc_avg_chains, esc_rasc_avg_input, esc_rasc_avg_output = esc_rasc_metrics
        log(f"ESC_rasc Accuracy: {esc_rasc_acc:.4f} (window={esc_window_size}, avg_chains={esc_rasc_avg_chains:.2f}, avg_input_tokens={esc_rasc_avg_input:.2f}, avg_output_tokens={esc_rasc_avg_output:.2f})")
    else:
        esc_rasc_acc = esc_rasc_avg_chains = esc_rasc_avg_input = esc_rasc_avg_output = None

    esc_esc_metrics = calculate_esc_ver_esc(samples, esc_window_size)
    if esc_esc_metrics:
        esc_esc_acc, esc_esc_avg_chains, esc_esc_avg_input, esc_esc_avg_output = esc_esc_metrics
        log(f"ESC_esc Accuracy: {esc_esc_acc:.4f} (window={esc_window_size}, avg_chains={esc_esc_avg_chains:.2f}, avg_input_tokens={esc_esc_avg_input:.2f}, avg_output_tokens={esc_esc_avg_output:.2f})")
    else:
        esc_esc_acc = esc_esc_avg_chains = esc_esc_avg_input = esc_esc_avg_output = None

    ac_metrics = calculate_ac(samples, ac_beta)
    if ac_metrics:
        ac_acc, ac_avg_chains, ac_avg_input, ac_avg_output = ac_metrics
        log(f"AC Accuracy: {ac_acc:.4f} (beta={ac_beta}, avg_chains={ac_avg_chains:.2f}, avg_input_tokens={ac_avg_input:.2f}, avg_output_tokens={ac_avg_output:.2f})")
    else:
        ac_acc = ac_avg_chains = ac_avg_input = ac_avg_output = None

    log("")  # blank line between sections

    return log_lines


def main(args):
    eval_dir = args.eval_dir if hasattr(args, 'eval_dir') and args.eval_dir else EVAL_DIR
    eval_subdir = args.eval_subdir if hasattr(args, 'eval_subdir') and args.eval_subdir else "Evaluation_CoTs"
    eval_root = os.path.join(eval_dir, eval_subdir)
    if not os.path.isdir(eval_root):
        raise FileNotFoundError(f"Evaluation directory not found: {eval_root}")

    found_any = False

    for dirpath, dirnames, filenames in os.walk(eval_root):
        json_files = sorted(f for f in filenames if f.lower().endswith('.json'))
        if not json_files:
            continue

        # model_name should correspond to the relative path under Evaluation_CoTs
        rel_path = os.path.relpath(dirpath, eval_root)
        full_path = rel_path.replace("\\", "/")
        
        # 모드별 폴더(non-thinking, thinking)인지 확인
        # 모든 모델(Qwen3, NVIDIA-Nemotron 등)에 대해 thinking/non-thinking 구조 지원
        path_parts = full_path.split("/")
        if len(path_parts) > 0 and path_parts[-1] in ["non-thinking", "thinking"]:
            # 모드별 폴더인 경우: model_name은 모드 폴더 제외, mode는 마지막 부분
            mode = path_parts[-1]
            model_name = "/".join(path_parts[:-1])
        else:
            # 모드별 폴더가 없는 경우 (직접 모델 폴더에 JSON 파일이 있는 경우)
            model_name = full_path
            mode = None
        
        model_log_lines = []

        for json_file in json_files:
            found_any = True
            if mode:
                print("\n" + "#" * 60)
                print(f"Evaluating {model_name} ({mode}) - {json_file}")
                print("#" * 60)
            else:
                print("\n" + "#" * 60)
                print(f"Evaluating {model_name} - {json_file}")
                print("#" * 60)
            log_lines = evaluate_file(
                full_path, json_file, args.esc_window_size, args.ac_beta,
                sc_num_chains=getattr(args, 'sc_num_chains', None),
                sc_num_trials=getattr(args, 'sc_num_trials', 10),
                eval_dir=eval_dir, eval_subdir=eval_subdir
            )
            model_log_lines.extend(log_lines)
            model_log_lines.append("")  # separator line

        if model_log_lines:
            # eval_result.txt는 현재 디렉토리(dirpath)에 저장
            result_log_path = os.path.join(dirpath, 'eval_result.txt')
            with open(result_log_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(model_log_lines).rstrip() + "\n")
            print(f"\n[Saved] {result_log_path}")

    if not found_any:
        print(f"No JSON files found under {eval_root}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate CoT, SC, ESC, and AC for all JSON results.')
    parser.add_argument('--esc_window_size', type=int, default=5,
                        help='Window size for ESC calculation (0 to skip ESC)')
    parser.add_argument('--ac_beta', type=float, default=0.95,
                        help='Beta value for AC calculation (0 to skip AC)')
    parser.add_argument('--sc_num_chains', type=int, nargs='+', default=None,
                        help='Number of chains to use for sampled SC evaluation (e.g., --sc_num_chains 10 20 40)')
    parser.add_argument('--sc_num_trials', type=int, default=10,
                        help='Number of trials for sampled SC evaluation (default: 10)')
    parser.add_argument('--eval_dir', type=str, default=None,
                        help='Evaluation directory path (default: ../eval)')
    parser.add_argument('--eval_subdir', type=str, default=None,
                        help='Evaluation subdirectory name (default: Evaluation_CoTs)')

    main(parser.parse_args())