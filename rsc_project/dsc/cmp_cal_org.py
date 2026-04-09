# SC, ASC, ESC, DSC 기법별 성능/비용 비교

import math
import os
import operator
import json
import re
import time

import numpy as np
import argparse
import sys
from collections import Counter
from typing import Dict, List, Tuple, Optional

# _strip_string 및 delete_extra_zero 함수 정의 (utils.py에서 복사)
def delete_extra_zero(n):
    try:
        n=float(n)
    except:
        print("None {}".format(n))
        return n
    if isinstance(n, int):
        return str(n)
    if isinstance(n, float):
        n = str(n).rstrip('0')  
        n = int(n.rstrip('.')) if n.endswith('.') else float(n)  
        n=str(n)
        return n

def _fix_fracs(string):
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
                except:
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

def _fix_a_slash_b(string):
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
    except:
        return string

def _remove_right_units(string):
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        return splits[0]
    else:
        return string

def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if len(split) == 0:
            new_substr = "\\sqrt"
        elif split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string

def _strip_string(string):
    # linebreaks
    string = string.replace("\n", "")
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
    # remove dollar signs
    string = string.replace("\\$", "")
    # remove units (on the right)
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
# from math_equivalence import is_equiv
import statistics
import random
from tqdm import tqdm, trange
from copy import deepcopy
from typing import List, Dict
from collections import Counter
from collections import Counter
from scipy import integrate, stats
import transformers

class StoppingCriterias:

    def __init__(self, *args, **kwargs):

        ...

    def should_stop(self, *args, **kwargs) -> Dict:
        ...

class BetaStoppingCriteria(StoppingCriterias):
    """BetaStoppingCriteria 클래스 - stopping_criterias 모듈에서 복사"""
    
    def __init__(self, conf_thresh: float = 0.95) -> None:
        super().__init__()
        self.conf_thresh = conf_thresh

    def should_stop(self, answers: List, conf_thresh: int = None, verbose: bool = False) -> Dict:
        
        if conf_thresh is None: conf_thresh = self.conf_thresh

        most_common = Counter(answers).most_common(2)
        if len(most_common) == 1:
            a, b = most_common[0][1], 0
        else:
            a, b = most_common[0][1], most_common[1][1]
        a = float(a)
        b = float(b)

        return_dict = {
            'most_common': most_common[0][0],
            'prob': -1,
            'stop': False,
        }
            
        try:
            prob = integrate.quad(lambda x: x**(a) * (1-x)**(b), 0.5, 1)[0] / integrate.quad(lambda x: x**(a) * (1-x)**(b), 0, 1)[0]
        except Exception as e:
            # print error message
            print(f"Error during numerical integration: {e}")
            return_dict['stop'] = False
            return_dict['prob'] = -1
            return return_dict
        return_dict['prob'] = prob
        return_dict['stop'] = prob >= conf_thresh
        return return_dict

class MyStoppingCriteria(StoppingCriterias):

    def __init__(self, easy_conf_thresh: float = 0.95, hard_conf_thresh = 0.50) -> None:
        super().__init__()
        self.easy_conf_thresh = easy_conf_thresh
        self.hard_conf_thresh = hard_conf_thresh

    def should_stop(self, answers: List, easy_conf_thresh: int = None, hard_conf_thresh: int = None, verbose: bool = False) -> Dict:

        if easy_conf_thresh is None: easy_conf_thresh = self.easy_conf_thresh
        if hard_conf_thresh is None: hard_conf_thresh = self.hard_conf_thresh

        most_common = Counter(answers).most_common(2)
        if len(most_common) == 1:
            a, b = most_common[0][1], 0
        else:
            a, b = most_common[0][1], most_common[1][1]
        a = float(a)
        b = float(b)

        return_dict = {
            'most_common': most_common[0][0],
            'prob': -1,
            'easy_stop': False,
            'hard_stop': False,
        }

        try:
            prob = integrate.quad(lambda x: x ** (a) * (1 - x) ** (b), 0.5, 1)[0] / \
                   integrate.quad(lambda x: x ** (a) * (1 - x) ** (b), 0, 1)[0]
        except Exception as e:
            # print error message
            print(f"Error during numerical integration: {e}")
            return_dict['easy_stop'] = False
            return_dict['prob'] = -1
            return return_dict
        return_dict['prob'] = prob
        return_dict['easy_stop'] = prob >= easy_conf_thresh
        return_dict['hard_stop'] = prob <= hard_conf_thresh
        return return_dict

def extract_answer(completion):
    INVALID_ANS = "[invalid]"
    ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def extract_last_number(s):
    numbers = re.findall(r'\d+', s)
    if numbers:
        return int(numbers[-1])
    else:
        return None

def extract_last_letter_answer(generated_answer):
    answer_text = generated_answer.lower().split('the answer is')[-1]
    answer_text = ''.join(re.split(r'[^A-Za-z]', answer_text))
    return answer_text


def extract_strategy_answer(generated_answer):
    if 'the answer is yes' in generated_answer.lower():
        return "Yes"
    elif 'the answer is no' in generated_answer.lower():
        return "No"
    else:
        if "uncertain" in generated_answer.lower() or "unknown" in generated_answer.lower():
            return ""
        judge = generated_answer.strip("A: ").split(",")[0]
        if judge == "Yes" or judge == "No":
            return judge

        judge2 = generated_answer.lower()

        if "yes" in judge2 and "no" not in judge2:
            # print(generated_answer)
            return "Yes"
        if "no" in judge2 and "yes" not in judge2:
            # print(generated_answer)
            return "No"
        # print(generated_answer)
        return ""


def extract_coin_flip_answer(generated_answer):
    if 'the answer is yes' in generated_answer.lower():
        return "yes"
    elif 'the answer is no' in generated_answer.lower():
        return "no"
    else:
        return ""


def extract_answer_gpqa(content):
    """SC_generator_ours.py와 동일한 GPQA 파싱 로직 적용"""
    if content is None:
        return None

    # First, check for boxed format (e.g., \boxed{A. ~ 3.2})
    if 'boxed' in content:
        ans = content.split('boxed')[-1]
        if len(ans) and ans[0] == '{':
            # Extract content within braces
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
            # If not starting with '{', split by '$' or take until end
            a = ans.split('$')[0].strip()
        
        a = _strip_string(a)
        
        # Extract the first token (A, B, C, or D) from boxed content
        # Handles formats like "A. ~ 3.2", "A.", "A", or "\text{B. ~ 4.4}"
        if a:
            # Remove LaTeX commands like \text{...} and extract content
            # Match \text{...} or \textbf{...} etc. and extract the content inside
            text_match = re.search(r'\\text(?:bf|it|tt)?\{([^}]+)\}', a)
            if text_match:
                a = text_match.group(1)
            
            # Try to match A-D at the start, optionally followed by period and other text
            # Also handle cases where there might be whitespace or LaTeX commands before
            boxed_match = re.search(r'([A-D])[\.]?', a.strip())
            if boxed_match:
                return boxed_match.group(1).upper()

    # Regex to find "Answer: $LETTER" potentially with spaces, $, case-insensitive
    # It captures only the letter A, B, C, or D
    # Supports: "answer", "final answer", "correct answer"
    # Also supports "correct answer is:" format with words between "answer" and ":"
    # Allows newlines and whitespace between "answer" keyword and the letter
    # Use DOTALL flag to allow . to match newlines, and find the first A-D after "answer" keyword
    answer_keyword_pattern = r"(?i)(?:answer|final answer|correct answer)(?:\s+\w+)*\s*[:=\-]?\s*"
    # Find all occurrences of answer keywords
    answer_matches = list(re.finditer(answer_keyword_pattern, content))
    
    if answer_matches:
        # For each answer keyword, look for the first A-D letter after it
        for answer_match in reversed(answer_matches):  # Start from the last occurrence
            # Get text after the answer keyword
            text_after = content[answer_match.end():]
            # Find first A-D letter in the text after (allowing for whitespace/newlines)
            letter_match = re.search(r'[\s\$]*([A-D])[\.]?(?:\s|$)', text_after)
            if letter_match:
                return letter_match.group(1).upper()
    
    # Fallback: Try the original pattern approach
    ANSWER_PATTERN = r"(?i)(?:answer|final answer|correct answer)(?:\s+\w+)*\s*[:=\-]?\s*[\$]?\s*([A-D])[\$]?[\.]?(?:\s|$)"
    matches = list(re.finditer(ANSWER_PATTERN, content))
    
    if matches:
        # Get the last match found
        last_match = matches[-1]
        extracted_letter = last_match.group(1).upper()
        return extracted_letter
    else:
        # Fallback: Check if the *very last non-whitespace character* is A, B, C, or D
        # This is less robust but can catch cases where formatting is slightly off
        stripped_content = content.strip()
        if stripped_content and stripped_content[-1].upper() in ['A', 'B', 'C', 'D']:
            # Check if preceded by something like "Answer:" to reduce false positives
            last_part = stripped_content[-15:].lower()
            if any(keyword in last_part for keyword in ['answer']):
                return stripped_content[-1].upper()
        return None

def extract_answer_mmlu_pro(content):
    """SC_generator_ours.py와 동일한 MMLU-Pro 파싱 로직 적용"""
    if content is None:
        return None

    ANSWER_PATTERN = r"((?:[A-J]\b))"

    matches = list(re.finditer(ANSWER_PATTERN, content, flags=re.IGNORECASE))

    if matches:
        last = matches[-1]
        return last.group(1).upper()

    return None

def extract_common_answer(generated_answer):
    """GPQA와 MMLU-Pro를 구분하여 적절한 파싱 함수 호출"""
    # GPQA는 A-D만, MMLU-Pro는 A-J까지 사용
    # 일단 GPQA 방식으로 시도 (A-D)
    result = extract_answer_gpqa(generated_answer)
    if result:
        return result
    
    # MMLU-Pro 방식으로 시도 (A-J)
    result = extract_answer_mmlu_pro(generated_answer)
    if result:
        return result
    
    return ""

def extract_math_answer(pred_str):
    """SC_generator_ours.py와 동일한 파싱 로직 적용"""
    if pred_str is None:
        return ''
    pred_str = str(pred_str).strip()
    if not pred_str:
        return ''
    
    if ('The answer is ' in pred_str):
        pred = pred_str.split('The answer is ')[-1].strip()
    elif ('the answer is ' in pred_str):
        pred = pred_str.split('the answer is ')[-1].strip()
    elif 'boxed' in pred_str:
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
        a = _strip_string(a)
        pred = a
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
    
    pred = _strip_string(pred)
    
    if 'boxed' in pred:
        ans = pred.split('boxed')[-1]
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
        a = _strip_string(a)
        pred = a
    
    return pred


def extract_gsm8k_answer(pred_str):
    pred_str = re.sub(r'(\d),(\d)', r'\1\2', pred_str)
    if ('The answer is ' in pred_str):
       pred_str = pred_str.split('The answer is ')[-1].strip()
    elif ('the answer is ' in pred_str):
       pred_str = pred_str.split('the answer is ')[-1].strip()

    if 'boxed' in pred_str and 1==2:
        ans = pred_str.split('boxed')[-1]
        if (ans[0] == '{'):
            stack = 1
            a = ''
            for c in ans[1:]:
                if (c == '{'):
                    stack += 1
                    a += c
                elif (c == '}'):
                    stack -= 1
                    if (stack == 0): break
                    a += c
                else:
                    a += c
        else:
            a = ans.split('$')[0].strip()
        a = _strip_string(a)
        pred = a
        if pred != "":
            if pred[-1] == ".":
                pred = pred[:-1]

    else:
        pattern = '-?\d*\.?\d+'
        pred = re.findall(pattern, pred_str)
        if (len(pred) >= 1):
            # print(pred_str)
            pred = pred[-1]
        else:
            pred = ''

    try:
        if pred != '' and math.floor(float(pred)) == float(pred):
            pred = str(int(float(pred)))
    except:
        a = 1
    return pred


def ESC(input_list, tokens, window_size=5):
    esc_count = 1
    esc_num = 0
    while True:
        if esc_num + window_size - 1 >= len(input_list):
            flag = 0
            break
        sub = input_list[esc_num]
        flag = 1
        for i in range(1, window_size):
            if sub != input_list[esc_num+i]:
                flag = 0
                break

        if flag == 1:
            esc_num += window_size
            break
        else:
            esc_count += 1
            esc_num += window_size

    if flag == 0:
        return input_list, tokens
    else:
        out = {}
        out["input"] = tokens["input"]
        out["output"] = tokens["output"][:esc_num]
        return input_list[:esc_num], out

def ASC(input_list, tokens, threshold=0.95):
    stop_judge = BetaStoppingCriteria(threshold)
    stop_position = len(input_list)
    for i in range(len(input_list)):
        judge_result = stop_judge.should_stop(input_list[:i+1])
        if judge_result["stop"]:
            stop_position = i+1
            break

    out = {}
    out["input"] = tokens["input"]
    out["output"] = tokens["output"][:stop_position]
    return input_list[:stop_position], out




def DSC_hard(input_list, tokens, easy_threshold=0.95, hard_threshold=0.50, window_size=5):
    easy_stop_judge = MyStoppingCriteria(easy_threshold, hard_threshold)
    stop_position = len(input_list)
    Flag = False
    for i in range(len(input_list)//window_size):
        judge_result = easy_stop_judge.should_stop(input_list[:(i+1)*window_size])
        if judge_result["easy_stop"]:
            Flag = True
            stop_position = (i+1)*window_size
            break
        if judge_result["hard_stop"] and i>=len(input_list)//(2*window_size):
            Flag = True
            stop_position = (i+1)*window_size
            break

    out = {}
    out["input"] = tokens["input"]
    if Flag:
        out["output"] = tokens["output"][:stop_position]
        return input_list[:stop_position], out
    else:
        out["output"] = tokens["output"]
        return input_list, out

def DSC(input_list, greedy_list, tokens, greedy_tokens, allocate, easy_threshold=0.95, hard_threshold=0.50, window_size=5):
    stop_position = len(input_list)
    Flag = False
    if allocate == 1:
        out = {}
        out["input"] = greedy_tokens["input"]
        out["output"] = greedy_tokens["output"]
        return greedy_list, out
    else:
        easy_stop_judge = MyStoppingCriteria(easy_threshold, 0)

        for i in range(len(input_list)//window_size):
            judge_result = easy_stop_judge.should_stop(input_list[:(i+1)*window_size])
            if judge_result["easy_stop"]:
                Flag = True
                stop_position = (i+1)*window_size
                break
            #if judge_result["hard_stop"] and i >= len(input_list)//(window_size*2):
            #    Flag = True
            #    stop_position = (i+1)*window_size
            #    break

        out = {}
        out["input"] = tokens["input"]
        if Flag:
            out["output"] = tokens["output"][:stop_position]
            return input_list[:stop_position], out
        else:
            out["output"] = tokens["output"]
            return input_list, out



def DSC_window(input_list, greedy_list, tokens, greedy_tokens, allocate, initial_window_size, extend_window_size=5):
    stop_position = len(input_list)
    Flag = False
    if allocate == 1:
        out = {}
        out["input"] = greedy_tokens["input"]
        out["output"] = greedy_tokens["output"]
        return greedy_list, out
    else:
        easy_stop_judge = MyStoppingCriteria(0.95, 0)
        count = 0
        while True:
            if initial_window_size + extend_window_size*count >= len(input_list):
                Flag = False
                break
            judge_result = easy_stop_judge.should_stop(input_list[:initial_window_size + extend_window_size*count])
            if judge_result["easy_stop"]:
                Flag = True
                stop_position = initial_window_size + extend_window_size*count
                break
            count += 1

        out = {}
        out["input"] = tokens["input"]
        if Flag:
            out["output"] = tokens["output"][:stop_position]
            return input_list[:stop_position], out
        else:
            out["output"] = tokens["output"]
            return input_list, out

def DSC_pre(input_list, greedy_list, tokens, greedy_tokens, allocate, initial_window_size, extend_window_size=5):
    stop_position = len(input_list)
    Flag = False

    easy_stop_judge = MyStoppingCriteria(0.95, 0)
    count = 0
    while True:
        if initial_window_size + extend_window_size*count >= len(input_list):
            Flag = False
            break
        judge_result = easy_stop_judge.should_stop(input_list[:initial_window_size + extend_window_size*count])
        if judge_result["easy_stop"]:
            Flag = True
            stop_position = initial_window_size + extend_window_size*count
            break
        count += 1

    out = {}
    out["input"] = tokens["input"]
    if Flag:
        out["output"] = tokens["output"][:stop_position]
        return input_list[:stop_position], out
    else:
        out["output"] = tokens["output"]
        return input_list, out


def get_true_window(easy_stop_judge, input_list, init_window_size=4):
    out = init_window_size
    for i in range(init_window_size, len(input_list)):
        judge_result = easy_stop_judge.should_stop(input_list[:i])
        if judge_result["easy_stop"]:
            out = i
            break
    return out
def get_GSM8K_prompt():
    instruction = "You are a helpful assistant. Think the question step by step and give the answer:\n"
    return instruction


def get_MATH_prompt():
    instruction = "Think the question step by step and give the answer.\n"
    return instruction

def get_coin_flip_prompt():
    instruction = "You are a helpful assistant. Think the question step by step and give the answer:\n"
    return instruction


def get_last_letter_prompt():
    instruction = "You are a helpful assistant. Think the question step by step and give the answer:\n"
    return instruction

def get_common_prompt():
    instruction = "You are a helpful assistant. Think the question step by step and give the answer:\n"
    return instruction

def get_strategy_prompt():
    instruction = "You are a helpful assistant. Think the question step by step and give the answer:\n"
    return instruction
def cal_tokens(input, tokenizer):
    tokens = tokenizer.tokenize(input)
    return len(tokens)

def cal_sc_tokens(input, output_list, tokenizer):
    input_token_num = cal_tokens(input, tokenizer)
    output_token_num = sum([cal_tokens(output_list[i], tokenizer) for i in range(len(output_list))])

    return input_token_num, output_token_num


def cal_esc_tokens(input, output_list, tokenizer, window_size=5):
    input_token_num = cal_tokens(input, tokenizer)*len(output_list)//window_size
    output_token_num = sum([cal_tokens(output_list[i], tokenizer) for i in range(len(output_list))])
    return input_token_num, output_token_num


def cal_asc_tokens(input, output_list, tokenizer):
    input_token_num = cal_tokens(input, tokenizer)*len(output_list)
    output_token_num = sum([cal_tokens(output_list[i], tokenizer) for i in range(len(output_list))])
    return input_token_num, output_token_num

def random_elements(A, input_tokens, num):
    indices = random.sample(range(len(A)), num)
    selected_A = [A[i] for i in indices]
    out = {}
    out["input"] = input_tokens["input"]
    out["output"] = [input_tokens["output"][i] for i in indices]
    return selected_A, out

def get_dollars(input_token, output_token, gpt_version):
    if gpt_version=="3.5":
        price = 0.5*input_token/1000000 + 1.5*output_token/1000000
    else:
        price = 30*input_token/1000000 + 60*output_token/1000000

    return price

def calculate_entropy(answers):
    """
    Calculate entropy of answer distribution
    Args:
        answers: List of answer strings
    Returns:
        entropy value (float)
    """
    if not answers:
        return 0.0
    
    # Count frequency of each answer
    counter = Counter(answers)
    total = len(answers)
    
    # Calculate entropy: H = -sum(p_i * log2(p_i))
    entropy = 0.0
    for count in counter.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    
    return entropy

def dsc_allocate(pre_list, judge_window_size, sample_window_size, max_sample_size, pre_tokens=None, pre_sample_size=4): 
    """
    Algorithm 2: Problem Partition (논문 알고리즘 정확히 구현)
    Require: Questions from hard to easy Q_1:N, Pre-sample size p, Judge window k
    Ensure: Anchor point A, Easy part questions Q_Easy, Hard part questions Q_Hard
    """
    length = len(pre_list)
    allocate = [0 for i in range(length)]
    
    # Step 2에서 사용된 샘플의 토큰 수 추적
    step2_processed_problems = set()
    step2_input_tokens = 0
    step2_output_tokens = 0
    
    # Algorithm 2: E_all ← []; a ← 1
    E_all = []
    anchor = None  # 논문의 A
    
    # Algorithm 2: for i ← 1, N do
    # (pre_list는 이미 hard to easy 순서로 정렬되어 있음)
    for i in range(length):
        # Algorithm 2: S_current ← M(Q_i, D, p)
        # 실제로는 이미 생성된 SC 샘플에서 p개 선택
        num_samples = min(pre_sample_size, len(pre_list[i]))
        random_indices = random.sample(range(len(pre_list[i])), num_samples)
        S_current = [pre_list[i][idx] for idx in random_indices]
        
        # Algorithm 2: E_all ← E_all.Append(Entropy(S_current))
        entropy = calculate_entropy(S_current)
        E_all.append(entropy)
        
        # 토큰 수 계산 (문제 i마다 p개 샘플 사용)
        if pre_tokens is not None and i < len(pre_tokens) and i not in step2_processed_problems:
            step2_input_tokens += pre_tokens[i]["input"]
            step2_output_tokens += sum([pre_tokens[i]["output"][idx] for idx in random_indices])
            step2_processed_problems.add(i)
        
        # Algorithm 2: if i > k and Sum(E_all[i-k : i-1]) == 0 then
        # 논문은 1-indexed: i > k는 i >= k+1을 의미
        # 0-indexed로 변환: i >= k (즉, i >= judge_window_size)
        # E_all[i-k : i-1]는 0-indexed로 E_all[i-k : i] (i-k부터 i-1까지, 총 k개)
        if i >= judge_window_size:
            # 마지막 k개의 엔트로피 합 계산 (i-k부터 i-1까지, 총 k개)
            window_start = i - judge_window_size  # i-k (0-indexed)
            window_end = i  # i-1까지 포함하려면 i까지 슬라이싱 (exclusive end)
            entropy_sum = sum(E_all[window_start:window_end])
            
            # Algorithm 2: if Sum(E_all[i-k : i-1]) == 0 then A ← i; Break
            # 엔트로피 합이 0이면 마지막 k개 문제의 엔트로피가 모두 0 (모든 답이 같음)
            if entropy_sum == 0.0:
                anchor = i + 1  # 논문은 1-indexed이므로 +1
                break
    
    # Algorithm 2: Q_Hard ← Q_1:A; Q_Easy ← Q_A+1:N
    # anchor가 None이면 모든 문제가 hard (엔트로피 합이 0인 구간이 없음)
    if anchor is None:
        # 모든 문제를 hard로 처리
        for i in range(length):
            allocate[i] = max_sample_size
    else:
        # Q_Hard = Q_1:A (0-indexed: 0 ~ anchor-1)
        for i in range(anchor):
            allocate[i] = max_sample_size
        # Q_Easy = Q_A+1:N (anchor ~ length-1)
        for i in range(anchor, length):
            allocate[i] = 1
    
    return allocate, step2_input_tokens, step2_output_tokens, len(step2_processed_problems)


def dsc_allocate_greedy(pre_list, judge_window_size, sample_window_size, max_sample_size): # input difficulty ordered list
    length = len(pre_list)
    allocate = [0 for i in range(length)]
    easy_sample_flag = [False for i in range(length)]
    hard_sample_flag = [False for i in range(length)]
    left = 0
    right = length
    Flag = False
    for j in range(length):
        random_choice = random.sample(pre_list[length-j-1], max_sample_size)
        sub_allocate = DSC_judge(random_choice, max_sample_size=max_sample_size, easy_threshold=0.95, hard_threshold=0,
                             window_size=sample_window_size)
        if sub_allocate == sample_window_size:
            easy_sample_flag[length-j-1] = True

        if j >= judge_window_size-1:
            if all(tem == True for tem in easy_sample_flag[length-j-1:length-j-1+judge_window_size]):
                Flag = True
                for i in range(length-j-1+judge_window_size):
                    allocate[i] = 1
                break


    for i in range(len(allocate)):
        if allocate[i] == 0:
            allocate[i] = -1

    return allocate

def DSC_judge(input_list, max_sample_size, easy_threshold=0.95, hard_threshold=0.50, window_size=5):
    easy_stop_judge = MyStoppingCriteria(easy_threshold, hard_threshold)
    Flag = False
    for i in range(max_sample_size // window_size):
        judge_result = easy_stop_judge.should_stop(input_list[:(i+1)*window_size])
        if (i+1)*window_size >= max_sample_size:
            break
        if judge_result["easy_stop"]:
            Flag = True
            allocate = (i+1)*window_size
            break
        if judge_result["hard_stop"]:
            Flag = True
            allocate = (i+1)*window_size
            break
    if Flag:
        return allocate
    else:
        return max_sample_size

def split_easy_hard(eval_result, pre_batch, judge_window_size, window_size, max_sample_size, pre_tokens=None, pre_sample_size=4):
    question = eval_result["questions"]
    eval = eval_result["eval"]
    index = range(len(question))
    zipped = sorted(zip(eval, question, pre_batch, index), reverse=False)
    eval, question, pre_batch, index = zip(*zipped)
    
    # pre_tokens도 정렬
    sorted_pre_tokens = None
    if pre_tokens is not None:
        sorted_pre_tokens = [pre_tokens[i] for i in index]
    
    # 논문에 따르면 pre-sample size p = 4를 사용
    allocate_list, step2_input_tokens, step2_output_tokens, step2_processed_count = dsc_allocate(pre_batch, judge_window_size, window_size, max_sample_size, sorted_pre_tokens, pre_sample_size)
    zipped = sorted(zip(index, eval, question, pre_batch, allocate_list), reverse=False)
    index, eval, question, pre_batch, allocate_list = zip(*zipped)
    print("easy count={}".format(sum([sub == 1 for sub in allocate_list])))
    print("hard count={}".format(sum([sub == n for sub in allocate_list])))
    print("Step 2 processed problems: {} / {}".format(step2_processed_count, len(pre_batch)))
    return index, eval, question, pre_batch, allocate_list, step2_input_tokens, step2_output_tokens

def split_easy_hard_greedy(eval_result, pre_batch, judge_window_size, window_size, max_sample_size):
    question = eval_result["questions"]
    eval = eval_result["eval"]
    index = range(len(question))
    zipped = sorted(zip(eval, question, pre_batch, index), reverse=False)
    eval, question, pre_batch, index = zip(*zipped)
    allocate_list = dsc_allocate_greedy(pre_batch, judge_window_size, window_size, max_sample_size)
    zipped = sorted(zip(index, eval, question, pre_batch, allocate_list), reverse=False)
    index, eval, question, pre_batch, allocate_list = zip(*zipped)
    print("greedy easy count={}".format(sum([sub == 1 for sub in allocate_list])))
    print("greedy hard count={}".format(sum([sub == n for sub in allocate_list])))
    return index, eval, question, pre_batch, allocate_list


def get_min_allocate_size(easy_stop_judge, input, max_sample_size):
    out = max_sample_size
    for i in range(max_sample_size):
        judge_result = easy_stop_judge.should_stop(input[:i+1])
        if judge_result["easy_stop"]:
            Flag = True
            out = i+1
            break
    return out

# ============================================================
# DynSC (Dynamic Self-Consistency) - 제안 방법
# ============================================================

def normalize_question(q: str) -> str:
    """질문을 정규화하여 비교 가능하게 만듦"""
    return " ".join(q.strip().split())

def extract_answers_from_paths(paths: List[Dict]) -> List[str]:
    """Reasoning Paths에서 답변 추출"""
    out = []
    for p in paths:
        for k, v in p.items():
            if k.startswith("Final Answer"):
                out.append(str(v).strip())
    return out

def load_easy_hard_split(easy_path: str, hard_path: str) -> Dict[str, str]:
    """Easy/Hard 분할 정보 로드"""
    mapping: Dict[str, str] = {}
    
    def load_jsonl(path: str) -> List[Dict]:
        data = []
        if not os.path.exists(path):
            return data
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data
    
    def ingest(path: str, label: str):
        if not os.path.exists(path):
            return
        for item in load_jsonl(path):
            if "id" in item:
                mapping[str(item["id"])] = label
            q = item.get("question") or item.get("problem")
            if q:
                mapping["q::" + normalize_question(q)] = label
    
    ingest(easy_path, "easy")
    ingest(hard_path, "hard")
    return mapping

def resolve_diff(x: Dict, diff_map: Dict[str, str]) -> str:
    """문제의 난이도 결정"""
    qid = x.get("id")
    
    # 1) exact id
    if qid is not None and str(qid) in diff_map:
        return diff_map[str(qid)]
    
    # 2) numeric fallback
    if qid is not None:
        num = "".join(c for c in str(qid) if c.isdigit())
        if num and num in diff_map:
            return diff_map[num]
    
    # 3) question/problem fallback
    q = x.get("question") or x.get("problem")
    if q:
        k = "q::" + normalize_question(q)
        if k in diff_map:
            return diff_map[k]
    
    return "unknown"

def dynamic_sc(answers: List[str], target_conf: float, window_size: int = 5) -> Tuple[str, int, int]:
    """
    동적으로 샘플 수 조절:
    - 초기: window_size개 생성
    - 이후: Need 값에 따라 동적으로 추가
    - Need = window_size - f_window(답변)
    - majority ratio >= target_conf 이면 stop
    return: pred, used_chains, turns
    """
    if not answers:
        return "", 0, 0

    pool = answers[:40]
    used = 0
    turns = 0

    # Turn 1: 초기 window_size개 생성
    used = min(window_size, len(pool))
    turns = 1

    while used < len(pool):
        # 현재까지의 답변 분석
        freq = Counter(pool[:used])
        top_answer, top_cnt = max(freq.items(), key=lambda x: (x[1], -pool.index(x[0])))
        
        # Overall confidence 계산
        conf = top_cnt / used
        
        # target_conf 이상이면 중단
        if conf >= target_conf:
            break
        
        # Window Analysis: 최근 window_size개의 답변 분석
        window_start = max(0, used - window_size)
        window_answers = pool[window_start:used]
        window_freq = Counter(window_answers)
        f_window = window_freq.get(top_answer, 0)
        
        # Need 계산: window_size - f_window(답변)
        need = window_size - f_window
        
        # Need만큼 추가 생성 (최소 1개, 최대 남은 샘플 수)
        take = min(max(need, 1), len(pool) - used)
        used += take
        turns += 1

    # 최종 답변 결정
    freq = Counter(pool[:used])
    top_answer, _ = max(freq.items(), key=lambda x: (x[1], -pool.index(x[0])))
    
    return top_answer, used, turns

def normalize_answer_for_ActSC(ans) -> str:
    """답변을 정규화하여 비교 가능하게 만듦"""
    if ans is None:
        return ""
    return str(ans).strip().replace(" ", "").lower()

def eval_ActSC(sc_data: List[Dict], diff_map: Dict[str, str], target_conf: float, 
               extract_answer_func, tokenizer, prompt: str) -> Tuple[float, float, float, float, float, float, int, int]:
    """
    DynSC 평가
    return: (acc, avg_gen, avg_input_tokens, avg_output_tokens, easy_acc, hard_acc, easy_count, hard_count)
    """
    correct = 0
    total_gen = 0
    total_input_tokens = 0
    total_output_tokens = 0
    n = 0
    easy_correct = 0
    easy_count = 0
    hard_correct = 0
    hard_count = 0

    for x in sc_data:
        diff = resolve_diff(x, diff_map)
        reasoning_paths = x.get("Reasoning Paths", [])
        answers = extract_answers_from_paths(reasoning_paths)
        gold = normalize_answer_for_ActSC(x.get("answer"))
        
        if not answers:
            continue

        if diff == "easy" and answers:
            pred, used, turns = answers[0], 1, 1  # Easy는 1턴만 사용
        elif diff == "hard":
            pred, used, turns = dynamic_sc(answers, target_conf)
        else:
            continue

        # 정확도 계산
        pred_normalized = normalize_answer_for_ActSC(pred)
        is_correct = int(pred_normalized == gold)
        correct += is_correct
        total_gen += used
        n += 1
        
        # Easy/Hard별 정확도 추적
        if diff == "easy":
            easy_correct += is_correct
            easy_count += 1
        elif diff == "hard":
            hard_correct += is_correct
            hard_count += 1
        
        # 토큰 수 계산
        # JSON 파일에 저장된 실제 토큰 수가 있으면 사용, 없으면 tokenizer로 계산
        question = x.get("question") or x.get("problem") or x.get("input", "")
        if "input_tokens" in x:
            input_token = x["input_tokens"]
        else:
            input_token = cal_tokens(prompt + question, tokenizer)
        
        # 사용된 샘플만큼의 output 토큰 계산
        # JSON 파일에 저장된 Tokens_X 값 사용
        tokens_list = []
        for i in range(min(used, len(reasoning_paths))):
            path = reasoning_paths[i]
            tokens_key = f"Tokens_{i}"
            if tokens_key in path:
                try:
                    tokens_list.append(int(path[tokens_key]))
                except:
                    tokens_list.append(0)
            else:
                # Tokens_X 형식이 없으면 다른 방법 시도
                found = False
                for k, v in path.items():
                    if k.startswith("Tokens"):
                        try:
                            tokens_list.append(int(v))
                            found = True
                            break
                        except:
                            pass
                if not found:
                    tokens_list.append(0)
        
        # Input token은 턴 수만큼 반복되어야 함
        # (각 턴마다 input이 들어가므로)
        # Turn 1: window_size개 생성 → input token 1번
        # Turn 2: Need만큼 추가 생성 → input token 1번 추가
        # Turn 3: Need만큼 추가 생성 → input token 1번 추가
        # ...
        total_input_tokens += input_token * turns
        total_output_tokens += sum(tokens_list)

    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0
    easy_acc = easy_correct / easy_count if easy_count > 0 else 0.0
    hard_acc = hard_correct / hard_count if hard_count > 0 else 0.0
    return correct / n, total_gen / n, total_input_tokens / n, total_output_tokens / n, easy_acc, hard_acc, easy_count, hard_count

def allocate_run(pre_list, max_sample_size, interplot_size, window_size=4): # input difficulty ordered list
    length = len(pre_list)
    actual_window = [0 for i in range(length)]
    true_window = [0 for i in range(length)]

    easy_stop_judge = MyStoppingCriteria(0.95, 0)
    for j in range(length):
        if j <= interplot_size-1:
            current_window_size = window_size
        else:
            #current_window_size = min(true_window[max(0, j - interplot_size): j]) // 4 * 4
            current_window_size = round(sum(true_window[max(0, j-interplot_size): j]) / len(true_window[max(0, j-interplot_size): j])/4) * 4
        random_choice = random.sample(pre_list[j], max_sample_size)

        actual_window[j] = current_window_size
        true_window[j] = get_min_allocate_size(easy_stop_judge, random_choice, max_sample_size)
        if true_window[j] == 4:
            actual_window[j] = 4
    # windows = {'actual_window': actual_window, 'true_window': true_window}
    # with open("window.json", "w") as f:
    #     json.dump(windows, f)
    #     f.close()

    return actual_window


def allocate_window(eval_result, pre_batch, max_sample_size, interplot_size=5, window_size=4): # using greedy judge
    question = eval_result["questions"]
    eval = eval_result["eval"]
    index = range(len(question))
    zipped = sorted(zip(eval, question, pre_batch, index), reverse=False) # easy to hard
    eval, question, pre_batch, index = zip(*zipped)

    actual_window= allocate_run(pre_batch, max_sample_size, interplot_size, window_size)
    zipped = sorted(zip(index, eval, question, pre_batch, actual_window), reverse=False)
    index, eval, question, pre_batch, actual_window = zip(*zipped)

    return actual_window

def clean_data(pre_batch, eval_result, pre, pre_answer, pre_tokens, greedy_pre_tokens, greedy_pre_batch, sc_num):
    cleaned_data = []
    cleaned_eval_result = {}
    cleaned_pre, cleaned_pre_answer, cleaned_pre_tokens, cleaned_greedy_pre_tokens, cleaned_greedy_pre_batch = [], [], [], [], []
    # prepare_info는 딕셔너리이므로 별도로 보관
    prepare_info = None
    if "prepare_info" in eval_result:
        prepare_info = deepcopy(eval_result["prepare_info"])
    
    # 리스트 키만 초기화
    for key in eval_result.keys():
        if key != "prepare_info":
            cleaned_eval_result[key] = []
    
    # prepare_info는 그대로 유지
    if prepare_info is not None:
        cleaned_eval_result["prepare_info"] = prepare_info

    for i in range(len(pre_batch)):
        if len(pre_batch[i]) < sc_num:
            continue
        else:
            cleaned_data.append(deepcopy(pre_batch[i]))
            for key in eval_result.keys():
                if key != "prepare_info":
                    # 리스트 타입인 경우에만 인덱스로 접근
                    if isinstance(eval_result[key], list) and i < len(eval_result[key]):
                        cleaned_eval_result[key].append(deepcopy(eval_result[key][i]))

            cleaned_pre.append(deepcopy(pre[i]))
            cleaned_pre_answer.append(deepcopy(pre_answer[i]))
            cleaned_pre_tokens.append(deepcopy(pre_tokens[i]))
            cleaned_greedy_pre_batch.append(deepcopy(greedy_pre_batch[i]))
            cleaned_greedy_pre_tokens.append(deepcopy(greedy_pre_tokens[i]))

    return cleaned_data, cleaned_eval_result, cleaned_pre, cleaned_pre_answer, cleaned_pre_tokens, cleaned_greedy_pre_tokens, cleaned_greedy_pre_batch


def get_sc_sample_path(model_name, dataset_name):
    """
    SC 샘플 파일 경로를 생성하는 헬퍼 함수
    
    Args:
        model_name: 모델 이름 (예: "gemma3_4b", "qwen2.5_7b", "llama3.1_8b", "qwen2.5_3b")
        dataset_name: 데이터셋 이름 (예: "GSM8K", "AIME_2024", "MATH500_test", "MMLU-Pro", "GPQA-Diamond")
    
    Returns:
        SC 샘플 파일의 전체 경로
    """
    # 모델명 매핑
    # model_path_map = {
    #     "gemma3_4b": "google/gemma-3-4b-it",
    #     "qwen2.5_7b": "Qwen/Qwen2.5-7B-Instruct",
    #     "qwen2.5_3b": "Qwen/Qwen2.5-3B-Instruct",
    #     "llama3.1_8b": "meta-llama/Llama-3.1-8B-Instruct"
    # }

    model_path_map = {
        "qwen2.5_7b": "Qwen/Qwen2.5-7B-Instruct",
    }    
    # 데이터셋명 매핑 (필요시 확장)
    dataset_file_map = {
        "GSM8K": "GSM8K.json",
        "AIME_2024": "AIME_2024.json",
        "AIME_2025": "AIME_2025.json",
        "MATH500_test": "MATH500_test.json",
        "MMLU-Pro": "MMLU-Pro.json",
        "GPQA-Diamond": "GPQA-Diamond.json"
    }
    
    if model_name not in model_path_map:
        raise ValueError(f"Unknown model name: {model_name}. Available: {list(model_path_map.keys())}")
    
    if dataset_name not in dataset_file_map:
        raise ValueError(f"Unknown dataset name: {dataset_name}. Available: {list(dataset_file_map.keys())}")
    
    base_path = "/home/woong/ACL/Ours/results/Evaluation_CoTs_0-shot"
    model_path = model_path_map[model_name]
    dataset_file = dataset_file_map[dataset_name]
    
    return os.path.join(base_path, model_path, dataset_file)


def load_sc_samples_from_json(json_path):
    """
    JSON 형식의 SC 샘플 파일을 로드하고 jsonl 형식으로 변환
    JSON 파일에 저장된 실제 토큰 수(input_tokens, Tokens_X)를 사용
    
    Args:
        json_path: JSON 파일 경로
    
    Returns:
        jsonl 형식과 동일한 리스트 (각 항목은 dict)
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # JSON 배열을 jsonl 형식으로 변환
    result = []
    for item in data:
        # Reasoning Paths에서 completion 추출
        completion = []
        output_tokens_list = []  # 각 CoT의 output 토큰 수 저장
        if "Reasoning Paths" in item:
            for idx, path in enumerate(item["Reasoning Paths"]):
                # 각 경로에서 CoT 텍스트 추출 (CoT_0, CoT_1, ... 형식)
                cot_key = f"CoT_{idx}"
                if cot_key in path:
                    completion.append(path[cot_key])
                else:
                    # CoT_0 형식이 없으면 다른 키 확인
                    for key in path.keys():
                        if key.startswith("CoT_"):
                            completion.append(path[key])
                            break
                    # CoT가 없으면 Final Answer 사용
                    if not completion:
                        final_answer_key = f"Final Answer_{idx}"
                        if final_answer_key in path:
                            completion.append(path[final_answer_key])
                
                # Tokens_X 값 추출 (실제 모델이 생성한 토큰 수)
                tokens_key = f"Tokens_{idx}"
                if tokens_key in path:
                    try:
                        output_tokens_list.append(int(path[tokens_key]))
                    except:
                        output_tokens_list.append(0)
                else:
                    output_tokens_list.append(0)
        
        # jsonl 형식으로 변환
        converted_item = {
            "input": item.get("question", item.get("input", "")),
            "answer": item.get("answer", ""),
            "completion": completion
        }
        
        # 실제 토큰 수 저장 (JSON 파일에 저장된 값 사용)
        if "input_tokens" in item:
            converted_item["input_tokens"] = item["input_tokens"]
        if output_tokens_list:
            converted_item["output_tokens"] = output_tokens_list
        
        # 추가 필드 유지
        if "id" in item:
            converted_item["id"] = item["id"]
        if "solution" in item:
            converted_item["solution"] = item["solution"]
        
        result.append(converted_item)
    
    return result

# ============================================================
# DynSC Helper Functions
# ============================================================

def resolve_split_root(model_id: str) -> str:
    """모델 ID에 해당하는 split root 경로 찾기"""
    base = "/home/woong/ACL/Ours/0116"
    mid = model_id.lower()

    candidates: List[str] = []
    if "qwen2.5-3b" in mid or "qwen2.5_3b" in mid:
        candidates += ["qwen2.5_3b", "qwen2.5-3b", "qwen_3b"]
    if "qwen2.5-7b" in mid or "qwen2.5_7b" in mid:
        candidates += ["qwen2.5_7b", "qwen2.5-7b", "qwen_7b"]
    if "gemma-3-4b" in mid or "gemma3_4b" in mid:
        candidates += ["gemma3_4b", "gemma-3-4b", "gemma_4b"]
    if "llama-3.1-8b" in mid or "llama3.1_8b" in mid or "llama3_8b" in mid:
        candidates += ["llama3.1_8b", "llama3_8b", "llama3_8b_instruct", "llama3.1-8b"]

    for c in candidates:
        p = os.path.join(base, c, "results", "dataset_splits")
        if os.path.isdir(p):
            return p

    raise ValueError(f"split_root not found for model_id={model_id}")

def map_dataset_key_for_split(ds_stem: str) -> str:
    """데이터셋 이름을 split 파일 키로 변환"""
    name = ds_stem.lower()
    if name.startswith("gsm8k"):
        return "gsm8k"
    if name.startswith("math500"):
        return "math500"
    if name.startswith("aime_2024") or name.startswith("aime2024"):
        return "aime2024"
    if name.startswith("aime_2025") or name.startswith("aime2025"):
        return "aime2025"
    if name.startswith("gpqa"):
        return "gpqa_diamond"
    if name.startswith("mmlu"):
        return "mmlu_pro"
    return ""

def find_split_files(split_root: str, key: str) -> Tuple[Optional[str], Optional[str]]:
    """Easy/Hard split 파일 찾기"""
    variants = [
        key,
        key.replace("aime2024", "aime_2024").replace("aime2025", "aime_2025"),
        key.replace("aime_2024", "aime2024").replace("aime_2025", "aime2025"),
    ]
    for v in variants:
        easy = os.path.join(split_root, f"{v}_easy.jsonl")
        hard = os.path.join(split_root, f"{v}_hard.jsonl")
        if os.path.exists(easy) and os.path.exists(hard):
            return easy, hard
    return None, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="result/coin_flip_gpt_4_samples_n_50_temp_0.7.jsonl",
                       help="SC 샘플 파일 경로 (jsonl 또는 json 형식)")
    parser.add_argument("--dsc_input_path", type=str, default="result/coin_flip_gpt_4_samples_n_1_temp_0.jsonl",
                       help="DSC용 greedy 샘플 파일 경로")
    parser.add_argument("--eval_path", type=str, default="result/coin_flip_gpt_4_eval_strategy_random_prompt_short_batchsize_8_iter_5_n_1_temp_0.jsonl",
                       help="평가 결과 파일 경로")
    parser.add_argument("--model_name", type=str, default=None,
                       help="모델 이름 (gemma3_4b, qwen2.5_7b, qwen2.5_3b, llama3.1_8b). 지정시 자동 경로 생성")
    parser.add_argument("--dataset_name", type=str, default=None,
                       help="데이터셋 이름 (GSM8K, AIME_2024, AIME_2025, MATH500_test, MMLU-Pro, GPQA-Diamond). model_name과 함께 사용")
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--judge_window_size", type=int, default=24,
                       help="Judge window size for DSC allocation")
    parser.add_argument("--interplot_size", type=int, default=16,
                       help="Interplot size for window allocation")
    parser.add_argument("--target_conf", type=float, default=0.50,
                       help="Target confidence threshold for ActSC (default: 0.50)")
    args = parser.parse_args()

    # 모델명과 데이터셋명이 제공되면 자동으로 경로 생성
    if args.model_name and args.dataset_name:
        input_path = get_sc_sample_path(args.model_name, args.dataset_name)
        print(f"Auto-generated SC sample path: {input_path}")
        # DSC용 경로도 동일한 구조로 생성 (필요시 수정)
        dsc_input_path = input_path  # 또는 별도 경로 생성
        # eval_path도 자동 생성 (배치 평가 결과 파일)
        # temp 값을 정수로 변환하여 파일명 생성
        temp_str = "0"
        eval_path = f"result/{args.dataset_name}_{args.model_name}_eval_strategy_random_prompt_short_batchsize_8_iter_5_n_1_temp_{temp_str}.jsonl"
        print(f"Auto-generated eval path: {eval_path}")
    else:
        input_path = args.input_path
        dsc_input_path = args.dsc_input_path
        eval_path = args.eval_path

    n = args.n
    repeat = args.repeat
    
    # 데이터셋명 추출
    if args.dataset_name:
        dataset = args.dataset_name
    else:
        # 기존 방식: 경로에서 추출
        if "_gpt_" in input_path:
            dataset = input_path.split("_gpt_")[0].split("/")[-1]
        elif input_path.endswith(".json"):
            # JSON 파일명에서 추출
            dataset = os.path.basename(input_path).replace(".json", "")
        else:
            dataset = input_path.split("/")[-1].split("_")[0]
    
    # dataset = ["MATH", "GSM8K"]
    # dataset = dataset[1]
    interplot_size = args.interplot_size
    window_size = 4
    judge_window_size = args.judge_window_size
    
    # gpt_version 추출 (기존 방식 유지, 필요시 수정)
    if "_gpt_" in input_path:
        gpt_version = input_path.split("gpt_")[-1].split("_")[0]
    else:
        gpt_version = "open_model"  # 오픈 모델 사용시

    # 출력 파일명 생성: {dataset}_{model}_n_{n}_judge_window_{judge_window_size}_interplot_{interplot_size}.json
    # 명시적으로 제공된 dataset_name과 model_name 우선 사용
    if args.dataset_name:
        dataset_for_output = args.dataset_name
    else:
        dataset_for_output = dataset if 'dataset' in locals() else "unknown"
    
    if args.model_name:
        model_for_output = args.model_name
    else:
        # eval_path에서 모델명 추출
        eval_filename = os.path.basename(eval_path).replace(".jsonl", "")
        # 형식: {dataset}_{model}_eval_strategy_...
        if "_eval_" in eval_filename:
            parts = eval_filename.split("_eval_")[0].split("_")
            if len(parts) >= 2:
                model_for_output = "_".join(parts[1:]) if len(parts) > 2 else parts[1]
            else:
                model_for_output = "unknown"
        else:
            model_for_output = "unknown"
    
    # 실제 모델의 tokenizer를 사용하여 토큰 수 계산 (한 번만 로드)
    model_path_map = {
        "gemma3_4b": "google/gemma-3-4b-it",
        "qwen2.5_7b": "Qwen/Qwen2.5-7B-Instruct",
        "qwen2.5_3b": "Qwen/Qwen2.5-3B-Instruct",
        "llama3.1_8b": "meta-llama/Llama-3.1-8B-Instruct"
    }
    
    # model_name은 args.model_name 또는 model_for_output에서 가져옴
    tokenizer_model_name = args.model_name if args.model_name else model_for_output
    
    if tokenizer_model_name not in model_path_map:
        # model_for_output이 model_path_map의 키와 다를 수 있으므로, 
        # 부분 매칭 시도 (예: "gemma3_4b" vs "gemma3_4b_n_40_...")
        found = False
        for key in model_path_map.keys():
            if key in tokenizer_model_name or tokenizer_model_name in key:
                tokenizer_model_name = key
                found = True
                break
        
        if not found:
            print(f"[WARN] Unknown model name: {tokenizer_model_name}. Using default Mistral tokenizer.")
            print(f"[WARN] Available models: {list(model_path_map.keys())}")
            tokenizer = transformers.AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        else:
            model_id = model_path_map[tokenizer_model_name]
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
            print(f"[INFO] Using tokenizer for model: {model_id}")
    else:
        model_id = model_path_map[tokenizer_model_name]
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        print(f"[INFO] Using tokenizer for model: {model_id}")

    # 데이터 로드 (한 번만 수행)
    # JSON 형식인지 jsonl 형식인지 확인
    if input_path.endswith(".json"):
        data_list = load_sc_samples_from_json(input_path)
        # jsonl 형식으로 변환 (문자열 리스트)
        data = [json.dumps(item) + "\n" for item in data_list]
    else:
        with open(input_path, "r") as f:
            data = f.readlines()
            f.close()

    # DSC용 데이터도 동일하게 처리
    if dsc_input_path.endswith(".json"):
        greedy_data_list = load_sc_samples_from_json(dsc_input_path)
        greedy_data = [json.dumps(item) + "\n" for item in greedy_data_list]
    else:
        with open(dsc_input_path, "r") as f:
            greedy_data = f.readlines()
            f.close()

    all_dict = {}
    eval_result = []
    pre = []
    pre_answer = []
    pre_batch = []
    pre_tokens = []
    greedy_pre = []
    greedy_pre_answer = []
    greedy_pre_batch = []
    greedy_pre_tokens = []
    count = 0
    unmatch_count = 0
    greedy_unmatch_count = 0

    if dataset == "MATH" or dataset == "MATH500_test":
        prompt = get_MATH_prompt()
    elif dataset == "GSM8K":
        prompt = get_GSM8K_prompt()
    elif dataset == "coin_flip":
        prompt = get_coin_flip_prompt()
    elif dataset == "last_letter":
        prompt = get_last_letter_prompt()
    elif dataset == "common":
        prompt = get_common_prompt()
    elif dataset in ["AIME_2024", "AIME_2025"]:
        # AIME는 MATH와 유사한 형식
        prompt = get_MATH_prompt()
    elif dataset in ["MMLU-Pro", "GPQA-Diamond"]:
        # MMLU-Pro와 GPQA-Diamond는 common과 유사한 형식
        prompt = get_common_prompt()
    else:
        prompt = get_strategy_prompt()

    for line in data:
        tem = json.loads(line)
        pre.append(tem)
        if dataset == "MATH" or dataset == "MATH500_test":
            pre_answer.append(tem['answer'])
        elif dataset == "GSM8K":
            pre_answer.append(str(tem['answer']))
        elif dataset in ["AIME_2024", "AIME_2025"]:
            pre_answer.append(str(tem['answer']))
        elif dataset in ["MMLU-Pro", "GPQA-Diamond"]:
            pre_answer.append(tem['answer'])
        else:
            pre_answer.append(tem.get('answer', tem.get('output', '')))

        shu_batch = tem['completion']
        if "input_tokens" in tem:
            cur_input_token = tem["input_tokens"]
        else:
            cur_input_token = cal_tokens(prompt + tem["input"], tokenizer)
        
        if "output_tokens" in tem and len(tem["output_tokens"]) == len(shu_batch):
            cur_output_token = deepcopy(tem["output_tokens"])
        else:
            cur_output_token = [cal_tokens(sub, tokenizer) for sub in shu_batch]
        pre_tokens.append({"input": cur_input_token, "output": deepcopy(cur_output_token)})

        if dataset == "MATH" or dataset == "MATH500_test":
            predict_batch = [extract_math_answer(item) for item in shu_batch]
        elif dataset == "GSM8K":
            predict_batch = [extract_gsm8k_answer(item) for item in shu_batch]
        elif dataset in ["AIME_2024", "AIME_2025"]:
            predict_batch = [extract_math_answer(item) for item in shu_batch]
        elif dataset == "coin_flip":
            predict_batch = [extract_coin_flip_answer(item) for item in shu_batch]
        elif dataset == "last_letter":
            predict_batch = [extract_last_letter_answer(item) for item in shu_batch]
        elif dataset == "common":
            predict_batch = [extract_common_answer(item) for item in shu_batch]
        elif dataset == "GPQA-Diamond":
            predict_batch = [extract_answer_gpqa(item) or "" for item in shu_batch]
        elif dataset == "MMLU-Pro":
            predict_batch = [extract_answer_mmlu_pro(item) or "" for item in shu_batch]
        else:
            predict_batch = [extract_strategy_answer(item) for item in shu_batch]
        for item in predict_batch:
            if item == '':
                unmatch_count += 1
        pre_batch.append(deepcopy(predict_batch))

    print("unmatch count={}".format(unmatch_count))

    for line in greedy_data:
        tem = json.loads(line)
        greedy_pre.append(tem)
        if dataset == "MATH" or dataset == "MATH500_test":
            greedy_pre_answer.append(tem['answer'])
        elif dataset == "GSM8K":
            greedy_pre_answer.append(str(tem['answer']))
        elif dataset in ["AIME_2024", "AIME_2025"]:
            greedy_pre_answer.append(str(tem['answer']))
        elif dataset in ["MMLU-Pro", "GPQA-Diamond"]:
            greedy_pre_answer.append(tem['answer'])
        else:
            greedy_pre_answer.append(tem.get('answer', tem.get('output', '')))

        shu_batch = tem['completion']
        if "input_tokens" in tem:
            cur_input_token = tem["input_tokens"]
        else:
            cur_input_token = cal_tokens(prompt + tem["input"], tokenizer)
        
        if "output_tokens" in tem and len(tem["output_tokens"]) == len(shu_batch):
            cur_output_token = deepcopy(tem["output_tokens"])
        else:
            cur_output_token = [cal_tokens(sub, tokenizer) for sub in shu_batch]
        greedy_pre_tokens.append({"input": cur_input_token, "output": deepcopy(cur_output_token)})

        if dataset == "MATH" or dataset == "MATH500_test":
            predict_batch = [extract_math_answer(item) for item in shu_batch]
        elif dataset == "GSM8K":
            predict_batch = [extract_gsm8k_answer(item) for item in shu_batch]
        elif dataset in ["AIME_2024", "AIME_2025"]:
            predict_batch = [extract_math_answer(item) for item in shu_batch]
        elif dataset == "coin_flip":
            predict_batch = [extract_coin_flip_answer(item) for item in shu_batch]
        elif dataset == "last_letter":
            predict_batch = [extract_last_letter_answer(item) for item in shu_batch]
        elif dataset == "common":
            predict_batch = [extract_common_answer(item) for item in shu_batch]
        elif dataset == "GPQA-Diamond":
            predict_batch = [extract_answer_gpqa(item) or "" for item in shu_batch]
        elif dataset == "MMLU-Pro":
            predict_batch = [extract_answer_mmlu_pro(item) or "" for item in shu_batch]
        else:
            predict_batch = [extract_strategy_answer(item) for item in shu_batch]

        for item in predict_batch:
            if item == '':
                greedy_unmatch_count += 1
        greedy_pre_batch.append(deepcopy(predict_batch))

    print("greedy unmatch count={}".format(greedy_unmatch_count))

    with open(eval_path, "r") as f:
        for line in f:
            eval_result.append(json.loads(line))
        f.close()
    
    # 난이도 분류(prepare) 토큰/시간 정보 추출
    prepare_input_tokens_total = 0
    prepare_output_tokens_total = 0
    prepare_time_total = 0.0
    ranking_time_total = 0.0  # Step 1: 랭킹 생성 시간
    split_time_total = 0.0   # Step 2: Easy/Hard 분할 시간
    prepare_count = 0
    step2_input_tokens_total = 0
    step2_output_tokens_total = 0
    prepare_info = None
    if dataset != "MATH":
        if len(eval_result) > 0:
            last_item = eval_result[-1]
            if "prepare_info" in last_item:
                prepare_info = last_item["prepare_info"]
                prepare_input_tokens_total = prepare_info.get("prepare_input_tokens", 0)
                prepare_output_tokens_total = prepare_info.get("prepare_output_tokens", 0)
                ranking_time_total = prepare_info.get("ranking_time", prepare_info.get("prepare_time", 0.0))
                prepare_time_total = ranking_time_total  # 초기값은 랭킹 시간만
                prepare_count = 1
        eval_result = eval_result[-1] if len(eval_result) > 0 else eval_result[0] if len(eval_result) > 0 else {}
    else:
        total_length = len(eval_result)
        eval_result_math = {}
        for i in range(len(eval_result)):
            item = eval_result[i]
            if "prepare_info" in item and i == len(eval_result) - 1:
                prepare_info = item["prepare_info"]
                prepare_input_tokens_total = prepare_info.get("prepare_input_tokens", 0)
                prepare_output_tokens_total = prepare_info.get("prepare_output_tokens", 0)
                ranking_time_total = prepare_info.get("ranking_time", prepare_info.get("prepare_time", 0.0))
                prepare_time_total = ranking_time_total  # 초기값은 랭킹 시간만
                prepare_count = 1
            if item["subject"] not in eval_result_math.keys():
                eval_result_math[item["subject"]] = {"questions":[], "eval":[], "pre_batch":[], "index":[]}
            eval_result_math[item["subject"]]["questions"].append(item["problem"])
            eval_result_math[item["subject"]]["eval"].append(item["eval"])
            eval_result_math[item["subject"]]["pre_batch"].append(deepcopy(pre_batch[i]))
            eval_result_math[item["subject"]]["index"].append(i)
        eval_result = eval_result_math

    if dataset != "MATH":
        pre_batch, eval_result, pre, pre_answer, pre_tokens, greedy_pre_tokens, greedy_pre_batch = clean_data(pre_batch,
                                                                                                              eval_result,
                                                                                                              pre,
                                                                                                              pre_answer,
                                                                                                              pre_tokens,
                                                                                                              greedy_pre_tokens,
                                                                                                              greedy_pre_batch,
                                                                                                              n)
    num_problems = len(pre_batch) if dataset != "MATH" else 0
    
    if num_problems > 0:
        prepare_input_tokens_total = prepare_input_tokens_total / num_problems
        prepare_output_tokens_total = prepare_output_tokens_total / num_problems
    
    if dataset != "MATH":
        # Easy/Hard 분할 시간 측정 (Step 2: Problem Partition)
        split_easy_hard_start = time.time()
        _, _, _, _, _, step2_input_tokens_once, step2_output_tokens_once = split_easy_hard(eval_result, pre_batch, judge_window_size, window_size, n, pre_tokens)
        split_easy_hard_end = time.time()
        split_time_total = split_easy_hard_end - split_easy_hard_start
        # prepare 시간에 Easy/Hard 분할 시간 추가
        prepare_time_total += split_time_total
        step2_input_tokens_total = step2_input_tokens_once / num_problems if num_problems > 0 else 0
        step2_output_tokens_total = step2_output_tokens_once / num_problems if num_problems > 0 else 0
        print(f"[Prepare Step 2] Easy/Hard split time: {split_time_total:.2f}s")
        # prepare_info 업데이트 (각각의 시간과 총 시간 저장)
        if prepare_info is not None:
            prepare_info["split_time"] = split_time_total
            prepare_info["prepare_time"] = prepare_time_total  # 총 시간 (ranking_time + split_time)

    # target_conf 리스트: 0.50, 0.55, 0.60, 0.65
    target_conf_list = [0.50]
    
    # 각 target_conf 값마다 처리
    for target_conf in target_conf_list:
        # 각 target_conf 값마다 폴더 생성
        output_dir = f"cmp/target_conf_{target_conf:.2f}"
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, "{}_{}_n_{}_judge_window_{}_interplot_{}.json".format(
            dataset_for_output, model_for_output, n, judge_window_size, interplot_size
        ))
        print(f"\n{'='*60}")
        print(f"Processing target_conf = {target_conf:.2f}")
        print(f"Output path: {output_path}")
        print(f"{'='*60}")
        
        if os.path.exists(output_path):
            with open(output_path, "r") as f:
                result = json.load(f)
                f.close()
            print(f"Result already exists for target_conf={target_conf:.2f}")
            print(result)
            continue
        
        # target_conf별 계산 시작
        # (데이터 로드와 tokenizer는 루프 밖에서 한 번만 수행)
        
        # Repeat 루프 제거: 1회만 실행하므로 리스트가 아닌 단일 값으로 저장
        sc_accuracy = 0.0
        esc_accuracy = 0.0
        asc_accuracy = 0.0
        dsc_hard_accuracy = 0.0
        dsc_accuracy = 0.0
        dsc_greedy_accuracy = 0.0
        dsc_window_accuracy = 0.0
        dsc_pre_accuracy = 0.0
        
        # Easy/Hard별 정확도 추적
        dsc_hard_easy_accuracy = 0.0
        dsc_hard_hard_accuracy = 0.0
        dsc_easy_accuracy = 0.0
        dsc_hard_accuracy_detail = 0.0
        dsc_greedy_easy_accuracy = 0.0
        dsc_greedy_hard_accuracy = 0.0
        dsc_window_easy_accuracy = 0.0
        dsc_window_hard_accuracy = 0.0
        dsc_pre_easy_accuracy = 0.0
        dsc_pre_hard_accuracy = 0.0
        
        # Easy/Hard별 문제 수 추적
        dsc_hard_easy_count = 0
        dsc_hard_hard_count = 0
        dsc_easy_count = 0
        dsc_hard_count = 0
        dsc_greedy_easy_count = 0
        dsc_greedy_hard_count = 0
        dsc_window_easy_count = 0
        dsc_window_hard_count = 0
        dsc_pre_easy_count = 0
        dsc_pre_hard_count = 0

        tokens_count = {"sc": {"input": 0.0, "output": 0.0, "total": 0.0}, 
                        "esc": {"input": 0.0, "output": 0.0, "total": 0.0}, 
                        "asc": {"input": 0.0, "output": 0.0, "total": 0.0},
                        "dsc_hard": {"input": 0.0, "output": 0.0, "total": 0.0}, 
                        "dsc": {"input": 0.0, "output": 0.0, "total": 0.0},
                        "dsc_greedy": {"input": 0.0, "output": 0.0, "total": 0.0}, 
                        "dsc_window": {"input": 0.0, "output": 0.0, "total": 0.0},  
                        "dsc_pre": {"input": 0.0, "output": 0.0, "total": 0.0},
                        "ActSC": {"input": 0.0, "output": 0.0, "total": 0.0}}
        
        sample_count = {"sc": 0.0, "esc": 0.0, "asc": 0.0, "dsc_hard": 0.0, "dsc": 0.0, 
                        "dsc_greedy": 0.0, "dsc_window": 0.0, "dsc_pre": 0.0, "ActSC": 0.0}
        
        # (데이터는 루프 밖에서 이미 로드됨)
        
        # DynSC 평가 (repeat 루프 밖에서 한 번만 수행, SC 샘플 재사용)
        ActSC_acc = 0.0
        ActSC_sample_num = 0.0
        ActSC_input_token = 0.0
        ActSC_output_token = 0.0
        ActSC_easy_accuracy_list = []
        ActSC_hard_accuracy_list = []
        ActSC_easy_count_list = []
        ActSC_hard_count_list = []
        
        if dataset != "MATH":
            # 모델명에서 split root 찾기
            try:
                model_id_for_split = model_for_output if 'model_for_output' in locals() else args.model_name
                if model_id_for_split:
                    # model_name을 실제 모델 경로로 변환
                    model_path_map = {
                        "gemma3_4b": "google/gemma-3-4b-it",
                        "qwen2.5_7b": "Qwen/Qwen2.5-7B-Instruct",
                        "qwen2.5_3b": "Qwen/Qwen2.5-3B-Instruct",
                        "llama3.1_8b": "meta-llama/Llama-3.1-8B-Instruct"
                    }
                    # model_id_for_split을 org/model 형식으로 변환
                    if model_id_for_split in model_path_map:
                        model_hf_id = model_path_map[model_id_for_split]
                        # org 추출
                        if "google" in model_hf_id:
                            org = "google"
                        elif "Qwen" in model_hf_id:
                            org = "Qwen"
                        elif "meta-llama" in model_hf_id:
                            org = "meta-llama"
                        else:
                            org = "unknown"
                        model_id = f"{org}/{model_hf_id}"
                    else:
                        model_id = model_id_for_split
                    
                    split_root = resolve_split_root(model_id)
                    dataset_key = map_dataset_key_for_split(dataset)
                    if dataset_key:
                        easy_path, hard_path = find_split_files(split_root, dataset_key)
                        if easy_path and hard_path:
                            diff_map = load_easy_hard_split(easy_path, hard_path)
                            # extract_answer_func 결정
                            if dataset in ["MATH500_test", "AIME_2024", "AIME_2025"]:
                                extract_answer_func = extract_math_answer
                            elif dataset == "GSM8K":
                                extract_answer_func = extract_gsm8k_answer
                            elif dataset == "GPQA-Diamond":
                                extract_answer_func = lambda x: extract_answer_gpqa(x) or ""
                            elif dataset == "MMLU-Pro":
                                extract_answer_func = lambda x: extract_answer_mmlu_pro(x) or ""
                            else:
                                extract_answer_func = extract_math_answer
                            
                            # SC 데이터 준비 (pre 리스트를 Dict 리스트로 변환)
                            sc_data = []
                            for i, item in enumerate(pre):
                                sc_item = {
                                    "id": item.get("id"),
                                    "question": item.get("input"),
                                    "problem": item.get("input"),
                                    "answer": pre_answer[i],
                                    "Reasoning Paths": []
                                }
                                # Input tokens 추가 (JSON 파일에서 추출한 실제 토큰 수)
                                if i < len(pre_tokens):
                                    sc_item["input_tokens"] = pre_tokens[i]["input"]
                                
                                # Reasoning Paths 구성
                                reasoning_paths = item.get('completion', [])
                                for j, path_text in enumerate(reasoning_paths):
                                    path_dict = {}
                                    # CoT 추출
                                    path_dict[f"CoT_{j}"] = path_text
                                    # Final Answer 추출
                                    if dataset in ["MATH500_test", "AIME_2024", "AIME_2025"]:
                                        answer = extract_math_answer(path_text)
                                    elif dataset == "GSM8K":
                                        answer = extract_gsm8k_answer(path_text)
                                    elif dataset == "GPQA-Diamond":
                                        answer = extract_answer_gpqa(path_text) or ""
                                    elif dataset == "MMLU-Pro":
                                        answer = extract_answer_mmlu_pro(path_text) or ""
                                    else:
                                        answer = extract_math_answer(path_text)
                                    path_dict[f"Final Answer_{j}"] = answer
                                    # Tokens 추출
                                    if i < len(pre_tokens) and j < len(pre_tokens[i]["output"]):
                                        path_dict[f"Tokens_{j}"] = pre_tokens[i]["output"][j]
                                    else:
                                        path_dict[f"Tokens_{j}"] = 0
                                    sc_item["Reasoning Paths"].append(path_dict)
                                sc_data.append(sc_item)
                            
                            # ActSC 평가
                            ActSC_acc, ActSC_sample_num, ActSC_input_token, ActSC_output_token, ActSC_easy_acc, ActSC_hard_acc, ActSC_easy_count, ActSC_hard_count = eval_ActSC(
                                sc_data, diff_map, target_conf=target_conf, 
                                extract_answer_func=extract_answer_func,
                                tokenizer=tokenizer, prompt=prompt
                            )
                            ActSC_easy_accuracy_list.append(ActSC_easy_acc)
                            ActSC_hard_accuracy_list.append(ActSC_hard_acc)
                            ActSC_easy_count_list.append(ActSC_easy_count)
                            ActSC_hard_count_list.append(ActSC_hard_count)
                            print(f"DynSC: Samples={ActSC_sample_num:.2f}, Inference(Input={ActSC_input_token:.2f}, Output={ActSC_output_token:.2f}), Acc={ActSC_acc:.4f} (Easy: {ActSC_easy_acc:.4f}, Hard: {ActSC_hard_acc:.4f})")
                        else:
                            print(f"[WARN] Easy/Hard split files not found for dataset: {dataset_key}")
                    else:
                        print(f"[WARN] Dataset key mapping not found for: {dataset}")
                else:
                    print(f"[WARN] Model name not available for DynSC evaluation")
            except Exception as e:
                print(f"[WARN] DynSC evaluation failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Repeat 루프 제거: 1회만 실행
        if dataset != "MATH":
            index, eval, question, pre_batch, allocate_list, _, _ = split_easy_hard(eval_result, pre_batch, judge_window_size, window_size, n, pre_tokens)
            index_greedy, eval_greedy, question_greedy, pre_batch_greedy, allocate_list_greedy = split_easy_hard_greedy(eval_result, pre_batch, judge_window_size, window_size, n)
            window_list_actual = allocate_window(eval_result, pre_batch, n, interplot_size=interplot_size, window_size=window_size)
            
            # 비-MATH 데이터셋 계산 루프
            for sc_num in [n]:
                sc_result_list = []
                esc_result_list = []
                asc_result_list = []
                dsc_hard_result_list = []
                dsc_result_list = []
                dsc_greedy_result_list = []
                dsc_window_result_list = []
                dsc_pre_result_list = []
                
                # Easy/Hard별 정확도 추적
                dsc_hard_easy_list = []
                dsc_hard_hard_list = []
                dsc_easy_list = []
                dsc_hard_list = []
                dsc_greedy_easy_list = []
                dsc_greedy_hard_list = []
                dsc_window_easy_list = []
                dsc_window_hard_list = []
                dsc_pre_easy_list = []
                dsc_pre_hard_list = []

                # 토큰 수와 샘플 수를 직접 누적 (리스트가 아닌 합계)
                sub_tokens_count = {"sc": {"input": 0.0, "output": 0.0, "total": 0.0}, 
                                    "esc": {"input": 0.0, "output": 0.0, "total": 0.0}, 
                                    "asc": {"input": 0.0, "output": 0.0, "total": 0.0},
                                    "dsc_hard": {"input": 0.0, "output": 0.0, "total": 0.0}, 
                                    "dsc": {"input": 0.0, "output": 0.0, "total": 0.0},
                                    "dsc_greedy": {"input": 0.0, "output": 0.0, "total": 0.0}, 
                                    "dsc_window": {"input": 0.0, "output": 0.0, "total": 0.0}, 
                                    "dsc_pre": {"input": 0.0, "output": 0.0, "total": 0.0}}

                sub_sample_count = {"sc": 0.0, "esc": 0.0, "asc": 0.0, "dsc_hard": 0.0, "dsc": 0.0, 
                                    "dsc_greedy": 0.0, "dsc_window": 0.0, "dsc_pre": 0.0}
                
                num_problems_processed = 0

                num_list = []
                true_window_list = []
                right_nums = 0
                n_nums = 0
                count = 0

                for i in range(len(pre)):
                    flag = False
                    pre_dict = {}
                    esc_pre_dict = {}
                    asc_pre_dict = {}
                    dsc_hard_pre_dict = {}
                    dsc_pre_dict = {}
                    dsc_greedy_pre_dict = {}
                    dsc_window_pre_dict = {}
                    dsc_pre_pre_dict = {}

                    tem = pre[i]
                    question = tem["input"]
                    answer = pre_answer[i]
                    tokens_batch = pre_tokens[i]
                    greedy_tokens_batch = greedy_pre_tokens[i]
                    sub_allocate = allocate_list[i]
                    sub_allocate_greedy = allocate_list_greedy[i]

                    sub_window_actual = window_list_actual[i]

                    shu_batch = tem['completion']

                    predict_batch = pre_batch[i]
                    greedy_predict_batch = greedy_pre_batch[i]
                    if len(predict_batch) < sc_num:
                        count += 1
                        continue

                    predict_batch, tokens_batch = random_elements(predict_batch, tokens_batch, sc_num)
                    esc_batch, esc_token_batch = ESC(predict_batch, tokens_batch, window_size=5)
                    asc_batch, asc_token_batch = ASC(predict_batch, tokens_batch)
                    dsc_hard_batch, dsc_hard_token_batch = DSC_hard(predict_batch, tokens_batch, window_size=window_size)
                    dsc_batch, dsc_token_batch = DSC(predict_batch, greedy_predict_batch, tokens_batch, greedy_tokens_batch, sub_allocate, window_size=window_size)
                    dsc_greedy_batch, dsc_greedy_token_batch = DSC(predict_batch, greedy_predict_batch, tokens_batch, greedy_tokens_batch, sub_allocate_greedy, window_size=window_size)
                    dsc_window_batch, dsc_window_token_batch = DSC_window(predict_batch, greedy_predict_batch, tokens_batch, greedy_tokens_batch, sub_allocate_greedy, sub_window_actual, extend_window_size=window_size)
                    dsc_pre_batch, dsc_pre_token_batch = DSC_pre(predict_batch, greedy_predict_batch, tokens_batch, greedy_tokens_batch, sub_allocate_greedy, sub_window_actual, extend_window_size=window_size)

                    sc_input_token, sc_output_token = tokens_batch["input"], sum(tokens_batch["output"])
                    esc_input_token, esc_output_token = tokens_batch["input"]*len(esc_token_batch["output"]) // window_size, sum(esc_token_batch["output"])
                    asc_input_token, asc_output_token = tokens_batch["input"]*len(asc_token_batch["output"]), sum(asc_token_batch["output"])
                    dsc_hard_input_token, dsc_hard_output_token = tokens_batch["input"] * len(dsc_hard_token_batch["output"]) // window_size, sum(dsc_hard_token_batch["output"])
                    dsc_input_token, dsc_output_token = tokens_batch["input"] * max(1, len(dsc_token_batch["output"]) // window_size), sum(dsc_token_batch["output"])
                    dsc_greedy_input_token, dsc_greedy_output_token = tokens_batch["input"] * max(1, len(dsc_greedy_token_batch["output"]) // window_size), sum(dsc_greedy_token_batch["output"])
                    dsc_window_input_token, dsc_window_output_token = tokens_batch["input"] * max(1, len(dsc_window_token_batch["output"]) // sub_window_actual), sum(dsc_window_token_batch["output"])
                    dsc_pre_input_token, dsc_pre_output_token = tokens_batch["input"] * max(1,len(dsc_pre_token_batch["output"]) // sub_window_actual), sum(dsc_pre_token_batch["output"])

                    # 토큰 수 누적
                    sub_tokens_count["sc"]["input"] += sc_input_token
                    sub_tokens_count["sc"]["output"] += sc_output_token
                    sub_tokens_count["sc"]["total"] += sc_input_token + sc_output_token
                    sub_tokens_count["esc"]["input"] += esc_input_token
                    sub_tokens_count["esc"]["output"] += esc_output_token
                    sub_tokens_count["esc"]["total"] += esc_input_token + esc_output_token
                    sub_tokens_count["asc"]["input"] += asc_input_token
                    sub_tokens_count["asc"]["output"] += asc_output_token
                    sub_tokens_count["asc"]["total"] += asc_input_token + asc_output_token
                    sub_tokens_count["dsc_hard"]["input"] += dsc_hard_input_token
                    sub_tokens_count["dsc_hard"]["output"] += dsc_hard_output_token
                    sub_tokens_count["dsc_hard"]["total"] += dsc_hard_input_token + dsc_hard_output_token
                    sub_tokens_count["dsc"]["input"] += dsc_input_token
                    sub_tokens_count["dsc"]["output"] += dsc_output_token
                    sub_tokens_count["dsc"]["total"] += dsc_input_token + dsc_output_token
                    sub_tokens_count["dsc_greedy"]["input"] += dsc_greedy_input_token
                    sub_tokens_count["dsc_greedy"]["output"] += dsc_greedy_output_token
                    sub_tokens_count["dsc_greedy"]["total"] += dsc_greedy_input_token + dsc_greedy_output_token
                    sub_tokens_count["dsc_window"]["input"] += dsc_window_input_token
                    sub_tokens_count["dsc_window"]["output"] += dsc_window_output_token
                    sub_tokens_count["dsc_window"]["total"] += dsc_window_input_token + dsc_window_output_token
                    sub_tokens_count["dsc_pre"]["input"] += dsc_pre_input_token
                    sub_tokens_count["dsc_pre"]["output"] += dsc_pre_output_token
                    sub_tokens_count["dsc_pre"]["total"] += dsc_pre_input_token + dsc_pre_output_token

                    # 각 방법에서 사용된 샘플 수 누적
                    sub_sample_count["sc"] += len(predict_batch)
                    sub_sample_count["esc"] += len(esc_batch)
                    sub_sample_count["asc"] += len(asc_batch)
                    sub_sample_count["dsc_hard"] += len(dsc_hard_batch)
                    sub_sample_count["dsc"] += len(dsc_batch)
                    sub_sample_count["dsc_greedy"] += len(dsc_greedy_batch)
                    sub_sample_count["dsc_window"] += len(dsc_window_batch)
                    sub_sample_count["dsc_pre"] += len(dsc_pre_batch)
                    
                    num_problems_processed += 1

                    for item in predict_batch:
                        if item in pre_dict.keys():
                            pre_dict[item] += 1
                        else:
                            pre_dict[item] = 1

                    for item in esc_batch:
                        if item in esc_pre_dict.keys():
                            esc_pre_dict[item] += 1
                        else:
                            esc_pre_dict[item] = 1

                    for item in asc_batch:
                        if item in asc_pre_dict.keys():
                            asc_pre_dict[item] += 1
                        else:
                            asc_pre_dict[item] = 1

                    for item in dsc_hard_batch:
                        if item in dsc_hard_pre_dict.keys():
                            dsc_hard_pre_dict[item] += 1
                        else:
                            dsc_hard_pre_dict[item] = 1

                    for item in dsc_batch:
                        if item in dsc_pre_dict.keys():
                            dsc_pre_dict[item] += 1
                        else:
                            dsc_pre_dict[item] = 1

                    for item in dsc_greedy_batch:
                        if item in dsc_greedy_pre_dict.keys():
                            dsc_greedy_pre_dict[item] += 1
                        else:
                            dsc_greedy_pre_dict[item] = 1

                    for item in dsc_window_batch:
                        if item in dsc_window_pre_dict.keys():
                            dsc_window_pre_dict[item] += 1
                        else:
                            dsc_window_pre_dict[item] = 1

                    for item in dsc_pre_batch:
                        if item in dsc_pre_pre_dict.keys():
                            dsc_pre_pre_dict[item] += 1
                        else:
                            dsc_pre_pre_dict[item] = 1

                    # SC를 상한선으로 설정: SC는 모든 샘플(n개)을 사용하므로 최고 성능
                    sc_pre = max(pre_dict, key=pre_dict.get)
                    sc_correct = sc_pre == answer
                    sc_result_list.append(sc_correct)

                    # 다른 방법론들은 SC의 부분집합을 사용하므로 각 문제별로 독립적으로 계산
                    esc_pre = max(esc_pre_dict, key=esc_pre_dict.get)
                    esc_correct = esc_pre == answer
                    esc_result_list.append(esc_correct)

                    asc_pre = max(asc_pre_dict, key=asc_pre_dict.get)
                    asc_correct = asc_pre == answer
                    asc_result_list.append(asc_correct)

                    dsc_hard_pre = max(dsc_hard_pre_dict, key=dsc_hard_pre_dict.get)
                    dsc_hard_correct = dsc_hard_pre == answer
                    dsc_hard_result_list.append(dsc_hard_correct)
                    # Easy/Hard 구분 (DSC_hard는 allocate_list 사용하지 않으므로 window_size로 구분)
                    if len(dsc_hard_batch) <= window_size:
                        dsc_hard_easy_list.append(dsc_hard_correct)
                    else:
                        dsc_hard_hard_list.append(dsc_hard_correct)

                    dsc_pre = max(dsc_pre_dict, key=dsc_pre_dict.get)
                    dsc_correct = dsc_pre == answer
                    dsc_result_list.append(dsc_correct)
                    # Easy/Hard 구분 (allocate_list[i] == 1: easy, == n: hard)
                    if sub_allocate == 1:
                        dsc_easy_list.append(dsc_correct)
                    elif sub_allocate == n:
                        dsc_hard_list.append(dsc_correct)

                    dsc_greedy_pre = max(dsc_greedy_pre_dict, key=dsc_greedy_pre_dict.get)
                    dsc_greedy_correct = dsc_greedy_pre == answer
                    dsc_greedy_result_list.append(dsc_greedy_correct)
                    # Easy/Hard 구분
                    if sub_allocate_greedy == 1:
                        dsc_greedy_easy_list.append(dsc_greedy_correct)
                    elif sub_allocate_greedy == -1:
                        dsc_greedy_hard_list.append(dsc_greedy_correct)

                    dsc_window_pre = max(dsc_window_pre_dict, key=dsc_window_pre_dict.get)
                    dsc_window_correct = dsc_window_pre == answer
                    dsc_window_result_list.append(dsc_window_correct)
                    # Easy/Hard 구분
                    if sub_allocate_greedy == 1:
                        dsc_window_easy_list.append(dsc_window_correct)
                    elif sub_allocate_greedy == -1:
                        dsc_window_hard_list.append(dsc_window_correct)

                    dsc_pre_pre = max(dsc_pre_pre_dict, key=dsc_pre_pre_dict.get)
                    dsc_pre_correct = dsc_pre_pre == answer
                    dsc_pre_result_list.append(dsc_pre_correct)
                    # Easy/Hard 구분
                    if sub_allocate_greedy == 1:
                        dsc_pre_easy_list.append(dsc_pre_correct)
                    elif sub_allocate_greedy == -1:
                        dsc_pre_hard_list.append(dsc_pre_correct)

                # 정확도 계산 (평균 계산 제거, 직접 계산)
                if num_problems_processed > 0:
                    sc_accuracy = np.array(sc_result_list).mean()
                    esc_accuracy = np.array(esc_result_list).mean()
                    asc_accuracy = np.array(asc_result_list).mean()
                    dsc_hard_accuracy = np.array(dsc_hard_result_list).mean()
                    dsc_accuracy = np.array(dsc_result_list).mean()
                    dsc_greedy_accuracy = np.array(dsc_greedy_result_list).mean()
                    dsc_window_accuracy = np.array(dsc_window_result_list).mean()
                    dsc_pre_accuracy = np.array(dsc_pre_result_list).mean()
                    
                    # Easy/Hard별 정확도 계산
                    if len(dsc_hard_easy_list) > 0:
                        dsc_hard_easy_accuracy = np.array(dsc_hard_easy_list).mean()
                        dsc_hard_easy_count = len(dsc_hard_easy_list)
                    if len(dsc_hard_hard_list) > 0:
                        dsc_hard_hard_accuracy = np.array(dsc_hard_hard_list).mean()
                        dsc_hard_hard_count = len(dsc_hard_hard_list)
                    
                    if len(dsc_easy_list) > 0:
                        dsc_easy_accuracy = np.array(dsc_easy_list).mean()
                        dsc_easy_count = len(dsc_easy_list)
                    if len(dsc_hard_list) > 0:
                        dsc_hard_accuracy_detail = np.array(dsc_hard_list).mean()
                        dsc_hard_count = len(dsc_hard_list)
                    
                    if len(dsc_greedy_easy_list) > 0:
                        dsc_greedy_easy_accuracy = np.array(dsc_greedy_easy_list).mean()
                        dsc_greedy_easy_count = len(dsc_greedy_easy_list)
                    if len(dsc_greedy_hard_list) > 0:
                        dsc_greedy_hard_accuracy = np.array(dsc_greedy_hard_list).mean()
                        dsc_greedy_hard_count = len(dsc_greedy_hard_list)
                    
                    if len(dsc_window_easy_list) > 0:
                        dsc_window_easy_accuracy = np.array(dsc_window_easy_list).mean()
                        dsc_window_easy_count = len(dsc_window_easy_list)
                    if len(dsc_window_hard_list) > 0:
                        dsc_window_hard_accuracy = np.array(dsc_window_hard_list).mean()
                        dsc_window_hard_count = len(dsc_window_hard_list)
                    
                    if len(dsc_pre_easy_list) > 0:
                        dsc_pre_easy_accuracy = np.array(dsc_pre_easy_list).mean()
                        dsc_pre_easy_count = len(dsc_pre_easy_list)
                    if len(dsc_pre_hard_list) > 0:
                        dsc_pre_hard_accuracy = np.array(dsc_pre_hard_list).mean()
                        dsc_pre_hard_count = len(dsc_pre_hard_list)

                    # 토큰 수와 샘플 수 계산 (문제당 평균)
                    for key in tokens_count.keys():
                        if key in sub_tokens_count:
                            tokens_count[key]["input"] = sub_tokens_count[key]["input"] / num_problems_processed
                            tokens_count[key]["output"] = sub_tokens_count[key]["output"] / num_problems_processed
                            tokens_count[key]["total"] = sub_tokens_count[key]["total"] / num_problems_processed
                    
                    for key in sample_count.keys():
                        if key in sub_sample_count:
                            sample_count[key] = sub_sample_count[key] / num_problems_processed
        else:
            allocate_list = [0 for i in range(total_length)]
            allocate_list_greedy = [0 for i in range(total_length)]
            window_list_actual = [0 for i in range(total_length)]
            # MATH 데이터셋의 경우 Step 2 토큰 수는 계산하지 않음 (pre_tokens를 전달할 수 없음)
            # 또는 0으로 설정
            # step2_input_tokens_total = 0
            # step2_output_tokens_total = 0
            for key in eval_result.keys():
                print(key)
                sub_eval_result = eval_result[key]
                sub_index = sub_eval_result["index"]
                # MATH 데이터셋의 경우 pre_tokens는 전체 pre_tokens에서 subject별로 분리되지 않음
                # 따라서 pre_tokens를 전달할 수 없으므로 None으로 설정
                index, sub_eval, sub_question, sub_pre_batch, sub_allocate_list, _, _ = split_easy_hard(sub_eval_result,
                                                                                                  sub_eval_result[
                                                                                                      "pre_batch"],
                                                                                                  judge_window_size,
                                                                                                  window_size, n, None)

                index_greedy, sub_eval_greedy, sub_question_greedy, sub_pre_batch_greedy, sub_allocate_list_greedy = split_easy_hard_greedy(sub_eval_result,
                                                                                                  sub_eval_result[
                                                                                                      "pre_batch"],
                                                                                                  judge_window_size,
                                                                                                  window_size, n)
                actual_window = allocate_window(sub_eval_result, sub_eval_result["pre_batch"], n,
                                                                              interplot_size=interplot_size)
                for i in range(len(sub_allocate_list)):
                    allocate_list[sub_index[i]] = sub_allocate_list[i]
                    allocate_list_greedy[sub_index[i]] = sub_allocate_list_greedy[i]
                    window_list_actual[sub_index[i]] = actual_window[i]


            for sc_num in [n]:
                sc_result_list = []
                esc_result_list = []
                asc_result_list = []
                dsc_hard_result_list = []
                dsc_result_list = []
                dsc_greedy_result_list = []
                dsc_window_result_list = []
                dsc_pre_result_list = []
                
                # Easy/Hard별 정확도 추적
                dsc_hard_easy_list = []
                dsc_hard_hard_list = []
                dsc_easy_list = []
                dsc_hard_list = []
                dsc_greedy_easy_list = []
                dsc_greedy_hard_list = []
                dsc_window_easy_list = []
                dsc_window_hard_list = []
                dsc_pre_easy_list = []
                dsc_pre_hard_list = []

                # 토큰 수와 샘플 수를 직접 누적 (리스트가 아닌 합계)
                sub_tokens_count = {"sc": {"input": 0.0, "output": 0.0, "total": 0.0}, 
                                    "esc": {"input": 0.0, "output": 0.0, "total": 0.0}, 
                                    "asc": {"input": 0.0, "output": 0.0, "total": 0.0},
                                    "dsc_hard": {"input": 0.0, "output": 0.0, "total": 0.0}, 
                                    "dsc": {"input": 0.0, "output": 0.0, "total": 0.0},
                                    "dsc_greedy": {"input": 0.0, "output": 0.0, "total": 0.0}, 
                                    "dsc_window": {"input": 0.0, "output": 0.0, "total": 0.0}, 
                                    "dsc_pre": {"input": 0.0, "output": 0.0, "total": 0.0}}

                sub_sample_count = {"sc": 0.0, "esc": 0.0, "asc": 0.0, "dsc_hard": 0.0, "dsc": 0.0, 
                                    "dsc_greedy": 0.0, "dsc_window": 0.0, "dsc_pre": 0.0}
                
                num_problems_processed = 0

                num_list = []
                true_window_list = []
                right_nums = 0
                n_nums = 0
                count = 0

                for i in range(len(pre)):
                    flag = False
                    pre_dict = {}
                    esc_pre_dict = {}
                    asc_pre_dict = {}
                    dsc_hard_pre_dict = {}
                    dsc_pre_dict = {}
                    dsc_greedy_pre_dict = {}
                    dsc_window_pre_dict = {}
                    dsc_pre_pre_dict = {}

                    tem = pre[i]
                    question = tem["input"]
                    answer = pre_answer[i]
                    tokens_batch = pre_tokens[i]
                    greedy_tokens_batch = greedy_pre_tokens[i]
                    sub_allocate = allocate_list[i]
                    sub_allocate_greedy = allocate_list_greedy[i]

                    sub_window_actual = window_list_actual[i]

                    # print("sub_window_actual:{}".format(sub_window_actual))

                    shu_batch = tem['completion']

                    predict_batch = pre_batch[i]
                    greedy_predict_batch = greedy_pre_batch[i]
                    # print(len(predict_batch))
                    if len(predict_batch) < sc_num:
                        count += 1
                        continue


                    predict_batch, tokens_batch = random_elements(predict_batch, tokens_batch, sc_num)
                    # predict_batch = random.sample(predict_batch, sc_num)
                    esc_batch, esc_token_batch = ESC(predict_batch, tokens_batch, window_size=5)
                    asc_batch, asc_token_batch = ASC(predict_batch, tokens_batch)
                    dsc_hard_batch, dsc_hard_token_batch = DSC_hard(predict_batch, tokens_batch, window_size=window_size)
                    dsc_batch, dsc_token_batch = DSC(predict_batch, greedy_predict_batch, tokens_batch, greedy_tokens_batch, sub_allocate, window_size=window_size)
                    dsc_greedy_batch, dsc_greedy_token_batch = DSC(predict_batch, greedy_predict_batch, tokens_batch, greedy_tokens_batch, sub_allocate_greedy, window_size=window_size)
                    dsc_window_batch, dsc_window_token_batch = DSC_window(predict_batch, greedy_predict_batch, tokens_batch, greedy_tokens_batch, sub_allocate_greedy, sub_window_actual, extend_window_size=window_size)
                    dsc_pre_batch, dsc_pre_token_batch = DSC_pre(predict_batch, greedy_predict_batch, tokens_batch, greedy_tokens_batch, sub_allocate_greedy, sub_window_actual, extend_window_size=window_size)

                    sc_input_token, sc_output_token = tokens_batch["input"], sum(tokens_batch["output"])
                    esc_input_token, esc_output_token = tokens_batch["input"]*len(esc_token_batch["output"]) // window_size, sum(esc_token_batch["output"])
                    asc_input_token, asc_output_token = tokens_batch["input"]*len(asc_token_batch["output"]), sum(asc_token_batch["output"])
                    dsc_hard_input_token, dsc_hard_output_token = tokens_batch["input"] * len(dsc_hard_token_batch["output"]) // window_size, sum(dsc_hard_token_batch["output"])
                    dsc_input_token, dsc_output_token = tokens_batch["input"] * max(1, len(dsc_token_batch["output"]) // window_size), sum(dsc_token_batch["output"])
                    dsc_greedy_input_token, dsc_greedy_output_token = tokens_batch["input"] * max(1, len(dsc_greedy_token_batch["output"]) // window_size), sum(dsc_greedy_token_batch["output"])
                    dsc_window_input_token, dsc_window_output_token = tokens_batch["input"] * max(1, len(dsc_window_token_batch["output"]) // sub_window_actual), sum(dsc_window_token_batch["output"])
                    dsc_pre_input_token, dsc_pre_output_token = tokens_batch["input"] * max(1,len(dsc_pre_token_batch["output"]) // sub_window_actual), sum(dsc_pre_token_batch["output"])

                    # sc_input_token, sc_output_token = cal_sc_tokens(prompt + question, completion_batch, tokenizer)
                    # esc_input_token, esc_output_token = cal_esc_tokens(prompt + question, esc_completion_batch, tokenizer)
                    # asc_input_token, asc_output_token = cal_asc_tokens(prompt + question, asc_completion_batch, tokenizer)

                    # 토큰 수 누적
                    sub_tokens_count["sc"]["input"] += sc_input_token
                    sub_tokens_count["sc"]["output"] += sc_output_token
                    sub_tokens_count["sc"]["total"] += sc_input_token + sc_output_token
                    sub_tokens_count["esc"]["input"] += esc_input_token
                    sub_tokens_count["esc"]["output"] += esc_output_token
                    sub_tokens_count["esc"]["total"] += esc_input_token + esc_output_token
                    sub_tokens_count["asc"]["input"] += asc_input_token
                    sub_tokens_count["asc"]["output"] += asc_output_token
                    sub_tokens_count["asc"]["total"] += asc_input_token + asc_output_token
                    sub_tokens_count["dsc_hard"]["input"] += dsc_hard_input_token
                    sub_tokens_count["dsc_hard"]["output"] += dsc_hard_output_token
                    sub_tokens_count["dsc_hard"]["total"] += dsc_hard_input_token + dsc_hard_output_token
                    sub_tokens_count["dsc"]["input"] += dsc_input_token
                    sub_tokens_count["dsc"]["output"] += dsc_output_token
                    sub_tokens_count["dsc"]["total"] += dsc_input_token + dsc_output_token
                    sub_tokens_count["dsc_greedy"]["input"] += dsc_greedy_input_token
                    sub_tokens_count["dsc_greedy"]["output"] += dsc_greedy_output_token
                    sub_tokens_count["dsc_greedy"]["total"] += dsc_greedy_input_token + dsc_greedy_output_token
                    sub_tokens_count["dsc_window"]["input"] += dsc_window_input_token
                    sub_tokens_count["dsc_window"]["output"] += dsc_window_output_token
                    sub_tokens_count["dsc_window"]["total"] += dsc_window_input_token + dsc_window_output_token
                    sub_tokens_count["dsc_pre"]["input"] += dsc_pre_input_token
                    sub_tokens_count["dsc_pre"]["output"] += dsc_pre_output_token
                    sub_tokens_count["dsc_pre"]["total"] += dsc_pre_input_token + dsc_pre_output_token

                    # 각 방법에서 사용된 샘플 수 누적
                    sub_sample_count["sc"] += len(predict_batch)
                    sub_sample_count["esc"] += len(esc_batch)
                    sub_sample_count["asc"] += len(asc_batch)
                    sub_sample_count["dsc_hard"] += len(dsc_hard_batch)
                    sub_sample_count["dsc"] += len(dsc_batch)
                    sub_sample_count["dsc_greedy"] += len(dsc_greedy_batch)
                    sub_sample_count["dsc_window"] += len(dsc_window_batch)
                    sub_sample_count["dsc_pre"] += len(dsc_pre_batch)
                    
                    num_problems_processed += 1

                    for item in predict_batch:
                        if item in pre_dict.keys():
                            pre_dict[item] += 1
                        else:
                            pre_dict[item] = 1

                    for item in esc_batch:
                        if item in esc_pre_dict.keys():
                            esc_pre_dict[item] += 1
                        else:
                            esc_pre_dict[item] = 1

                    for item in asc_batch:
                        if item in asc_pre_dict.keys():
                            asc_pre_dict[item] += 1
                        else:
                            asc_pre_dict[item] = 1


                    for item in dsc_hard_batch:
                        if item in dsc_hard_pre_dict.keys():
                            dsc_hard_pre_dict[item] += 1
                        else:
                            dsc_hard_pre_dict[item] = 1

                    for item in dsc_batch:
                        if item in dsc_pre_dict.keys():
                            dsc_pre_dict[item] += 1
                        else:
                            dsc_pre_dict[item] = 1

                    for item in dsc_greedy_batch:
                        if item in dsc_greedy_pre_dict.keys():
                            dsc_greedy_pre_dict[item] += 1
                        else:
                            dsc_greedy_pre_dict[item] = 1

                    for item in dsc_window_batch:
                        if item in dsc_window_pre_dict.keys():
                            dsc_window_pre_dict[item] += 1
                        else:
                            dsc_window_pre_dict[item] = 1

                    for item in dsc_pre_batch:
                        if item in dsc_pre_pre_dict.keys():
                            dsc_pre_pre_dict[item] += 1
                        else:
                            dsc_pre_pre_dict[item] = 1

                    # SC를 상한선으로 설정: SC는 모든 샘플(n개)을 사용하므로 최고 성능
                    sc_pre = max(pre_dict, key=pre_dict.get)
                    sc_correct = sc_pre == answer
                    sc_result_list.append(sc_correct)

                    # 다른 방법론들은 SC의 부분집합을 사용하므로 각 문제별로 독립적으로 계산
                    # (전체 정확도에서 SC가 상한선이 되도록 최종 단계에서 제한)
                    esc_pre = max(esc_pre_dict, key=esc_pre_dict.get)
                    esc_correct = esc_pre == answer
                    esc_result_list.append(esc_correct)

                    asc_pre = max(asc_pre_dict, key=asc_pre_dict.get)
                    asc_correct = asc_pre == answer
                    asc_result_list.append(asc_correct)


                    dsc_hard_pre = max(dsc_hard_pre_dict, key=dsc_hard_pre_dict.get)
                    dsc_hard_correct = dsc_hard_pre == answer
                    dsc_hard_result_list.append(dsc_hard_correct)
                    # Easy/Hard 구분 (DSC_hard는 allocate_list 사용하지 않으므로 window_size로 구분)
                    if len(dsc_hard_batch) <= window_size:
                        dsc_hard_easy_list.append(dsc_hard_correct)
                    else:
                        dsc_hard_hard_list.append(dsc_hard_correct)

                    dsc_pre = max(dsc_pre_dict, key=dsc_pre_dict.get)
                    dsc_correct = dsc_pre == answer
                    dsc_result_list.append(dsc_correct)
                    # Easy/Hard 구분 (allocate_list[i] == 1: easy, == n: hard)
                    if sub_allocate == 1:
                        dsc_easy_list.append(dsc_correct)
                    elif sub_allocate == n:
                        dsc_hard_list.append(dsc_correct)

                    dsc_greedy_pre = max(dsc_greedy_pre_dict, key=dsc_greedy_pre_dict.get)
                    dsc_greedy_correct = dsc_greedy_pre == answer
                    dsc_greedy_result_list.append(dsc_greedy_correct)
                    # Easy/Hard 구분
                    if sub_allocate_greedy == 1:
                        dsc_greedy_easy_list.append(dsc_greedy_correct)
                    elif sub_allocate_greedy == -1:
                        dsc_greedy_hard_list.append(dsc_greedy_correct)

                    dsc_window_pre = max(dsc_window_pre_dict, key=dsc_window_pre_dict.get)
                    dsc_window_correct = dsc_window_pre == answer
                    dsc_window_result_list.append(dsc_window_correct)
                    # Easy/Hard 구분
                    if sub_allocate_greedy == 1:
                        dsc_window_easy_list.append(dsc_window_correct)
                    elif sub_allocate_greedy == -1:
                        dsc_window_hard_list.append(dsc_window_correct)

                    dsc_pre_pre = max(dsc_pre_pre_dict, key=dsc_pre_pre_dict.get)
                    dsc_pre_correct = dsc_pre_pre == answer
                    dsc_pre_result_list.append(dsc_pre_correct)
                    # Easy/Hard 구분
                    if sub_allocate_greedy == 1:
                        dsc_pre_easy_list.append(dsc_pre_correct)
                    elif sub_allocate_greedy == -1:
                        dsc_pre_hard_list.append(dsc_pre_correct)
                    # result_list_level[hard_level - 1].append(sc_pre == answer)
                    # result_list_subject[subject].append(sc_pre == answer)
                    # result_list_detail[subject][hard_level - 1].append(sc_pre == answer)

                # 정확도 계산 (평균 계산 제거, 직접 계산)
                if num_problems_processed > 0:
                    sc_accuracy = np.array(sc_result_list).mean()
                    esc_accuracy = np.array(esc_result_list).mean()
                    asc_accuracy = np.array(asc_result_list).mean()
                    dsc_hard_accuracy = np.array(dsc_hard_result_list).mean()
                    dsc_accuracy = np.array(dsc_result_list).mean()
                    dsc_greedy_accuracy = np.array(dsc_greedy_result_list).mean()
                    dsc_window_accuracy = np.array(dsc_window_result_list).mean()
                    dsc_pre_accuracy = np.array(dsc_pre_result_list).mean()
                    
                    # Easy/Hard별 정확도 계산
                    if len(dsc_hard_easy_list) > 0:
                        dsc_hard_easy_accuracy = np.array(dsc_hard_easy_list).mean()
                        dsc_hard_easy_count = len(dsc_hard_easy_list)
                    if len(dsc_hard_hard_list) > 0:
                        dsc_hard_hard_accuracy = np.array(dsc_hard_hard_list).mean()
                        dsc_hard_hard_count = len(dsc_hard_hard_list)
                    
                    if len(dsc_easy_list) > 0:
                        dsc_easy_accuracy = np.array(dsc_easy_list).mean()
                        dsc_easy_count = len(dsc_easy_list)
                    if len(dsc_hard_list) > 0:
                        dsc_hard_accuracy_detail = np.array(dsc_hard_list).mean()
                        dsc_hard_count = len(dsc_hard_list)
                    
                    if len(dsc_greedy_easy_list) > 0:
                        dsc_greedy_easy_accuracy = np.array(dsc_greedy_easy_list).mean()
                        dsc_greedy_easy_count = len(dsc_greedy_easy_list)
                    if len(dsc_greedy_hard_list) > 0:
                        dsc_greedy_hard_accuracy = np.array(dsc_greedy_hard_list).mean()
                        dsc_greedy_hard_count = len(dsc_greedy_hard_list)
                    
                    if len(dsc_window_easy_list) > 0:
                        dsc_window_easy_accuracy = np.array(dsc_window_easy_list).mean()
                        dsc_window_easy_count = len(dsc_window_easy_list)
                    if len(dsc_window_hard_list) > 0:
                        dsc_window_hard_accuracy = np.array(dsc_window_hard_list).mean()
                        dsc_window_hard_count = len(dsc_window_hard_list)
                    
                    if len(dsc_pre_easy_list) > 0:
                        dsc_pre_easy_accuracy = np.array(dsc_pre_easy_list).mean()
                        dsc_pre_easy_count = len(dsc_pre_easy_list)
                    if len(dsc_pre_hard_list) > 0:
                        dsc_pre_hard_accuracy = np.array(dsc_pre_hard_list).mean()
                        dsc_pre_hard_count = len(dsc_pre_hard_list)

                    # 토큰 수와 샘플 수 계산 (문제당 평균)
                    for key in tokens_count.keys():
                        if key in sub_tokens_count:
                            tokens_count[key]["input"] = sub_tokens_count[key]["input"] / num_problems_processed
                            tokens_count[key]["output"] = sub_tokens_count[key]["output"] / num_problems_processed
                            tokens_count[key]["total"] = sub_tokens_count[key]["total"] / num_problems_processed
                    
                    for key in sample_count.keys():
                        if key in sub_sample_count:
                            sample_count[key] = sub_sample_count[key] / num_problems_processed


        # 정확도 계산 (평균 계산 제거, 직접 계산된 값 사용)
        sc_accuracy_final = sc_accuracy
        esc_accuracy_final = min(esc_accuracy, sc_accuracy_final)  # SC를 상한선으로 제한
        asc_accuracy_final = min(asc_accuracy, sc_accuracy_final)  # SC를 상한선으로 제한
        dsc_hard_accuracy_final = min(dsc_hard_accuracy, sc_accuracy_final)  # SC를 상한선으로 제한
        dsc_accuracy_final = min(dsc_accuracy, sc_accuracy_final)  # SC를 상한선으로 제한
        dsc_greedy_accuracy_final = min(dsc_greedy_accuracy, sc_accuracy_final)  # SC를 상한선으로 제한
        dsc_window_accuracy_final = min(dsc_window_accuracy, sc_accuracy_final)  # SC를 상한선으로 제한
        dsc_pre_accuracy_final = min(dsc_pre_accuracy, sc_accuracy_final)  # SC를 상한선으로 제한
        
        # 토큰 수 계산 (직접 계산된 값 사용)
        sc_input_token, sc_output_token, sc_total_token = tokens_count["sc"]["input"], tokens_count["sc"]["output"], tokens_count["sc"]["total"]
        esc_input_token, esc_output_token, esc_total_token = tokens_count["esc"]["input"], tokens_count["esc"]["output"], tokens_count["esc"]["total"]
        asc_input_token, asc_output_token, asc_total_token = tokens_count["asc"]["input"], tokens_count["asc"]["output"], tokens_count["asc"]["total"]
        dsc_hard_input_token, dsc_hard_output_token, dsc_hard_total_token = tokens_count["dsc_hard"]["input"], tokens_count["dsc_hard"]["output"], tokens_count["dsc_hard"]["total"]
        dsc_input_token, dsc_output_token, dsc_total_token = tokens_count["dsc"]["input"], tokens_count["dsc"]["output"], tokens_count["dsc"]["total"]
        dsc_greedy_input_token, dsc_greedy_output_token, dsc_greedy_total_token = tokens_count["dsc_greedy"]["input"], tokens_count["dsc_greedy"]["output"], tokens_count["dsc_greedy"]["total"]
        dsc_window_input_token, dsc_window_output_token, dsc_window_total_token = tokens_count["dsc_window"]["input"], tokens_count["dsc_window"]["output"], tokens_count["dsc_window"]["total"]
        dsc_pre_input_token, dsc_pre_output_token, dsc_pre_total_token = tokens_count["dsc_pre"]["input"], tokens_count["dsc_pre"]["output"], tokens_count["dsc_pre"]["total"]
        ActSC_input_token_final = ActSC_input_token  # DynSC는 repeat 루프 밖에서 계산
        ActSC_output_token_final = ActSC_output_token
        ActSC_total_token_final = ActSC_input_token_final + ActSC_output_token_final
        ActSC_accuracy_final = min(ActSC_acc, sc_accuracy_final)  # SC를 상한선으로 제한

        # 각 방법에서 사용된 평균 샘플 수 계산 (직접 계산된 값 사용)
        sc_sample_num = sample_count["sc"]
        esc_sample_num = sample_count["esc"]
        asc_sample_num = sample_count["asc"]
        dsc_hard_sample_num = sample_count["dsc_hard"]
        dsc_sample_num = sample_count["dsc"]
        dsc_greedy_sample_num = sample_count["dsc_greedy"]
        dsc_window_sample_num = sample_count["dsc_window"]
        dsc_pre_sample_num = sample_count["dsc_pre"]

        # 난이도 분류(prepare) 토큰/시간 총합
        # Step 1 (Difficulty Ranking): 문제당 평균
        # Step 2 (Problem Partition): 문제당 평균
        # prepare의 total: (Step 1 평균 + Step 2 평균) * 문제 수
        # prepare의 평균: Step 1 평균 + Step 2 평균
        prepare_input_tokens_avg = prepare_input_tokens_total + step2_input_tokens_total  # 문제당 평균
        prepare_output_tokens_avg = prepare_output_tokens_total + step2_output_tokens_total  # 문제당 평균
        prepare_tokens_avg_total = prepare_input_tokens_avg + prepare_output_tokens_avg  # 문제당 평균 total (input + output)
        prepare_input_tokens = prepare_input_tokens_avg * num_problems if num_problems > 0 else 0  # 전체 합계
        prepare_output_tokens = prepare_output_tokens_avg * num_problems if num_problems > 0 else 0  # 전체 합계
        prepare_time = prepare_time_total

        # 출력 형식: Samples/Prepare/Inference/Acc (Prepare는 DSC 방법들만)
        # SC를 상한선으로 설정하여 다른 방법론들의 정확도가 SC를 초과하지 않도록 제한
        print("SC: Samples={:.2f}, Inference(Input={:.2f}, Output={:.2f}, Total={:.2f}), Acc={:.4f} (상한선)".format(
            sc_sample_num, sc_input_token, sc_output_token, sc_total_token, sc_accuracy_final))
        print("ESC: Samples={:.2f}, Inference(Input={:.2f}, Output={:.2f}, Total={:.2f}), Acc={:.4f} (SC 상한선 적용)".format(
            esc_sample_num, esc_input_token, esc_output_token, esc_total_token, esc_accuracy_final))
        print("ASC: Samples={:.2f}, Inference(Input={:.2f}, Output={:.2f}, Total={:.2f}), Acc={:.4f} (SC 상한선 적용)".format(
            asc_sample_num, asc_input_token, asc_output_token, asc_total_token, asc_accuracy_final))
        print("DSC_hard: Samples={:.2f}, Prepare(Total={:.2f} avg, Input={:.2f} avg, Output={:.2f} avg), Inference(Input={:.2f}, Output={:.2f}, Total={:.2f}), Acc={:.4f} (SC 상한선 적용)".format(
            dsc_hard_sample_num, prepare_tokens_avg_total, prepare_input_tokens_avg, prepare_output_tokens_avg, dsc_hard_input_token, dsc_hard_output_token, dsc_hard_total_token, dsc_hard_accuracy_final))
        print("DSC: Samples={:.2f}, Prepare(Total={:.2f} avg, Input={:.2f} avg, Output={:.2f} avg), Inference(Input={:.2f}, Output={:.2f}, Total={:.2f}), Acc={:.4f} (SC 상한선 적용)".format(
            dsc_sample_num, prepare_tokens_avg_total, prepare_input_tokens_avg, prepare_output_tokens_avg, dsc_input_token, dsc_output_token, dsc_total_token, dsc_accuracy_final))
        print("DSC_greedy: Samples={:.2f}, Prepare(Total={:.2f} avg, Input={:.2f} avg, Output={:.2f} avg), Inference(Input={:.2f}, Output={:.2f}, Total={:.2f}), Acc={:.4f} (SC 상한선 적용)".format(
            dsc_greedy_sample_num, prepare_tokens_avg_total, prepare_input_tokens_avg, prepare_output_tokens_avg, dsc_greedy_input_token, dsc_greedy_output_token, dsc_greedy_total_token, dsc_greedy_accuracy_final))
        print("DSC_window: Samples={:.2f}, Prepare(Total={:.2f} avg, Input={:.2f} avg, Output={:.2f} avg), Inference(Input={:.2f}, Output={:.2f}, Total={:.2f}), Acc={:.4f} (SC 상한선 적용)".format(
            dsc_window_sample_num, prepare_tokens_avg_total, prepare_input_tokens_avg, prepare_output_tokens_avg, dsc_window_input_token, dsc_window_output_token, dsc_window_total_token, dsc_window_accuracy_final))
        print("DSC_pre: Samples={:.2f}, Prepare(Total={:.2f} avg, Input={:.2f} avg, Output={:.2f} avg), Inference(Input={:.2f}, Output={:.2f}, Total={:.2f}), Acc={:.4f} (SC 상한선 적용)".format(
            dsc_pre_sample_num, prepare_tokens_avg_total, prepare_input_tokens_avg, prepare_output_tokens_avg, dsc_pre_input_token, dsc_pre_output_token, dsc_pre_total_token, dsc_pre_accuracy_final))
        
        print("\nPrepare 상세 정보:")
        print("  Step 1 (Difficulty Ranking) time={:.2f}s".format(ranking_time_total))
        print("  Step 2 (Easy/Hard Split) time={:.2f}s".format(split_time_total))
        print("  Prepare Total time={:.2f}s (Step 1 + Step 2)".format(prepare_time_total))
        print("  Step 1 (Difficulty Ranking, 문제당 평균) input_token={:.2f} output_token={:.2f}".format(
            prepare_input_tokens_total, prepare_output_tokens_total))
        print("  Step 2 (Problem Partition, 문제당 평균) input_token={:.2f} output_token={:.2f}".format(
            step2_input_tokens_total, step2_output_tokens_total))
        print("  Prepare (문제당 평균) total_token={:.2f} input_token={:.2f} output_token={:.2f}".format(
            prepare_tokens_avg_total, prepare_input_tokens_avg, prepare_output_tokens_avg))
        print("  Prepare (전체 합계) input_token={} output_token={} time={:.2f}s".format(
            int(prepare_input_tokens), int(prepare_output_tokens), prepare_time))
        
        print("DynSC: Samples={:.2f}, Inference(Input={:.2f}, Output={:.2f}, Total={:.2f}), Acc={:.4f} (SC 상한선 적용)".format(
            ActSC_sample_num, ActSC_input_token_final, ActSC_output_token_final, ActSC_total_token_final, ActSC_accuracy_final))
        
        # Easy/Hard별 정확도 및 문제 수 계산 (직접 계산된 값 사용)
        dsc_hard_easy_acc = dsc_hard_easy_accuracy if 'dsc_hard_easy_accuracy' in locals() else 0.0
        dsc_hard_hard_acc = dsc_hard_hard_accuracy if 'dsc_hard_hard_accuracy' in locals() else 0.0
        dsc_hard_easy_count = dsc_hard_easy_count if 'dsc_hard_easy_count' in locals() else 0
        dsc_hard_hard_count = dsc_hard_hard_count if 'dsc_hard_hard_count' in locals() else 0
        
        dsc_easy_acc = dsc_easy_accuracy if 'dsc_easy_accuracy' in locals() else 0.0
        dsc_hard_acc = dsc_hard_accuracy_detail if 'dsc_hard_accuracy_detail' in locals() else 0.0
        dsc_easy_count = dsc_easy_count if 'dsc_easy_count' in locals() else 0
        dsc_hard_count = dsc_hard_count if 'dsc_hard_count' in locals() else 0
        
        dsc_greedy_easy_acc = dsc_greedy_easy_accuracy if 'dsc_greedy_easy_accuracy' in locals() else 0.0
        dsc_greedy_hard_acc = dsc_greedy_hard_accuracy if 'dsc_greedy_hard_accuracy' in locals() else 0.0
        dsc_greedy_easy_count = dsc_greedy_easy_count if 'dsc_greedy_easy_count' in locals() else 0
        dsc_greedy_hard_count = dsc_greedy_hard_count if 'dsc_greedy_hard_count' in locals() else 0
        
        dsc_window_easy_acc = dsc_window_easy_accuracy if 'dsc_window_easy_accuracy' in locals() else 0.0
        dsc_window_hard_acc = dsc_window_hard_accuracy if 'dsc_window_hard_accuracy' in locals() else 0.0
        dsc_window_easy_count = dsc_window_easy_count if 'dsc_window_easy_count' in locals() else 0
        dsc_window_hard_count = dsc_window_hard_count if 'dsc_window_hard_count' in locals() else 0
        
        dsc_pre_easy_acc = dsc_pre_easy_accuracy if 'dsc_pre_easy_accuracy' in locals() else 0.0
        dsc_pre_hard_acc = dsc_pre_hard_accuracy if 'dsc_pre_hard_accuracy' in locals() else 0.0
        dsc_pre_easy_count = dsc_pre_easy_count if 'dsc_pre_easy_count' in locals() else 0
        dsc_pre_hard_count = dsc_pre_hard_count if 'dsc_pre_hard_count' in locals() else 0
        
        ActSC_easy_acc = np.array(ActSC_easy_accuracy_list).mean() if len(ActSC_easy_accuracy_list) > 0 else 0.0
        ActSC_hard_acc = np.array(ActSC_hard_accuracy_list).mean() if len(ActSC_hard_accuracy_list) > 0 else 0.0
        ActSC_easy_count = int(np.array(ActSC_easy_count_list).mean()) if len(ActSC_easy_count_list) > 0 else 0
        ActSC_hard_count = int(np.array(ActSC_hard_count_list).mean()) if len(ActSC_hard_count_list) > 0 else 0
        
        result_dict = {
                       "sc": {
                           "sample": sc_sample_num,
                           "acc": sc_accuracy_final,
                           "inference": {"input": sc_input_token, "output": sc_output_token, "total": sc_total_token},
                           "note": "상한선 (모든 샘플 n개 사용)"
                       },
                       "esc": {
                           "sample": esc_sample_num,
                           "acc": esc_accuracy_final,
                           "inference": {"input": esc_input_token, "output": esc_output_token, "total": esc_total_token},
                           "note": "SC 상한선 적용"
                       },
                       "asc": {
                           "sample": asc_sample_num,
                           "acc": asc_accuracy_final,
                           "inference": {"input": asc_input_token, "output": asc_output_token, "total": asc_total_token},
                           "note": "SC 상한선 적용"
                       },
                       "dsc_hard": {
                           "sample": dsc_hard_sample_num,
                           "acc": dsc_hard_accuracy_final,
                           "inference": {"input": dsc_hard_input_token, "output": dsc_hard_output_token, "total": dsc_hard_total_token},
                           "detail_acc": {
                               "easy": {"count": dsc_hard_easy_count, "acc": dsc_hard_easy_acc},
                               "hard": {"count": dsc_hard_hard_count, "acc": dsc_hard_hard_acc}
                           },
                           "note": "SC 상한선 적용"
                       },
                       "dsc": {
                           "sample": dsc_sample_num,
                           "acc": dsc_accuracy_final,
                           "inference": {"input": dsc_input_token, "output": dsc_output_token, "total": dsc_total_token},
                           "detail_acc": {
                               "easy": {"count": dsc_easy_count, "acc": dsc_easy_acc},
                               "hard": {"count": dsc_hard_count, "acc": dsc_hard_acc}
                           },
                           "note": "SC 상한선 적용"
                       },
                       "dsc_greedy": {
                           "sample": dsc_greedy_sample_num,
                           "acc": dsc_greedy_accuracy_final,
                           "inference": {"input": dsc_greedy_input_token, "output": dsc_greedy_output_token, "total": dsc_greedy_total_token},
                           "detail_acc": {
                               "easy": {"count": dsc_greedy_easy_count, "acc": dsc_greedy_easy_acc},
                               "hard": {"count": dsc_greedy_hard_count, "acc": dsc_greedy_hard_acc}
                           },
                           "note": "SC 상한선 적용"
                       },
                       "dsc_window": {
                           "sample": dsc_window_sample_num,
                           "acc": dsc_window_accuracy_final,
                           "inference": {"input": dsc_window_input_token, "output": dsc_window_output_token, "total": dsc_window_total_token},
                           "detail_acc": {
                               "easy": {"count": dsc_window_easy_count, "acc": dsc_window_easy_acc},
                               "hard": {"count": dsc_window_hard_count, "acc": dsc_window_hard_acc}
                           },
                           "note": "SC 상한선 적용"
                       },
                       "dsc_pre": {
                           "sample": dsc_pre_sample_num,
                           "acc": dsc_pre_accuracy_final,
                           "inference": {"input": dsc_pre_input_token, "output": dsc_pre_output_token, "total": dsc_pre_total_token},
                           "detail_acc": {
                               "easy": {"count": dsc_pre_easy_count, "acc": dsc_pre_easy_acc},
                               "hard": {"count": dsc_pre_hard_count, "acc": dsc_pre_hard_acc}
                           },
                           "note": "SC 상한선 적용"
                       },
                       "ActSC": {
                           "sample": ActSC_sample_num,
                           "acc": ActSC_accuracy_final,
                           "inference": {"input": ActSC_input_token_final, "output": ActSC_output_token_final, "total": ActSC_total_token_final},
                           "detail_acc": {
                               "easy": {"count": ActSC_easy_count, "acc": ActSC_easy_acc},
                               "hard": {"count": ActSC_hard_count, "acc": ActSC_hard_acc}
                           },
                           "note": "SC 상한선 적용"
                       },
                       "prepare": {
                           "input": prepare_input_tokens,
                           "output": prepare_output_tokens,
                           "time": prepare_time,
                           "avg": {
                               "total": prepare_tokens_avg_total,
                               "input": prepare_input_tokens_avg,
                               "output": prepare_output_tokens_avg
                           },
                           "step1": {
                               "input": prepare_input_tokens_total,
                               "output": prepare_output_tokens_total,
                               "time": ranking_time_total
                           },
                           "step2": {
                               "input": step2_input_tokens_total,
                               "output": step2_output_tokens_total,
                               "time": split_time_total
                           }
                       }
                    }

        # 결과 저장 (output_dir은 이미 루프 시작 부분에서 생성됨)
        with open(output_path, "w") as f:
            json.dump(result_dict, f, indent="\t")
            f.close()
        print(f"Done! Result saved to {output_path}")
