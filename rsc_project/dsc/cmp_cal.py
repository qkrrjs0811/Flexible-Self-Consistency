# DSC 기법별 성능/비용 비교

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
from typing import Any, Dict, List, Tuple, Optional
from scipy import integrate

# rsc_project 모듈 import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from evaluation.metrics import normalize_answer_eval
from rsc.aggregator import SCAggregator

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
import transformers

class StoppingCriterias:

    def __init__(self, *args, **kwargs):

        ...

    def should_stop(self, *args, **kwargs) -> Dict:
        ...

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


# ESC, ASC 함수 제거 (DSC만 사용)




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


def _load_mmlu_pro_category_by_index(dsc_input_path: str, pre_len: int) -> Optional[List[str]]:
    """Stage1 JSON 과 dsc_eval 줄 수가 같을 때 인덱스별 MMLU category."""
    if not dsc_input_path.endswith(".json") or not os.path.isfile(dsc_input_path):
        return None
    try:
        with open(dsc_input_path, "r", encoding="utf-8") as f:
            stage1 = json.load(f)
    except Exception:
        return None
    cats = []
    for item in stage1:
        meta = item.get("metadata") or {}
        raw = meta.get("category")
        cats.append(str(raw).strip() if raw is not None and str(raw).strip() else "unknown")
    if len(cats) != pre_len:
        print(
            f"Warning: MMLU-Pro Stage1 len {len(cats)} != dsc_eval len {pre_len}; "
            "category aggregation uses index fallback (unknown if out of range)."
        )
    return cats


def _mmlu_dsc_cat_agg_new():
    from collections import defaultdict

    return defaultdict(
        lambda: {
            "n": 0,
            "dh_c": 0,
            "dh_i": 0.0,
            "dh_o": 0.0,
            "dh_s": 0.0,
            "d_c": 0,
            "d_i": 0.0,
            "d_o": 0.0,
            "d_s": 0.0,
            "dg_c": 0,
            "dg_i": 0.0,
            "dg_o": 0.0,
            "dg_s": 0.0,
            "dw_c": 0,
            "dw_i": 0.0,
            "dw_o": 0.0,
            "dw_s": 0.0,
            "dp_c": 0,
            "dp_i": 0.0,
            "dp_o": 0.0,
            "dp_s": 0.0,
        }
    )


def _finalize_mmlu_dsc_trial_agg(agg) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    def _pack(z, n, c, i, o, s):
        return {
            "accuracy": round(100.0 * z[c] / n, 4),
            "avg_input_tokens": round(z[i] / n, 4),
            "avg_output_tokens": round(z[o] / n, 4),
            "avg_total_tokens": round((z[i] + z[o]) / n, 4),
            "avg_chain_steps": round(z[s] / n, 4),
        }

    for cat, z in sorted(agg.items()):
        n = z["n"]
        if n <= 0:
            continue
        out[cat] = {
            "sample_count": n,
            "dsc_hard": _pack(z, n, "dh_c", "dh_i", "dh_o", "dh_s"),
            "dsc": _pack(z, n, "d_c", "d_i", "d_o", "d_s"),
            "dsc_greedy": _pack(z, n, "dg_c", "dg_i", "dg_o", "dg_s"),
            "dsc_window": _pack(z, n, "dw_c", "dw_i", "dw_o", "dw_s"),
            "dsc_pre": _pack(z, n, "dp_c", "dp_i", "dp_o", "dp_s"),
        }
    return out


def _average_mmlu_dsc_category_trials(trials: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not trials:
        return {}
    if len(trials) == 1:
        return trials[0]
    all_cats = set()
    for t in trials:
        all_cats.update(t.keys())
    out: Dict[str, Any] = {}
    methods = ["dsc_hard", "dsc", "dsc_greedy", "dsc_window", "dsc_pre"]
    fields = ["accuracy", "avg_input_tokens", "avg_output_tokens", "avg_total_tokens", "avg_chain_steps"]
    for cat in sorted(all_cats):
        per_trial = [t[cat] for t in trials if cat in t]
        if not per_trial:
            continue
        n0 = per_trial[0].get("sample_count", 0)
        block: Dict[str, Any] = {"sample_count": n0}
        for m in methods:
            block[m] = {}
            for fld in fields:
                vals = [pt[m][fld] for pt in per_trial if m in pt and fld in pt[m]]
                if vals:
                    block[m][fld] = round(float(np.mean(vals)), 4)
        out[cat] = block
    return out


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

def random_elements(A, input_tokens, num):
    # metrics.py 동작과 유사하게, pool이 num보다 작으면 가능한 만큼만 사용
    if not A:
        out = {"input": input_tokens.get("input", 0), "output": []}
        return [], out
    num = min(int(num), len(A))
    if num <= 0:
        out = {"input": input_tokens.get("input", 0), "output": []}
        return [], out
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
    print("hard count={}".format(sum([sub == max_sample_size for sub in allocate_list])))
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
    print("greedy hard count={}".format(sum([sub == -1 for sub in allocate_list])))
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

# ActSC/DynSC 관련 함수 제거 (DSC만 사용)

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
        # pool이 max_sample_size보다 작으면 가능한 만큼만 사용
        k = min(max_sample_size, len(pre_list[j])) if pre_list[j] else 0
        random_choice = random.sample(pre_list[j], k) if k > 0 else []

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

    # 튜플을 리스트로 변환
    return list(actual_window)

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
        # metrics.py와 동일하게, pool이 부족한 샘플도 스킵하지 않고 가능한 만큼으로 평가
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


def get_sc_sample_path(
    model_name,
    dataset_name,
    results_dir="experiments/results",
    stages_config="40",
):
    """
    SC 샘플 파일 경로를 생성하는 헬퍼 함수
    
    Args:
        model_name: 전체 HF 모델 ID (예: "Qwen/Qwen2.5-7B-Instruct", "google/gemma-3-4b-it")
        dataset_name: 데이터셋 이름 (예: "GSM8K", "AIME_2024", "MATH500_test", "MMLU-Pro", "GPQA-Diamond")
        results_dir: 결과 디렉토리 경로
        stages_config: Stages 설정 (기본값: "40")
    
    Returns:
        SC 샘플 파일의 전체 경로 (results_dir/{model_name}/stages-{n}/Stage1/{dataset_name}.json)
    """
    # 경로 구성: results_dir/{model_name}/stages-{n}/Stage1/{dataset_name}.json
    sc_sample_path = os.path.join(
        results_dir,
        model_name,
        f"stages-{stages_config}",
        "Stage1",
        f"{dataset_name}.json"
    )
    
    return sc_sample_path


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

def add_dsc_to_eval_result_all(results_dir: str, model_name: str, dataset_name: str,
                               stages_config: str, trial_accuracies: Dict[str, list],
                               trial_tokens: Dict = None, trial_prepare: Dict = None,
                               trial_samples: Dict = None):
    """
    trial_tokens: optional Dict e.g. {"dsc_pre": {"input": [float,...], "output": [float,...]}}
    trial_prepare: optional Dict e.g. {"input": [float,...], "output": [float,...], "time": [float,...]} (시행별 prepare)
    trial_samples: optional Dict e.g. {"dsc_pre": [float,...]} (시행별 avg_chains)
    """
    eval_result_dir = os.path.join(results_dir, model_name, f"stages-{stages_config}")
    eval_result_all_path = os.path.join(eval_result_dir, 'eval_result_all.txt')

    methods_to_save = {
        "DSC": trial_accuracies.get("dsc_pre", [])
    }

    dsc_input_trials = []
    dsc_output_trials = []
    if trial_tokens and "dsc_pre" in trial_tokens:
        dsc_input_trials = trial_tokens["dsc_pre"].get("input", [])
        dsc_output_trials = trial_tokens["dsc_pre"].get("output", [])

    dsc_chains_trials = []
    if trial_samples and "dsc_pre" in trial_samples:
        dsc_chains_trials = trial_samples.get("dsc_pre", [])

    prepare_input_trials = trial_prepare.get("input", []) if trial_prepare else []
    prepare_output_trials = trial_prepare.get("output", []) if trial_prepare else []
    prepare_time_trials = trial_prepare.get("time", []) if trial_prepare else []

    if not any(methods_to_save.values()):
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

        for method_name, trials in methods_to_save.items():
            if not trials:
                continue

            method_prefix = f"{method_name} Trials:"
            line_indices = []
            for i in range(section_start_idx, section_end_idx):
                if all_lines[i].startswith(method_prefix):
                    line_indices.append(i)

            for idx in sorted(line_indices, reverse=True):
                all_lines.pop(idx)
                section_end_idx -= 1

            # DSC Chains/Tokens, DSC Prepare Input/Output/Time 기존 라인 제거
            for label in ["DSC Chains Trials:", "DSC Input Tokens Trials:", "DSC Output Tokens Trials:",
                          "DSC Prepare Input Tokens Trials:", "DSC Prepare Output Tokens Trials:", "DSC Prepare Time Trials:"]:
                line_indices = []
                for i in range(section_start_idx, section_end_idx):
                    if all_lines[i].startswith(label):
                        line_indices.append(i)
                for idx in sorted(line_indices, reverse=True):
                    all_lines.pop(idx)
                    section_end_idx -= 1

            # ESC 블록(ESC Trials, ESC Chains, ESC Tokens 등) 바로 뒤에 삽입
            esc_idx = -1
            for i in range(section_start_idx, section_end_idx):
                if all_lines[i].startswith("ESC Trials:"):
                    esc_idx = i
                    break
            if esc_idx >= 0:
                # ESC로 시작하는 연속 블록의 마지막 줄을 찾는다
                last_esc_idx = esc_idx
                for j in range(esc_idx + 1, section_end_idx):
                    line = all_lines[j]
                    if line.startswith("ESC "):
                        last_esc_idx = j
                        continue
                    # 빈 줄은 ESC 블록의 일부로 간주하지 않고 종료
                    if line.strip() == "":
                        last_esc_idx = j
                        continue
                    break
                insert_idx = last_esc_idx + 1
            else:
                insert_idx = section_end_idx
                while insert_idx > section_start_idx and all_lines[insert_idx - 1].strip() == "":
                    insert_idx -= 1

            trials_str = f"{method_name} Trials: " + "[" + ", ".join(f"{x:.4f}" for x in trials) + "]\n"
            all_lines.insert(insert_idx, trials_str)
            section_end_idx += 1
            # 순서: DSC Trials(위), DSC Chains, DSC Input Tokens, DSC Output Tokens, DSC Prepare Input/Output/Time
            # insert(insert_idx+1)를 반복하면 역순으로 쌓이므로, 원하는 순서의 역순으로 삽입
            if prepare_time_trials:
                all_lines.insert(insert_idx + 1, "DSC Prepare Time Trials: " + "[" + ", ".join(f"{float(x):.4f}" for x in prepare_time_trials) + "]\n")
                section_end_idx += 1
            if prepare_output_trials:
                all_lines.insert(insert_idx + 1, "DSC Prepare Output Tokens Trials: " + "[" + ", ".join(f"{float(x):.2f}" for x in prepare_output_trials) + "]\n")
                section_end_idx += 1
            if prepare_input_trials:
                all_lines.insert(insert_idx + 1, "DSC Prepare Input Tokens Trials: " + "[" + ", ".join(f"{float(x):.2f}" for x in prepare_input_trials) + "]\n")
                section_end_idx += 1
            if dsc_output_trials:
                all_lines.insert(insert_idx + 1, "DSC Output Tokens Trials: " + "[" + ", ".join(f"{float(x):.2f}" for x in dsc_output_trials) + "]\n")
                section_end_idx += 1
            if dsc_input_trials:
                all_lines.insert(insert_idx + 1, "DSC Input Tokens Trials: " + "[" + ", ".join(f"{float(x):.2f}" for x in dsc_input_trials) + "]\n")
                section_end_idx += 1
            if dsc_chains_trials:
                all_lines.insert(insert_idx + 1, "DSC Chains Trials: " + "[" + ", ".join(f"{float(x):.2f}" for x in dsc_chains_trials) + "]\n")
                section_end_idx += 1
    else:
        new_section = f"======================================================================\n"
        new_section += f"## Dataset: {dataset_name}\n"
        new_section += f"======================================================================\n"
        for method_name, trials in methods_to_save.items():
            if trials:
                new_section += f"{method_name} Trials: " + "[" + ", ".join(f"{x:.4f}" for x in trials) + "]\n"
        # 순서: DSC Trials(위에서 이미 추가), DSC Chains, DSC Input Tokens, DSC Output Tokens, DSC Prepare Input/Output/Time
        if dsc_chains_trials:
            new_section += "DSC Chains Trials: " + "[" + ", ".join(f"{float(x):.2f}" for x in dsc_chains_trials) + "]\n"
        if dsc_input_trials:
            new_section += "DSC Input Tokens Trials: " + "[" + ", ".join(f"{float(x):.2f}" for x in dsc_input_trials) + "]\n"
        if dsc_output_trials:
            new_section += "DSC Output Tokens Trials: " + "[" + ", ".join(f"{float(x):.2f}" for x in dsc_output_trials) + "]\n"
        if prepare_input_trials:
            new_section += "DSC Prepare Input Tokens Trials: " + "[" + ", ".join(f"{float(x):.2f}" for x in prepare_input_trials) + "]\n"
        if prepare_output_trials:
            new_section += "DSC Prepare Output Tokens Trials: " + "[" + ", ".join(f"{float(x):.2f}" for x in prepare_output_trials) + "]\n"
        if prepare_time_trials:
            new_section += "DSC Prepare Time Trials: " + "[" + ", ".join(f"{float(x):.4f}" for x in prepare_time_trials) + "]\n"
        new_section += "\n"
        all_lines.extend(new_section.splitlines(keepends=True))

    os.makedirs(eval_result_dir, exist_ok=True)
    with open(eval_result_all_path, 'w', encoding='utf-8') as f:
        f.writelines(all_lines)

    print(f"DSC results added to eval_result_all.txt: {eval_result_all_path}")

def add_dsc_to_eval_result(results_dir: str, model_name: str, dataset_name: str, 
                           stages_config: str, dsc_result: Dict):
    """
    eval_result.txt 파일에 DSC 결과 추가 (AC Accuracy 바로 아래).
    단일 실험 모드(RSC_EVAL_SINGLE_OUTPUT_PATH 설정 시)에는 eval_result.txt는 수정하지 않고
    eval_single_{seed}.txt에만 동일한 형식으로 DSC 결과를 기입한다.
    
    Args:
        results_dir: 결과 디렉토리 경로
        model_name: 모델 이름 (예: "Qwen/Qwen2.5-7B-Instruct")
        dataset_name: 데이터셋 이름
        stages_config: stages 설정 (예: "40")
        dsc_result: DSC 결과 딕셔너리 (dsc_pre와 prepare 정보 포함)
    """
    single_output_path = os.environ.get('RSC_EVAL_SINGLE_OUTPUT_PATH', '').strip()
    is_single_run = bool(single_output_path)
    
    if is_single_run:
        target_path = single_output_path
        if not os.path.exists(target_path):
            print(f"Warning: eval_single file not found at {target_path}, skipping DSC result addition")
            return
    else:
        eval_result_dir = os.path.join(results_dir, model_name, f"stages-{stages_config}")
        eval_result_path = os.path.join(eval_result_dir, 'eval_result.txt')
        target_path = eval_result_path
        if not os.path.exists(eval_result_path):
            print(f"Warning: eval_result.txt not found at {eval_result_path}, skipping DSC result addition")
            return
        eval_result_all_path = os.path.join(eval_result_dir, 'eval_result_all.txt')
    
    # 이미 DSC 평가가 되어있는지 확인 (단일 실험 시에는 target_path만 확인)
    def _has_metric_in_file(filepath: str, ds_name: str, m_name: str) -> bool:
        if not os.path.exists(filepath):
            return False
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        ds_pattern = rf"## Dataset:\s*{re.escape(ds_name)}\s*\n"
        ds_match = re.search(ds_pattern, content)
        if not ds_match:
            return False
            
        start_idx = ds_match.end()
        next_ds_pattern = r"## Dataset:"
        next_ds_match = re.search(next_ds_pattern, content[start_idx:])
        
        end_idx = start_idx + next_ds_match.start() if next_ds_match else len(content)
        section_content = content[start_idx:end_idx]
        
        return re.search(rf"^\s*{re.escape(m_name)}", section_content, flags=re.MULTILINE) is not None

    if is_single_run:
        if _has_metric_in_file(target_path, dataset_name, "DSC Accuracy:"):
            print(f"DSC results already exist in {target_path} for {dataset_name}. Skipping addition.")
            return
    else:
        if _has_metric_in_file(eval_result_path, dataset_name, "DSC Accuracy:") and os.path.exists(eval_result_all_path):
            if _has_metric_in_file(eval_result_all_path, dataset_name, "DSC Trials:"):
                print(f"DSC results already exist in {eval_result_path} and {eval_result_all_path} for {dataset_name}. Skipping addition.")
                return
    
    # dsc_pre 정보 추출
    dsc_pre = dsc_result.get('dsc_pre', {})
    dsc_acc = dsc_pre.get('acc', 0.0)
    avg_chains = dsc_pre.get('sample', 0.0)
    avg_input_tokens = dsc_pre.get('inference', {}).get('input', 0.0)
    avg_output_tokens = dsc_pre.get('inference', {}).get('output', 0.0)
    
    # prepare 정보 추출
    prepare = dsc_result.get('prepare', {})
    prepare_input = prepare.get('input', 0.0)
    prepare_output = prepare.get('output', 0.0)
    prepare_time = prepare.get('time', 0.0)
    
    # DSC 결과 라인 생성
    dsc_accuracy_line = f"DSC Accuracy: {dsc_acc:.4f} (avg_chains={avg_chains:.2f}, avg_input_tokens={avg_input_tokens:.2f}, avg_output_tokens={avg_output_tokens:.2f})\n"
    dsc_prepare_line = f"DSC Prepare (input: {prepare_input:.2f}, output: {prepare_output:.2f}, time: {prepare_time:.4f})\n"
    
    # 기존 내용 읽기 (단일 실험 시 eval_single_*.txt, 아니면 eval_result.txt)
    with open(target_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 해당 데이터셋 섹션 찾기
    dataset_header = f"## Dataset: {dataset_name}"
    section_start_idx = -1
    section_end_idx = -1
    
    for i, line in enumerate(lines):
        if dataset_header in line:
            section_start_idx = i
            break
    
    if section_start_idx < 0:
        print(f"Warning: Dataset section '{dataset_name}' not found in target file, skipping DSC result addition")
        return
    
    # 섹션 끝 찾기 (다음 데이터셋 섹션 시작 전까지)
    for i in range(section_start_idx + 1, len(lines)):
        if lines[i].startswith("## Dataset:"):
            section_end_idx = i
            break
    
    if section_end_idx < 0:
        # 마지막 섹션인 경우
        section_end_idx = len(lines)
    
    # 섹션 내에서 "ESC Accuracy" 라인 찾기 (없으면 AC Accuracy, 그것도 없으면 SC Accuracy 순으로 fallback)
    esc_accuracy_idx = -1
    ac_accuracy_idx = -1
    sc_accuracy_idx = -1
    for i in range(section_start_idx, section_end_idx):
        line = lines[i]
        if 'ESC Accuracy:' in line and esc_accuracy_idx < 0:
            esc_accuracy_idx = i
        if 'AC Accuracy:' in line and ac_accuracy_idx < 0:
            ac_accuracy_idx = i
        if 'SC Accuracy:' in line and sc_accuracy_idx < 0:
            sc_accuracy_idx = i

    # 삽입 기준 인덱스 선택: ESC > AC > SC
    base_idx = esc_accuracy_idx if esc_accuracy_idx >= 0 else (
        ac_accuracy_idx if ac_accuracy_idx >= 0 else sc_accuracy_idx
    )

    if base_idx < 0:
        print(f"Warning: 'ESC/AC/SC Accuracy' line not found in dataset section '{dataset_name}', skipping DSC result addition")
        return
    
    # 기존 DSC 라인들 찾기 (있으면 제거)
    dsc_line_indices = []
    for i in range(section_start_idx, section_end_idx):
        line = lines[i]
        if 'DSC Accuracy:' in line or 'DSC Prepare' in line:
            dsc_line_indices.append(i)
    
    # 기존 DSC 라인들 제거 (뒤에서부터 제거하여 인덱스 유지)
    for idx in sorted(dsc_line_indices, reverse=True):
        lines.pop(idx)
        # 기준 인덱스 조정
        if idx <= base_idx:
            base_idx -= 1
    
    # ESC(또는 fallback된 Accuracy) 바로 아래에 DSC 결과 삽입
    insert_idx = base_idx + 1
    
    # 기준 라인 다음의 빈 줄이 있으면 제거
    if insert_idx < len(lines) and lines[insert_idx].strip() == "":
        # 빈 줄 제거
        lines.pop(insert_idx)
    
    # DSC Accuracy를 AC Accuracy 바로 다음에 삽입
    lines.insert(insert_idx, dsc_accuracy_line)
    # DSC Prepare를 그 다음에 삽입
    lines.insert(insert_idx + 1, dsc_prepare_line)
    
    # DSC Prepare 다음에 빈 줄 추가 (Ours와 분리하기 위해)
    # DSC Prepare 다음이 빈 줄이 아니면 빈 줄 추가
    prepare_next_idx = insert_idx + 2
    if prepare_next_idx >= len(lines) or lines[prepare_next_idx].strip() != "":
        lines.insert(prepare_next_idx, "\n")
    
    # 파일에 저장 (단일 실험 시 eval_single_*.txt, 아니면 eval_result.txt)
    with open(target_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    if is_single_run:
        print(f"DSC results added to eval_single (eval_result.txt 미수정): {target_path}")
    else:
        print(f"DSC results added to eval_result.txt: {target_path}")


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
                       help="데이터셋 이름 (GSM8K, AIME_2024, AIME_2025, MATH500_test, MMLU-Pro, MMLU-Pro_full, GPQA-Diamond). model_name과 함께 사용")
    parser.add_argument("--n", type=int, default=40,
                       help="DSC/ESC/AC/SC 평가에서 사용할 체인 수 (기본값: 40)")
    parser.add_argument("--subset_n", type=int, default=40,
                       help="pool에서 먼저 샘플링할 subset 크기 (기본값: 40, 모든 메트릭과 동일)")
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--judge_window_size", type=int, default=24,
                       help="Judge window size for DSC allocation")
    parser.add_argument("--interplot_size", type=int, default=16,
                       help="Interplot size for window allocation")
    parser.add_argument("--num_trials", type=int, default=1,
                       help="Number of bootstrap trials (default: 1, no bootstrapping)")
    parser.add_argument("--bootstrap_seed", type=int, default=42,
                       help="Random seed for bootstrapping (default: 42)")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="experiments/results",
        help="Results directory (default: experiments/results)",
    )
    parser.add_argument("--stages_config", type=str, default="40",
                       help="Stages configuration (default: 40)")
    parser.add_argument(
        "--reuse_result_json",
        type=str,
        default=None,
        help="기존 DSC 결과 JSON 경로를 사용하여 eval_result.txt만 갱신할 때 사용"
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    stages_config = args.stages_config
    # stages_config는 "40" 또는 "stages-40" 형태 모두 허용
    if isinstance(stages_config, str):
        stages_config = stages_config.strip()
        if stages_config.startswith("stages-"):
            stages_config = stages_config[len("stages-"):]

    # ------------------------------------------------------------------
    # 재사용 모드: 이미 계산된 DSC 결과 JSON을 이용해 eval_result.txt만 갱신
    # ------------------------------------------------------------------
    if args.reuse_result_json:
        result_path = args.reuse_result_json
        if not os.path.exists(result_path):
            print(f"[ERROR] Reuse mode: result JSON not found: {result_path}")
            sys.exit(1)
        with open(result_path, "r", encoding="utf-8") as f:
            dsc_result = json.load(f)
        # model_name / dataset_name 은 run_all_cmp.sh 에서 넘겨줌
        model_for_output = args.model_name
        dataset_for_output = args.dataset_name
        if not model_for_output or not dataset_for_output:
            print("[ERROR] Reuse mode requires --model_name and --dataset_name.")
            sys.exit(1)
        add_dsc_to_eval_result(
            results_dir=results_dir,
            model_name=model_for_output,
            dataset_name=dataset_for_output,
            stages_config=stages_config,
            dsc_result=dsc_result,
        )
        if not os.environ.get('RSC_EVAL_SINGLE_OUTPUT_PATH', '').strip() and "trial_accuracies" in dsc_result and dsc_result["trial_accuracies"]:
            add_dsc_to_eval_result_all(
                results_dir=results_dir,
                model_name=model_for_output,
                dataset_name=dataset_for_output,
                stages_config=stages_config,
                trial_accuracies=dsc_result["trial_accuracies"],
                trial_tokens=dsc_result.get("trial_tokens"),
                trial_prepare=dsc_result.get("trial_prepare"),
                trial_samples=dsc_result.get("trial_samples"),
            )
        sys.exit(0)

    # 모델명과 데이터셋명이 제공되면 자동으로 경로 생성
    if args.model_name and args.dataset_name:
        input_path = get_sc_sample_path(args.model_name, args.dataset_name, results_dir, stages_config)
        print(f"Auto-generated SC sample path: {input_path}")
        # DSC용 경로도 동일한 구조로 생성 (필요시 수정)
        dsc_input_path = input_path  # 또는 별도 경로 생성
        # eval_path도 자동 생성 (배치 평가 결과 파일)
        # 모델명은 HF ID 그대로 사용 (경로 유지)
        model_safe = args.model_name.replace("/", "_")
        temp_str = "0"
        eval_path = os.path.join(
            results_dir,
            args.model_name,
            f"stages-{stages_config}",
            "dsc_eval",
            f"{args.dataset_name}_{model_safe}_eval_strategy_random_prompt_short_batchsize_8_iter_5_n_1_temp_{temp_str}.jsonl"
        )
        print(f"Auto-generated eval path: {eval_path}")
    else:
        input_path = args.input_path
        dsc_input_path = args.dsc_input_path
        eval_path = args.eval_path

    n = args.n
    subset_n = args.subset_n
    repeat = args.repeat
    num_trials = args.num_trials
    bootstrap_seed = args.bootstrap_seed
    
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
    # model_name은 전체 HF ID를 직접 사용
    tokenizer_model_name = args.model_name if args.model_name else model_for_output
    
    # 모델명이 전체 HF ID인 경우 그대로 사용, 아니면 기본값 사용
    if tokenizer_model_name and "/" in tokenizer_model_name:
        # 전체 HF ID인 경우
        model_id = tokenizer_model_name
    elif tokenizer_model_name:
        # 짧은 별칭인 경우 (하위 호환성)
        model_path_map = {
            "gemma3_4b": "google/gemma-3-4b-it",
            "qwen2.5_7b": "Qwen/Qwen2.5-7B-Instruct",
            "qwen2.5_3b": "Qwen/Qwen2.5-3B-Instruct",
            "llama3.1_8b": "meta-llama/Llama-3.1-8B-Instruct"
        }
        if tokenizer_model_name in model_path_map:
            model_id = model_path_map[tokenizer_model_name]
        else:
            print(f"[WARN] Unknown model name: {tokenizer_model_name}. Using default Mistral tokenizer.")
            model_id = "mistralai/Mistral-7B-v0.1"
    else:
        print(f"[WARN] Model name not provided. Using default Mistral tokenizer.")
        model_id = "mistralai/Mistral-7B-v0.1"
    
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
    elif dataset in ["MMLU-Pro", "MMLU-Pro_full", "GPQA-Diamond"]:
        # MMLU-Pro 계열과 GPQA-Diamond는 common과 유사한 형식
        prompt = get_common_prompt()
    else:
        prompt = get_strategy_prompt()

    aggregator = SCAggregator(dataset_name=dataset)

    for line in data:
        tem = json.loads(line)
        pre.append(tem)
        pre_answer.append(
            normalize_answer_eval(tem.get("answer", tem.get("output", "")))
        )

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

        predict_batch = [
            normalize_answer_eval(aggregator.extract_answer(item))
            for item in shu_batch
        ]
        for item in predict_batch:
            if item == '':
                unmatch_count += 1
        pre_batch.append(deepcopy(predict_batch))

    print("unmatch count={}".format(unmatch_count))

    for line in greedy_data:
        tem = json.loads(line)
        greedy_pre.append(tem)
        greedy_pre_answer.append(
            normalize_answer_eval(tem.get("answer", tem.get("output", "")))
        )

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

        predict_batch = [
            normalize_answer_eval(aggregator.extract_answer(item))
            for item in shu_batch
        ]

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
        total_length = len(pre)  # MATH 데이터셋의 경우 전체 문제 수

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

    # model_for_output이 전체 HF ID인지 확인하고 처리
    if args.model_name:
        model_for_output = args.model_name
    else:
        # eval_path에서 모델명 추출 (하위 호환성)
        eval_filename = os.path.basename(eval_path).replace(".jsonl", "")
        if "_eval_" in eval_filename:
            parts = eval_filename.split("_eval_")[0].split("_")
            if len(parts) >= 2:
                model_for_output = "_".join(parts[1:]) if len(parts) > 2 else parts[1]
            else:
                model_for_output = "unknown"
        else:
            model_for_output = "unknown"
    
    model_safe_for_output = model_for_output.replace("/", "_")

    # 출력 경로: experiments/results/{model_name}/stages-{n}/dsc_cmp/
    output_dir = os.path.join(
        results_dir,
        model_for_output,
        f"stages-{stages_config}",
        "dsc_cmp"
    )
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "{}_{}_n_{}_judge_window_{}_interplot_{}.json".format(
        dataset_for_output, model_safe_for_output, n, judge_window_size, interplot_size
    ))
    print(f"\n{'='*60}")
    print(f"Output path: {output_path}")
    print(f"{'='*60}")
    
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            result = json.load(f)
            f.close()
        print(f"Result already exists")
        print(result)
        exit(0)
    
    # DSC 계산 시작
    
    # DSC 관련 정확도 변수 초기화
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

    tokens_count = {
                    "dsc_hard": {"input": 0.0, "output": 0.0, "total": 0.0}, 
                    "dsc": {"input": 0.0, "output": 0.0, "total": 0.0},
                    "dsc_greedy": {"input": 0.0, "output": 0.0, "total": 0.0}, 
                    "dsc_window": {"input": 0.0, "output": 0.0, "total": 0.0},  
                    "dsc_pre": {"input": 0.0, "output": 0.0, "total": 0.0}}
    
    sample_count = {"dsc_hard": 0.0, "dsc": 0.0, 
                    "dsc_greedy": 0.0, "dsc_window": 0.0, "dsc_pre": 0.0}
    
    # ActSC/DynSC 평가 코드 제거 (DSC만 사용)
    
    # 부트스트래핑: 여러 trial의 결과를 수집하여 평균 계산
    trial_accuracies = {
        "dsc_hard": [], "dsc": [], "dsc_greedy": [], "dsc_window": [], "dsc_pre": []
    }
    trial_tokens = {
        "dsc_hard": {"input": [], "output": [], "total": []},
        "dsc": {"input": [], "output": [], "total": []},
        "dsc_greedy": {"input": [], "output": [], "total": []},
        "dsc_window": {"input": [], "output": [], "total": []},
        "dsc_pre": {"input": [], "output": [], "total": []}
    }
    trial_samples = {
        "dsc_hard": [], "dsc": [], "dsc_greedy": [], "dsc_window": [], "dsc_pre": []
    }
    trial_detail_acc = {
        "dsc_hard": {"easy": {"acc": [], "count": []}, "hard": {"acc": [], "count": []}},
        "dsc": {"easy": {"acc": [], "count": []}, "hard": {"acc": [], "count": []}},
        "dsc_greedy": {"easy": {"acc": [], "count": []}, "hard": {"acc": [], "count": []}},
        "dsc_window": {"easy": {"acc": [], "count": []}, "hard": {"acc": [], "count": []}},
        "dsc_pre": {"easy": {"acc": [], "count": []}, "hard": {"acc": [], "count": []}}
    }

    mmlu_category_by_index = None
    mmlu_dsc_per_trial: List[Dict[str, Any]] = []
    if dataset in ("MMLU-Pro", "MMLU-Pro_full"):
        mmlu_category_by_index = _load_mmlu_pro_category_by_index(dsc_input_path, len(pre))
    
    # 부트스트래핑 루프
    for trial in range(num_trials):
        # 각 trial마다 랜덤 시드 설정
        random.seed(bootstrap_seed + trial)
        np.random.seed(bootstrap_seed + trial)
        
        if num_trials > 1:
            print(f"\n[Trial {trial + 1}/{num_trials}]")

        _mmlu_dsc_cat_agg = _mmlu_dsc_cat_agg_new() if mmlu_category_by_index is not None else None
        
        # 각 trial마다 변수 초기화
        dsc_hard_accuracy = 0.0
        dsc_accuracy = 0.0
        dsc_greedy_accuracy = 0.0
        dsc_window_accuracy = 0.0
        dsc_pre_accuracy = 0.0
        
        # Easy/Hard별 정확도 초기화
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
        
        # Easy/Hard별 문제 수 초기화
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
        
        tokens_count = {
                        "dsc_hard": {"input": 0.0, "output": 0.0, "total": 0.0}, 
                        "dsc": {"input": 0.0, "output": 0.0, "total": 0.0},
                        "dsc_greedy": {"input": 0.0, "output": 0.0, "total": 0.0}, 
                        "dsc_window": {"input": 0.0, "output": 0.0, "total": 0.0},  
                        "dsc_pre": {"input": 0.0, "output": 0.0, "total": 0.0}}
        
        sample_count = {"dsc_hard": 0.0, "dsc": 0.0, 
                        "dsc_greedy": 0.0, "dsc_window": 0.0, "dsc_pre": 0.0}
    
        # Repeat 루프 제거: 1회만 실행 (부트스트래핑 루프 내에서 실행)
        if dataset != "MATH":
            index, eval, question, pre_batch, allocate_list, _, _ = split_easy_hard(eval_result, pre_batch, judge_window_size, window_size, n, pre_tokens)
            index_greedy, eval_greedy, question_greedy, pre_batch_greedy, allocate_list_greedy = split_easy_hard_greedy(eval_result, pre_batch, judge_window_size, window_size, n)
            window_list_actual = allocate_window(eval_result, pre_batch, n, interplot_size=interplot_size, window_size=window_size)
            
            # 비-MATH 데이터셋 계산 루프 (DSC만)
            for sc_num in [n]:
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
                    # 모든 메트릭(SC, AC, ESC, DSC, Ours)이 동일한 방식으로 시작:
                    # pool에서 subset_n(40)개를 먼저 샘플링하고, 그 안에서 각 방법론의 알고리즘을 적용
                    # 1) pool에서 subset_n(40)개를 먼저 샘플링
                    #    주의: stages-40에서 pool 크기와 subset_n이 같으면(=전부 사용)
                    #    random.sample(range(N), N)는 permutation(셔플)을 만들어 순서가 바뀝니다.
                    #    DSC의 stopping/window은 순서에 민감하므로, 전부 사용이면 랜덤 섞기를 건너뜁니다.
                    local_subset_n = min(subset_n, len(predict_batch)) if predict_batch else 0
                    if local_subset_n <= 0:
                        count += 1
                        continue

                    if local_subset_n >= len(predict_batch):
                        # identity: 원래 순서 그대로 사용
                        subset_predict_batch = predict_batch
                        subset_tokens_batch = {
                            "input": tokens_batch["input"],
                            "output": tokens_batch["output"]
                        }
                    else:
                        random.seed(bootstrap_seed + trial)
                        subset_indices = random.sample(range(len(predict_batch)), local_subset_n)
                        subset_predict_batch = [predict_batch[i] for i in subset_indices]
                        subset_tokens_batch = {
                            "input": tokens_batch["input"],
                            "output": [tokens_batch["output"][i] for i in subset_indices]
                        }
                    
                    # 2) subset(40개) 내에서 앞에서부터 sc_num개를 순서대로 사용 (DSC 알고리즘 적용)
                    # Ours와 동일하게 subset만 랜덤 샘플링하고, subset 내에서는 순서대로 사용
                    local_sc_num = min(sc_num, len(subset_predict_batch))
                    if local_sc_num <= 0:
                        count += 1
                        continue
                    
                    # subset에서 앞에서부터 sc_num개를 순서대로 사용 (랜덤 샘플링 없음)
                    # Ours와 동일하게 subset만 랜덤 샘플링하고, subset 내에서는 순서대로 사용
                    predict_batch = subset_predict_batch[:local_sc_num]
                    tokens_batch = {
                        "input": subset_tokens_batch["input"],
                        "output": subset_tokens_batch["output"][:local_sc_num]
                    }
                    dsc_hard_batch, dsc_hard_token_batch = DSC_hard(predict_batch, tokens_batch, window_size=window_size)
                    dsc_batch, dsc_token_batch = DSC(predict_batch, greedy_predict_batch, tokens_batch, greedy_tokens_batch, sub_allocate, window_size=window_size)
                    dsc_greedy_batch, dsc_greedy_token_batch = DSC(predict_batch, greedy_predict_batch, tokens_batch, greedy_tokens_batch, sub_allocate_greedy, window_size=window_size)
                    dsc_window_batch, dsc_window_token_batch = DSC_window(predict_batch, greedy_predict_batch, tokens_batch, greedy_tokens_batch, sub_allocate_greedy, sub_window_actual, extend_window_size=window_size)
                    dsc_pre_batch, dsc_pre_token_batch = DSC_pre(predict_batch, greedy_predict_batch, tokens_batch, greedy_tokens_batch, sub_allocate_greedy, sub_window_actual, extend_window_size=window_size)

                    dsc_hard_input_token, dsc_hard_output_token = tokens_batch["input"] * len(dsc_hard_token_batch["output"]) // window_size, sum(dsc_hard_token_batch["output"])
                    dsc_input_token, dsc_output_token = tokens_batch["input"] * max(1, len(dsc_token_batch["output"]) // window_size), sum(dsc_token_batch["output"])
                    dsc_greedy_input_token, dsc_greedy_output_token = tokens_batch["input"] * max(1, len(dsc_greedy_token_batch["output"]) // window_size), sum(dsc_greedy_token_batch["output"])
                    dsc_window_input_token, dsc_window_output_token = tokens_batch["input"] * max(1, len(dsc_window_token_batch["output"]) // sub_window_actual), sum(dsc_window_token_batch["output"])
                    dsc_pre_input_token, dsc_pre_output_token = tokens_batch["input"] * max(1,len(dsc_pre_token_batch["output"]) // sub_window_actual), sum(dsc_pre_token_batch["output"])

                    # 토큰 수 누적 (DSC만)
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

                    # 각 방법에서 사용된 샘플 수 누적 (DSC만)
                    sub_sample_count["dsc_hard"] += len(dsc_hard_batch)
                    sub_sample_count["dsc"] += len(dsc_batch)
                    sub_sample_count["dsc_greedy"] += len(dsc_greedy_batch)
                    sub_sample_count["dsc_window"] += len(dsc_window_batch)
                    sub_sample_count["dsc_pre"] += len(dsc_pre_batch)
                
                    num_problems_processed += 1

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

                    # DSC 방법들의 정확도 계산
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
                    # Easy/Hard 구분 (sub_window_actual 기준: window_size 이하면 easy, 초과면 hard)
                    # sub_window_actual이 0이거나 None인 경우를 처리 (기본값으로 window_size 사용)
                    actual_window_val = sub_window_actual if (sub_window_actual is not None and sub_window_actual > 0) else window_size
                    if actual_window_val <= window_size:
                        dsc_window_easy_list.append(dsc_window_correct)
                    else:
                        dsc_window_hard_list.append(dsc_window_correct)

                    dsc_pre_pre = max(dsc_pre_pre_dict, key=dsc_pre_pre_dict.get)
                    dsc_pre_correct = dsc_pre_pre == answer
                    dsc_pre_result_list.append(dsc_pre_correct)
                    # Easy/Hard 구분
                    if sub_allocate_greedy == 1:
                        dsc_pre_easy_list.append(dsc_pre_correct)
                    elif sub_allocate_greedy == -1:
                        dsc_pre_hard_list.append(dsc_pre_correct)
                    else:
                        # sub_allocate_greedy가 1도 -1도 아닌 경우 (이론적으로는 없어야 함)
                        # dsc_allocate_greedy 함수에서 모든 allocate[i] == 0을 -1로 변환하므로
                        # 하지만 안전을 위해 hard로 분류
                        dsc_pre_hard_list.append(dsc_pre_correct)

                    if _mmlu_dsc_cat_agg is not None:
                        _cat = (
                            mmlu_category_by_index[i]
                            if i < len(mmlu_category_by_index)
                            else "unknown"
                        )
                        z = _mmlu_dsc_cat_agg[_cat]
                        z["n"] += 1
                        z["dh_c"] += int(dsc_hard_correct)
                        z["dh_i"] += dsc_hard_input_token
                        z["dh_o"] += dsc_hard_output_token
                        z["dh_s"] += float(len(dsc_hard_batch))
                        z["d_c"] += int(dsc_correct)
                        z["d_i"] += dsc_input_token
                        z["d_o"] += dsc_output_token
                        z["d_s"] += float(len(dsc_batch))
                        z["dg_c"] += int(dsc_greedy_correct)
                        z["dg_i"] += dsc_greedy_input_token
                        z["dg_o"] += dsc_greedy_output_token
                        z["dg_s"] += float(len(dsc_greedy_batch))
                        z["dw_c"] += int(dsc_window_correct)
                        z["dw_i"] += dsc_window_input_token
                        z["dw_o"] += dsc_window_output_token
                        z["dw_s"] += float(len(dsc_window_batch))
                        z["dp_c"] += int(dsc_pre_correct)
                        z["dp_i"] += dsc_pre_input_token
                        z["dp_o"] += dsc_pre_output_token
                        z["dp_s"] += float(len(dsc_pre_batch))

            # 정확도 계산 (DSC만)
            if num_problems_processed > 0:
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

                if _mmlu_dsc_cat_agg is not None:
                    mmlu_dsc_per_trial.append(_finalize_mmlu_dsc_trial_agg(_mmlu_dsc_cat_agg))

            # 이번 trial의 결과를 수집 (non-MATH 데이터셋)
            if num_trials > 1:
                trial_accuracies["dsc_hard"].append(dsc_hard_accuracy)
                trial_accuracies["dsc"].append(dsc_accuracy)
                trial_accuracies["dsc_greedy"].append(dsc_greedy_accuracy)
                trial_accuracies["dsc_window"].append(dsc_window_accuracy)
                trial_accuracies["dsc_pre"].append(dsc_pre_accuracy)
                
                # 토큰 수 수집
                for key in ["dsc_hard", "dsc", "dsc_greedy", "dsc_window", "dsc_pre"]:
                    trial_tokens[key]["input"].append(tokens_count[key]["input"])
                    trial_tokens[key]["output"].append(tokens_count[key]["output"])
                    trial_tokens[key]["total"].append(tokens_count[key]["total"])
                    trial_samples[key].append(sample_count[key])
                
                # Easy/Hard별 정확도 수집 (count는 항상 수집)
                if 'dsc_hard_easy_accuracy' in locals():
                    trial_detail_acc["dsc_hard"]["easy"]["acc"].append(dsc_hard_easy_accuracy)
                    trial_detail_acc["dsc_hard"]["easy"]["count"].append(dsc_hard_easy_count)
                if 'dsc_hard_hard_accuracy' in locals():
                    trial_detail_acc["dsc_hard"]["hard"]["acc"].append(dsc_hard_hard_accuracy)
                    trial_detail_acc["dsc_hard"]["hard"]["count"].append(dsc_hard_hard_count)
                
                if 'dsc_easy_accuracy' in locals():
                    trial_detail_acc["dsc"]["easy"]["acc"].append(dsc_easy_accuracy)
                    trial_detail_acc["dsc"]["easy"]["count"].append(dsc_easy_count)
                if 'dsc_hard_accuracy_detail' in locals():
                    trial_detail_acc["dsc"]["hard"]["acc"].append(dsc_hard_accuracy_detail)
                    trial_detail_acc["dsc"]["hard"]["count"].append(dsc_hard_count)
                
                if 'dsc_greedy_easy_accuracy' in locals():
                    trial_detail_acc["dsc_greedy"]["easy"]["acc"].append(dsc_greedy_easy_accuracy)
                    trial_detail_acc["dsc_greedy"]["easy"]["count"].append(dsc_greedy_easy_count)
                if 'dsc_greedy_hard_accuracy' in locals():
                    trial_detail_acc["dsc_greedy"]["hard"]["acc"].append(dsc_greedy_hard_accuracy)
                    trial_detail_acc["dsc_greedy"]["hard"]["count"].append(dsc_greedy_hard_count)
                
                if 'dsc_window_easy_accuracy' in locals():
                    trial_detail_acc["dsc_window"]["easy"]["acc"].append(dsc_window_easy_accuracy)
                    trial_detail_acc["dsc_window"]["easy"]["count"].append(dsc_window_easy_count)
                if 'dsc_window_hard_accuracy' in locals():
                    trial_detail_acc["dsc_window"]["hard"]["acc"].append(dsc_window_hard_accuracy)
                    trial_detail_acc["dsc_window"]["hard"]["count"].append(dsc_window_hard_count)
                
                if 'dsc_pre_easy_accuracy' in locals():
                    trial_detail_acc["dsc_pre"]["easy"]["acc"].append(dsc_pre_easy_accuracy)
                    trial_detail_acc["dsc_pre"]["easy"]["count"].append(dsc_pre_easy_count)
                if 'dsc_pre_hard_accuracy' in locals():
                    trial_detail_acc["dsc_pre"]["hard"]["acc"].append(dsc_pre_hard_accuracy)
                    trial_detail_acc["dsc_pre"]["hard"]["count"].append(dsc_pre_hard_count)
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
                    # 모든 메트릭(SC, AC, ESC, DSC, Ours)이 동일한 방식으로 시작:
                    # pool에서 subset_n(40)개를 먼저 샘플링하고, 그 안에서 각 방법론의 알고리즘을 적용
                    # 1) pool에서 subset_n(40)개를 먼저 샘플링
                    #    주의: stages-40에서 pool 크기와 subset_n이 같으면(=전부 사용)
                    #    random.sample(range(N), N)는 permutation(셔플)을 만들어 순서가 바뀝니다.
                    #    DSC의 stopping/window은 순서에 민감하므로, 전부 사용이면 랜덤 섞기를 건너뜁니다.
                    local_subset_n = min(subset_n, len(predict_batch)) if predict_batch else 0
                    if local_subset_n <= 0:
                        count += 1
                        continue

                    if local_subset_n >= len(predict_batch):
                        # identity: 원래 순서 그대로 사용
                        subset_predict_batch = predict_batch
                        subset_tokens_batch = {
                            "input": tokens_batch["input"],
                            "output": tokens_batch["output"]
                        }
                    else:
                        random.seed(bootstrap_seed + trial)
                        subset_indices = random.sample(range(len(predict_batch)), local_subset_n)
                        subset_predict_batch = [predict_batch[i] for i in subset_indices]
                        subset_tokens_batch = {
                            "input": tokens_batch["input"],
                            "output": [tokens_batch["output"][i] for i in subset_indices]
                        }
                    
                    # 2) subset(40개) 내에서 앞에서부터 sc_num개를 순서대로 사용 (DSC 알고리즘 적용)
                    # Ours와 동일하게 subset만 랜덤 샘플링하고, subset 내에서는 순서대로 사용
                    local_sc_num = min(sc_num, len(subset_predict_batch))
                    if local_sc_num <= 0:
                        count += 1
                        continue
                    
                    # subset에서 앞에서부터 sc_num개를 순서대로 사용 (랜덤 샘플링 없음)
                    # Ours와 동일하게 subset만 랜덤 샘플링하고, subset 내에서는 순서대로 사용
                    predict_batch = subset_predict_batch[:local_sc_num]
                    tokens_batch = {
                        "input": subset_tokens_batch["input"],
                        "output": subset_tokens_batch["output"][:local_sc_num]
                    }
                    dsc_hard_batch, dsc_hard_token_batch = DSC_hard(predict_batch, tokens_batch, window_size=window_size)
                    dsc_batch, dsc_token_batch = DSC(predict_batch, greedy_predict_batch, tokens_batch, greedy_tokens_batch, sub_allocate, window_size=window_size)
                    dsc_greedy_batch, dsc_greedy_token_batch = DSC(predict_batch, greedy_predict_batch, tokens_batch, greedy_tokens_batch, sub_allocate_greedy, window_size=window_size)
                    dsc_window_batch, dsc_window_token_batch = DSC_window(predict_batch, greedy_predict_batch, tokens_batch, greedy_tokens_batch, sub_allocate_greedy, sub_window_actual, extend_window_size=window_size)
                    dsc_pre_batch, dsc_pre_token_batch = DSC_pre(predict_batch, greedy_predict_batch, tokens_batch, greedy_tokens_batch, sub_allocate_greedy, sub_window_actual, extend_window_size=window_size)

                    dsc_hard_input_token, dsc_hard_output_token = tokens_batch["input"] * len(dsc_hard_token_batch["output"]) // window_size, sum(dsc_hard_token_batch["output"])
                    dsc_input_token, dsc_output_token = tokens_batch["input"] * max(1, len(dsc_token_batch["output"]) // window_size), sum(dsc_token_batch["output"])
                    dsc_greedy_input_token, dsc_greedy_output_token = tokens_batch["input"] * max(1, len(dsc_greedy_token_batch["output"]) // window_size), sum(dsc_greedy_token_batch["output"])
                    dsc_window_input_token, dsc_window_output_token = tokens_batch["input"] * max(1, len(dsc_window_token_batch["output"]) // sub_window_actual), sum(dsc_window_token_batch["output"])
                    dsc_pre_input_token, dsc_pre_output_token = tokens_batch["input"] * max(1,len(dsc_pre_token_batch["output"]) // sub_window_actual), sum(dsc_pre_token_batch["output"])

                    # 토큰 수 누적 (DSC만)
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

                    # 각 방법에서 사용된 샘플 수 누적 (DSC만)
                    sub_sample_count["dsc_hard"] += len(dsc_hard_batch)
                    sub_sample_count["dsc"] += len(dsc_batch)
                    sub_sample_count["dsc_greedy"] += len(dsc_greedy_batch)
                    sub_sample_count["dsc_window"] += len(dsc_window_batch)
                    sub_sample_count["dsc_pre"] += len(dsc_pre_batch)
                
                    num_problems_processed += 1

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

                    # DSC 방법들의 정확도 계산
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
                    # Easy/Hard 구분 (sub_window_actual 기준: window_size 이하면 easy, 초과면 hard)
                    # sub_window_actual이 0이거나 None인 경우를 처리 (기본값으로 window_size 사용)
                    actual_window_val = sub_window_actual if (sub_window_actual is not None and sub_window_actual > 0) else window_size
                    if actual_window_val <= window_size:
                        dsc_window_easy_list.append(dsc_window_correct)
                    else:
                        dsc_window_hard_list.append(dsc_window_correct)

                    dsc_pre_pre = max(dsc_pre_pre_dict, key=dsc_pre_pre_dict.get)
                    dsc_pre_correct = dsc_pre_pre == answer
                    dsc_pre_result_list.append(dsc_pre_correct)
                    # Easy/Hard 구분
                    if sub_allocate_greedy == 1:
                        dsc_pre_easy_list.append(dsc_pre_correct)
                    elif sub_allocate_greedy == -1:
                        dsc_pre_hard_list.append(dsc_pre_correct)
                    else:
                        # sub_allocate_greedy가 1도 -1도 아닌 경우 (이론적으로는 없어야 함)
                        # dsc_allocate_greedy 함수에서 모든 allocate[i] == 0을 -1로 변환하므로
                        # 하지만 안전을 위해 hard로 분류
                        dsc_pre_hard_list.append(dsc_pre_correct)
                    # result_list_level[hard_level - 1].append(sc_pre == answer)
                    # result_list_subject[subject].append(sc_pre == answer)
                    # result_list_detail[subject][hard_level - 1].append(sc_pre == answer)

                # 정확도 계산 (DSC만)
                if num_problems_processed > 0:
                    dsc_hard_accuracy = np.array(dsc_hard_result_list).mean()
                    dsc_accuracy = np.array(dsc_result_list).mean()
                    dsc_greedy_accuracy = np.array(dsc_greedy_result_list).mean()
                    dsc_window_accuracy = np.array(dsc_window_result_list).mean()
                    dsc_pre_accuracy = np.array(dsc_pre_result_list).mean()
                
                # Easy/Hard별 정확도 계산
                if len(dsc_hard_easy_list) > 0:
                    dsc_hard_easy_accuracy = np.array(dsc_hard_easy_list).mean()
                    dsc_hard_easy_count = len(dsc_hard_easy_list)
                else:
                    dsc_hard_easy_accuracy = 0.0
                    dsc_hard_easy_count = 0
                if len(dsc_hard_hard_list) > 0:
                    dsc_hard_hard_accuracy = np.array(dsc_hard_hard_list).mean()
                    dsc_hard_hard_count = len(dsc_hard_hard_list)
                else:
                    dsc_hard_hard_accuracy = 0.0
                    dsc_hard_hard_count = 0
                
                if len(dsc_easy_list) > 0:
                    dsc_easy_accuracy = np.array(dsc_easy_list).mean()
                    dsc_easy_count = len(dsc_easy_list)
                else:
                    dsc_easy_accuracy = 0.0
                    dsc_easy_count = 0
                if len(dsc_hard_list) > 0:
                    dsc_hard_accuracy_detail = np.array(dsc_hard_list).mean()
                    dsc_hard_count = len(dsc_hard_list)
                else:
                    dsc_hard_accuracy_detail = 0.0
                    dsc_hard_count = 0
                
                if len(dsc_greedy_easy_list) > 0:
                    dsc_greedy_easy_accuracy = np.array(dsc_greedy_easy_list).mean()
                    dsc_greedy_easy_count = len(dsc_greedy_easy_list)
                else:
                    dsc_greedy_easy_accuracy = 0.0
                    dsc_greedy_easy_count = 0
                if len(dsc_greedy_hard_list) > 0:
                    dsc_greedy_hard_accuracy = np.array(dsc_greedy_hard_list).mean()
                    dsc_greedy_hard_count = len(dsc_greedy_hard_list)
                else:
                    dsc_greedy_hard_accuracy = 0.0
                    dsc_greedy_hard_count = 0
                
                if len(dsc_window_easy_list) > 0:
                    dsc_window_easy_accuracy = np.array(dsc_window_easy_list).mean()
                    dsc_window_easy_count = len(dsc_window_easy_list)
                else:
                    dsc_window_easy_accuracy = 0.0
                    dsc_window_easy_count = 0
                if len(dsc_window_hard_list) > 0:
                    dsc_window_hard_accuracy = np.array(dsc_window_hard_list).mean()
                    dsc_window_hard_count = len(dsc_window_hard_list)
                else:
                    dsc_window_hard_accuracy = 0.0
                    dsc_window_hard_count = 0
                
                if len(dsc_pre_easy_list) > 0:
                    dsc_pre_easy_accuracy = np.array(dsc_pre_easy_list).mean()
                    dsc_pre_easy_count = len(dsc_pre_easy_list)
                else:
                    dsc_pre_easy_accuracy = 0.0
                    dsc_pre_easy_count = 0
                if len(dsc_pre_hard_list) > 0:
                    dsc_pre_hard_accuracy = np.array(dsc_pre_hard_list).mean()
                    dsc_pre_hard_count = len(dsc_pre_hard_list)
                else:
                    dsc_pre_hard_accuracy = 0.0
                    dsc_pre_hard_count = 0

                    # 토큰 수와 샘플 수 계산 (문제당 평균)
                    for key in tokens_count.keys():
                        if key in sub_tokens_count:
                            tokens_count[key]["input"] = sub_tokens_count[key]["input"] / num_problems_processed
                            tokens_count[key]["output"] = sub_tokens_count[key]["output"] / num_problems_processed
                            tokens_count[key]["total"] = sub_tokens_count[key]["total"] / num_problems_processed
                
                    for key in sample_count.keys():
                        if key in sub_sample_count:
                            sample_count[key] = sub_sample_count[key] / num_problems_processed

                # 이번 trial의 결과를 수집
                if num_trials > 1:
                    trial_accuracies["dsc_hard"].append(dsc_hard_accuracy)
                    trial_accuracies["dsc"].append(dsc_accuracy)
                    trial_accuracies["dsc_greedy"].append(dsc_greedy_accuracy)
                    trial_accuracies["dsc_window"].append(dsc_window_accuracy)
                    trial_accuracies["dsc_pre"].append(dsc_pre_accuracy)
                
                    # 토큰 수 수집
                    for key in ["dsc_hard", "dsc", "dsc_greedy", "dsc_window", "dsc_pre"]:
                        trial_tokens[key]["input"].append(tokens_count[key]["input"])
                        trial_tokens[key]["output"].append(tokens_count[key]["output"])
                        trial_tokens[key]["total"].append(tokens_count[key]["total"])
                        trial_samples[key].append(sample_count[key])
                
                # Easy/Hard별 정확도 수집 (count는 항상 수집)
                if 'dsc_hard_easy_accuracy' in locals():
                    trial_detail_acc["dsc_hard"]["easy"]["acc"].append(dsc_hard_easy_accuracy)
                    trial_detail_acc["dsc_hard"]["easy"]["count"].append(dsc_hard_easy_count)
                if 'dsc_hard_hard_accuracy' in locals():
                    trial_detail_acc["dsc_hard"]["hard"]["acc"].append(dsc_hard_hard_accuracy)
                    trial_detail_acc["dsc_hard"]["hard"]["count"].append(dsc_hard_hard_count)
                
                if 'dsc_easy_accuracy' in locals():
                    trial_detail_acc["dsc"]["easy"]["acc"].append(dsc_easy_accuracy)
                    trial_detail_acc["dsc"]["easy"]["count"].append(dsc_easy_count)
                if 'dsc_hard_accuracy_detail' in locals():
                    trial_detail_acc["dsc"]["hard"]["acc"].append(dsc_hard_accuracy_detail)
                    trial_detail_acc["dsc"]["hard"]["count"].append(dsc_hard_count)
                
                if 'dsc_greedy_easy_accuracy' in locals():
                    trial_detail_acc["dsc_greedy"]["easy"]["acc"].append(dsc_greedy_easy_accuracy)
                    trial_detail_acc["dsc_greedy"]["easy"]["count"].append(dsc_greedy_easy_count)
                if 'dsc_greedy_hard_accuracy' in locals():
                    trial_detail_acc["dsc_greedy"]["hard"]["acc"].append(dsc_greedy_hard_accuracy)
                    trial_detail_acc["dsc_greedy"]["hard"]["count"].append(dsc_greedy_hard_count)
                
                if 'dsc_window_easy_accuracy' in locals():
                    trial_detail_acc["dsc_window"]["easy"]["acc"].append(dsc_window_easy_accuracy)
                    trial_detail_acc["dsc_window"]["easy"]["count"].append(dsc_window_easy_count)
                if 'dsc_window_hard_accuracy' in locals():
                    trial_detail_acc["dsc_window"]["hard"]["acc"].append(dsc_window_hard_accuracy)
                    trial_detail_acc["dsc_window"]["hard"]["count"].append(dsc_window_hard_count)
                
                if 'dsc_pre_easy_accuracy' in locals():
                    trial_detail_acc["dsc_pre"]["easy"]["acc"].append(dsc_pre_easy_accuracy)
                    trial_detail_acc["dsc_pre"]["easy"]["count"].append(dsc_pre_easy_count)
                if 'dsc_pre_hard_accuracy' in locals():
                    trial_detail_acc["dsc_pre"]["hard"]["acc"].append(dsc_pre_hard_accuracy)
                    trial_detail_acc["dsc_pre"]["hard"]["count"].append(dsc_pre_hard_count)

    # 부트스트래핑 결과 평균 계산
    if num_trials > 1 and len(trial_accuracies["dsc_hard"]) > 0:
        # 정확도 평균
        dsc_hard_accuracy_final = np.mean(trial_accuracies["dsc_hard"])
        dsc_accuracy_final = np.mean(trial_accuracies["dsc"])
        dsc_greedy_accuracy_final = np.mean(trial_accuracies["dsc_greedy"])
        dsc_window_accuracy_final = np.mean(trial_accuracies["dsc_window"])
        dsc_pre_accuracy_final = np.mean(trial_accuracies["dsc_pre"])
        
        # 토큰 수 평균
        for key in ["dsc_hard", "dsc", "dsc_greedy", "dsc_window", "dsc_pre"]:
            tokens_count[key]["input"] = np.mean(trial_tokens[key]["input"])
            tokens_count[key]["output"] = np.mean(trial_tokens[key]["output"])
            tokens_count[key]["total"] = np.mean(trial_tokens[key]["total"])
            sample_count[key] = np.mean(trial_samples[key])
        
        # Easy/Hard별 정확도 평균
        if len(trial_detail_acc["dsc_hard"]["easy"]["acc"]) > 0:
            dsc_hard_easy_accuracy = np.mean(trial_detail_acc["dsc_hard"]["easy"]["acc"])
            # count도 부트스트래핑 평균으로 계산 (정수로 반올림)
            dsc_hard_easy_count = int(np.round(np.mean(trial_detail_acc["dsc_hard"]["easy"]["count"]))) if len(trial_detail_acc["dsc_hard"]["easy"]["count"]) > 0 else 0
        if len(trial_detail_acc["dsc_hard"]["hard"]["acc"]) > 0:
            dsc_hard_hard_accuracy = np.mean(trial_detail_acc["dsc_hard"]["hard"]["acc"])
            # count도 부트스트래핑 평균으로 계산 (정수로 반올림)
            dsc_hard_hard_count = int(np.round(np.mean(trial_detail_acc["dsc_hard"]["hard"]["count"]))) if len(trial_detail_acc["dsc_hard"]["hard"]["count"]) > 0 else 0
        
        if len(trial_detail_acc["dsc"]["easy"]["acc"]) > 0:
            dsc_easy_accuracy = np.mean(trial_detail_acc["dsc"]["easy"]["acc"])
            # count도 부트스트래핑 평균으로 계산 (정수로 반올림)
            dsc_easy_count = int(np.round(np.mean(trial_detail_acc["dsc"]["easy"]["count"]))) if len(trial_detail_acc["dsc"]["easy"]["count"]) > 0 else 0
        if len(trial_detail_acc["dsc"]["hard"]["acc"]) > 0:
            dsc_hard_accuracy_detail = np.mean(trial_detail_acc["dsc"]["hard"]["acc"])
            # count도 부트스트래핑 평균으로 계산 (정수로 반올림)
            dsc_hard_count = int(np.round(np.mean(trial_detail_acc["dsc"]["hard"]["count"]))) if len(trial_detail_acc["dsc"]["hard"]["count"]) > 0 else 0
        
        if len(trial_detail_acc["dsc_greedy"]["easy"]["acc"]) > 0:
            dsc_greedy_easy_accuracy = np.mean(trial_detail_acc["dsc_greedy"]["easy"]["acc"])
            # count도 부트스트래핑 평균으로 계산 (정수로 반올림)
            dsc_greedy_easy_count = int(np.round(np.mean(trial_detail_acc["dsc_greedy"]["easy"]["count"]))) if len(trial_detail_acc["dsc_greedy"]["easy"]["count"]) > 0 else 0
        if len(trial_detail_acc["dsc_greedy"]["hard"]["acc"]) > 0:
            dsc_greedy_hard_accuracy = np.mean(trial_detail_acc["dsc_greedy"]["hard"]["acc"])
            # count도 부트스트래핑 평균으로 계산 (정수로 반올림)
            dsc_greedy_hard_count = int(np.round(np.mean(trial_detail_acc["dsc_greedy"]["hard"]["count"]))) if len(trial_detail_acc["dsc_greedy"]["hard"]["count"]) > 0 else 0
        
        if len(trial_detail_acc["dsc_window"]["easy"]["acc"]) > 0:
            dsc_window_easy_accuracy = np.mean(trial_detail_acc["dsc_window"]["easy"]["acc"])
            # count도 부트스트래핑 평균으로 계산 (정수로 반올림)
            dsc_window_easy_count = int(np.round(np.mean(trial_detail_acc["dsc_window"]["easy"]["count"]))) if len(trial_detail_acc["dsc_window"]["easy"]["count"]) > 0 else 0
        if len(trial_detail_acc["dsc_window"]["hard"]["acc"]) > 0:
            dsc_window_hard_accuracy = np.mean(trial_detail_acc["dsc_window"]["hard"]["acc"])
            # count도 부트스트래핑 평균으로 계산 (정수로 반올림)
            dsc_window_hard_count = int(np.round(np.mean(trial_detail_acc["dsc_window"]["hard"]["count"]))) if len(trial_detail_acc["dsc_window"]["hard"]["count"]) > 0 else 0
        
        if len(trial_detail_acc["dsc_pre"]["easy"]["acc"]) > 0:
            dsc_pre_easy_accuracy = np.mean(trial_detail_acc["dsc_pre"]["easy"]["acc"])
            # count도 부트스트래핑 평균으로 계산 (정수로 반올림)
            dsc_pre_easy_count = int(np.round(np.mean(trial_detail_acc["dsc_pre"]["easy"]["count"]))) if len(trial_detail_acc["dsc_pre"]["easy"]["count"]) > 0 else 0
        if len(trial_detail_acc["dsc_pre"]["hard"]["acc"]) > 0:
            dsc_pre_hard_accuracy = np.mean(trial_detail_acc["dsc_pre"]["hard"]["acc"])
            # count도 부트스트래핑 평균으로 계산 (정수로 반올림)
            dsc_pre_hard_count = int(np.round(np.mean(trial_detail_acc["dsc_pre"]["hard"]["count"]))) if len(trial_detail_acc["dsc_pre"]["hard"]["count"]) > 0 else 0
        
        if num_trials > 1:
            print(f"\n[부트스트래핑 완료] {num_trials}번 trial의 평균값을 계산했습니다.")
    else:
        # 부트스트래핑 없이 단일 실행
        dsc_hard_accuracy_final = dsc_hard_accuracy
        dsc_accuracy_final = dsc_accuracy
        dsc_greedy_accuracy_final = dsc_greedy_accuracy
        dsc_window_accuracy_final = dsc_window_accuracy
        dsc_pre_accuracy_final = dsc_pre_accuracy
    
    # 토큰 수 계산 (DSC만)
    dsc_hard_input_token, dsc_hard_output_token, dsc_hard_total_token = tokens_count["dsc_hard"]["input"], tokens_count["dsc_hard"]["output"], tokens_count["dsc_hard"]["total"]
    dsc_input_token, dsc_output_token, dsc_total_token = tokens_count["dsc"]["input"], tokens_count["dsc"]["output"], tokens_count["dsc"]["total"]
    dsc_greedy_input_token, dsc_greedy_output_token, dsc_greedy_total_token = tokens_count["dsc_greedy"]["input"], tokens_count["dsc_greedy"]["output"], tokens_count["dsc_greedy"]["total"]
    dsc_window_input_token, dsc_window_output_token, dsc_window_total_token = tokens_count["dsc_window"]["input"], tokens_count["dsc_window"]["output"], tokens_count["dsc_window"]["total"]
    dsc_pre_input_token, dsc_pre_output_token, dsc_pre_total_token = tokens_count["dsc_pre"]["input"], tokens_count["dsc_pre"]["output"], tokens_count["dsc_pre"]["total"]

    # 각 방법에서 사용된 평균 샘플 수 계산 (DSC만)
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

    # 시행별 prepare (prepare는 trial마다 동일하므로 동일 값으로 리스트 생성)
    n_trials = num_trials if num_trials > 1 else 1
    trial_prepare = {
        "input": [float(prepare_input_tokens)] * n_trials,
        "output": [float(prepare_output_tokens)] * n_trials,
        "time": [float(prepare_time)] * n_trials,
    }

    # 출력 형식: Samples/Prepare/Inference/Acc (DSC만)
    print("DSC_hard: Samples={:.2f}, Prepare(Total={:.2f} avg, Input={:.2f} avg, Output={:.2f} avg), Inference(Input={:.2f}, Output={:.2f}, Total={:.2f}), Acc={:.4f}".format(
        dsc_hard_sample_num, prepare_tokens_avg_total, prepare_input_tokens_avg, prepare_output_tokens_avg, dsc_hard_input_token, dsc_hard_output_token, dsc_hard_total_token, dsc_hard_accuracy_final))
    print("DSC: Samples={:.2f}, Prepare(Total={:.2f} avg, Input={:.2f} avg, Output={:.2f} avg), Inference(Input={:.2f}, Output={:.2f}, Total={:.2f}), Acc={:.4f}".format(
        dsc_sample_num, prepare_tokens_avg_total, prepare_input_tokens_avg, prepare_output_tokens_avg, dsc_input_token, dsc_output_token, dsc_total_token, dsc_accuracy_final))
    print("DSC_greedy: Samples={:.2f}, Prepare(Total={:.2f} avg, Input={:.2f} avg, Output={:.2f} avg), Inference(Input={:.2f}, Output={:.2f}, Total={:.2f}), Acc={:.4f}".format(
        dsc_greedy_sample_num, prepare_tokens_avg_total, prepare_input_tokens_avg, prepare_output_tokens_avg, dsc_greedy_input_token, dsc_greedy_output_token, dsc_greedy_total_token, dsc_greedy_accuracy_final))
    print("DSC_window: Samples={:.2f}, Prepare(Total={:.2f} avg, Input={:.2f} avg, Output={:.2f} avg), Inference(Input={:.2f}, Output={:.2f}, Total={:.2f}), Acc={:.4f}".format(
        dsc_window_sample_num, prepare_tokens_avg_total, prepare_input_tokens_avg, prepare_output_tokens_avg, dsc_window_input_token, dsc_window_output_token, dsc_window_total_token, dsc_window_accuracy_final))
    print("DSC_pre: Samples={:.2f}, Prepare(Total={:.2f} avg, Input={:.2f} avg, Output={:.2f} avg), Inference(Input={:.2f}, Output={:.2f}, Total={:.2f}), Acc={:.4f}".format(
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
    
    result_dict = {
                   "dsc_hard": {
                       "sample": dsc_hard_sample_num,
                       "acc": dsc_hard_accuracy_final,
                       "inference": {"input": dsc_hard_input_token, "output": dsc_hard_output_token, "total": dsc_hard_total_token},
                       "detail_acc": {
                           "easy": {"count": dsc_hard_easy_count, "acc": dsc_hard_easy_acc},
                           "hard": {"count": dsc_hard_hard_count, "acc": dsc_hard_hard_acc}
                       }
                   },
                   "dsc": {
                       "sample": dsc_sample_num,
                       "acc": dsc_accuracy_final,
                       "inference": {"input": dsc_input_token, "output": dsc_output_token, "total": dsc_total_token},
                       "detail_acc": {
                           "easy": {"count": dsc_easy_count, "acc": dsc_easy_acc},
                           "hard": {"count": dsc_hard_count, "acc": dsc_hard_acc}
                       }
                   },
                   "dsc_greedy": {
                       "sample": dsc_greedy_sample_num,
                       "acc": dsc_greedy_accuracy_final,
                       "inference": {"input": dsc_greedy_input_token, "output": dsc_greedy_output_token, "total": dsc_greedy_total_token},
                       "detail_acc": {
                           "easy": {"count": dsc_greedy_easy_count, "acc": dsc_greedy_easy_acc},
                           "hard": {"count": dsc_greedy_hard_count, "acc": dsc_greedy_hard_acc}
                       }
                   },
                   "dsc_window": {
                       "sample": dsc_window_sample_num,
                       "acc": dsc_window_accuracy_final,
                       "inference": {"input": dsc_window_input_token, "output": dsc_window_output_token, "total": dsc_window_total_token},
                       "detail_acc": {
                           "easy": {"count": dsc_window_easy_count, "acc": dsc_window_easy_acc},
                           "hard": {"count": dsc_window_hard_count, "acc": dsc_window_hard_acc}
                       }
                   },
                   "dsc_pre": {
                       "sample": dsc_pre_sample_num,
                       "acc": dsc_pre_accuracy_final,
                       "inference": {"input": dsc_pre_input_token, "output": dsc_pre_output_token, "total": dsc_pre_total_token},
                       "detail_acc": {
                           "easy": {"count": dsc_pre_easy_count, "acc": dsc_pre_easy_acc},
                           "hard": {"count": dsc_pre_hard_count, "acc": dsc_pre_hard_acc}
                       }
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
                   },
                   "trial_accuracies": trial_accuracies if num_trials > 1 else {},
                   "trial_tokens": {k: {"input": [float(x) for x in v["input"]], "output": [float(x) for x in v["output"]], "total": [float(x) for x in v["total"]]} for k, v in trial_tokens.items()} if num_trials > 1 and trial_tokens else {},
                   "trial_prepare": trial_prepare,
                   "trial_samples": {k: [float(x) for x in v] for k, v in trial_samples.items()} if num_trials > 1 and trial_samples else {}
                }

    # 결과 저장 (output_dir은 이미 루프 시작 부분에서 생성됨)
    with open(output_path, "w") as f:
        json.dump(result_dict, f, indent="\t")
        f.close()
    print(f"Done! Result saved to {output_path}")
    
    # eval_result.txt 또는 단일 실험 시 eval_single_{seed}.txt에 DSC 결과 추가
    add_dsc_to_eval_result(
        results_dir=results_dir,
        model_name=args.model_name if args.model_name else model_for_output,
        dataset_name=dataset_for_output,
        stages_config=stages_config,
        dsc_result=result_dict
    )

    if dataset in ("MMLU-Pro", "MMLU-Pro_full") and mmlu_dsc_per_trial:
        try:
            from evaluation.mmlu_pro_category_aggregate import merge_mmlu_pro_category_section

            _mmlu_merged = _average_mmlu_dsc_category_trials(mmlu_dsc_per_trial)
            _mmlu_path = merge_mmlu_pro_category_section(
                os.path.join(results_dir, model_for_output, f"stages-{stages_config}"),
                "dsc",
                _mmlu_merged,
                dataset_name=dataset,
                extra_meta={
                    "source": "cmp_cal",
                    "judge_window_size": judge_window_size,
                    "interplot_size": interplot_size,
                },
            )
            print(f"[{dataset}] 카테고리별 DSC 집계 저장: {_mmlu_path}")
        except Exception as _mmlu_ex:
            print(f"Warning: {dataset} DSC category aggregate save failed: {_mmlu_ex}")

    # 단일 실험 모드에서는 eval_result_all.txt 수정하지 않음
    if not os.environ.get('RSC_EVAL_SINGLE_OUTPUT_PATH', '').strip() and num_trials > 1 and len(trial_accuracies["dsc_hard"]) > 0:
        add_dsc_to_eval_result_all(
            results_dir=results_dir,
            model_name=args.model_name if args.model_name else model_for_output,
            dataset_name=dataset_for_output,
            stages_config=stages_config,
            trial_accuracies=trial_accuracies,
            trial_tokens=trial_tokens,
            trial_prepare=trial_prepare,
            trial_samples=trial_samples,
        )