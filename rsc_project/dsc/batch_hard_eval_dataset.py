import json
import time
import sys
import os
from os import PathLike
import gzip
import math
import re
from copy import deepcopy
import random
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from collections import Counter
# from permsc import KemenyOptimalAggregator, sum_kendall_tau, ranks_from_preferences
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# rsc_project 모듈 import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from evaluation.datasets import DatasetLoader
from evaluation.metrics import normalize_answer_eval
from rsc.aggregator import SCAggregator
# _strip_string은 evaluation.metrics에서 직접 import (private이지만 필요)
try:
    from evaluation.metrics import _strip_string
except ImportError:
    # fallback: 직접 정의 (evaluation.metrics의 로직 복사)
    def _strip_string(string):
        from evaluation.metrics import _fix_fracs, _fix_a_slash_b, _remove_right_units, _fix_sqrt
        string = string.replace("\n", "")
        string = string.replace("\\!", "")
        string = string.replace("\\\\", "\\")
        string = string.replace("tfrac", "frac")
        string = string.replace("dfrac", "frac")
        string = string.replace("\\left", "")
        string = string.replace("\\right", "")
        string = string.replace("^{\\circ}", "")
        string = string.replace("^\\circ", "")
        string = string.replace("\\$", "")
        string = _remove_right_units(string)
        string = string.replace("\\%", "")
        string = string.replace("\%", "")
        string = string.replace(" .", " 0.")
        string = string.replace("{.", "{0.")
        if len(string) == 0:
            return string
        if string[0] == ".":
            string = "0" + string
        if len(string.split("=")) == 2:
            if len(string.split("=")[0]) <= 2:
                string = string.split("=")[1]
        string = _fix_sqrt(string)
        string = string.replace(" ", "")
        string = _fix_fracs(string)
        if string == "0.5":
            string = "\\frac{1}{2}"
        string = _fix_a_slash_b(string)
        return string

# _strip_string 관련 함수들은 evaluation.metrics에서 import하여 사용
# 기존 중복 코드 제거, evaluation.metrics의 함수 활용


from typing import Dict, Iterable

def write_jsonl(filename: str, data: Iterable[Dict], append: bool = False):
    """
    Writes an iterable of dictionaries to jsonl
    """
    if append:
        mode = "ab"
    else:
        mode = "wb"
    filename = os.path.expanduser(filename)
    if filename.endswith(".gz"):
        with open(filename, mode) as fp:
            with gzip.GzipFile(fileobj=fp, mode="wb") as gzfp:
                for x in data:
                    gzfp.write((json.dumps(x) + "\n").encode("utf-8"))
    else:
        with open(filename, mode) as fp:
            for x in data:
                fp.write((json.dumps(x) + "\n").encode("utf-8"))


class Get:
    """
    로컬/오픈소스 LLM을 사용해서 난이도 랭킹을 생성하는 클래스.

    기존 OpenAI GPT API 호출을 모두 제거하고, vLLM을 사용합니다.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        gpu_memory_utilization: float = 0.9,
    ):
        """
        model_name: 전체 HF 모델 ID (예: 'Qwen/Qwen2.5-7B-Instruct', 'google/gemma-3-4b-it')
        """
        self.model_name = model_name
        self.hf_id = model_name

        # vLLM 모델 로드
        if "gemma" in self.hf_id.lower():
            dtype = "bfloat16"
        else:
            dtype = "float16"
        self.llm = LLM(
            model=self.hf_id,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=16384,
        )
        # tokenizer도 로드 (chat template 사용을 위해)
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_id)

    def calc(
        self,
        questions,
        instruction: str = "",
        temp: float = 0,
        top_p: float = 0.9,
        n: int = 1,
        model=None,
        rank_prompt: str = "short",
        task: str = "",
    ):
        """
        questions: 리스트[str]
        temp: temperature
        n: 샘플 개수 (보통 1)
        model 인자는 더 이상 사용하지 않고, __init__ 에서 선택한 self.generator 를 사용.
        
        Returns:
            리스트[dict]: 각 dict는 {"generated_text": str, "input_tokens": int, "output_tokens": int, "time": float} 형태
        """

        if rank_prompt == "long":
            instruction = (
                "You will be given a batch of {} questions. Your task is to rank them from easy to hard "
                "based on their difficulty level. You should carefully horizontally compare the given questions "
                "in order to assign a suitable ranking place to each question. Please make sure you read and "
                "understand these instructions carefully. Please keep this document open while reviewing, and "
                "refer to it as needed. Questions to be evaluated:\n"
            ).format(len(questions))
            post_instruction = (
                "Evaluation Form (Answer by starting with \"I will do my best to provide individual analysis for each "
                "sample. Analysis:\" to analyze the given questions  as concise as possible (Attention: Don't give "
                "your ranking during this step). After analyzing all the questions, please give all the ranking place "
                "(from easy to hard) in order following the template "
                "\"Ranking: [Q{number of the easiest question},...,Q{number of the hardest question}]\".- Difficulty:"
            )
        else:
            instruction = (
                "Your task is to rank the given questions from easy to hard based on their difficulty level. "
                "Questions to be evaluated:\n"
            )
            post_instruction = (
                "The output format should be a comma-separated list containing the Q{number of corresponding question}. "
                "Do not give any explain. Difficulty Ranking result (from easy to hard):"
            )

        # 단일 프롬프트 문자열 생성
        query = ""
        for i, q in enumerate(questions, start=1):
            query += f"Q{i}: {q}\n"

        # chat template 사용 (instruct 모델의 경우)
        try:
            # messages 형식으로 변환
            messages = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": query + post_instruction}
            ]
            # chat template 적용
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except:
            # chat template이 없는 경우 단순 텍스트 사용
            prompt_text = instruction + query + post_instruction

        # 입력 토큰 수 계산
        # RSC와 동일하게 special tokens 제외
        input_tokens = len(self.tokenizer.encode(prompt_text, add_special_tokens=False))

        # 오류시 재시도
        count = 0
        while True:
            try:
                start_time = time.time()
                sampling_params = SamplingParams(
                    n=n,
                    temperature=temp,
                    top_p=top_p,
                    max_tokens=256,
                )
                outputs = self.llm.generate([prompt_text], sampling_params)
                elapsed_time = time.time() - start_time

                # vLLM 반환 처리
                res = []
                request_output = outputs[0]
                prompt_tokens = len(request_output.prompt_token_ids) if request_output.prompt_token_ids else input_tokens
                for out in request_output.outputs:
                    generated_text = out.text
                    output_tokens = len(out.token_ids) if out.token_ids is not None else len(
                        self.tokenizer.encode(generated_text, add_special_tokens=False)
                    )
                    res.append(
                        {
                            "generated_text": generated_text,
                            "input_tokens": prompt_tokens,
                            "output_tokens": output_tokens,
                            "time": elapsed_time / max(1, n),  # 각 시퀀스당 평균 시간
                        }
                    )
                return res
            except Exception as e:
                if count > 5:
                    print("Generation failed after retries:", e)
                    return [{"generated_text": " ", "input_tokens": input_tokens, "output_tokens": 0, "time": 0.0}]
                print("An error occurred during generation:", e)
                count += 1
                time.sleep(18)

def extract_all_number(s):
    # Q1, Q2, Q3 형식 또는 단순 숫자 형식 모두 지원
    numbers = re.findall(r'Q(\d+)', s)
    if numbers:
        return [int(number) for number in numbers]
    # Q가 없는 경우, 숫자만 추출 (1, 2, 3 형식)
    numbers = re.findall(r'\b(\d+)\b', s)
    if numbers:
        # 1부터 시작하는 숫자만 필터링 (질문 번호)
        filtered = [int(n) for n in numbers if 1 <= int(n) <= 100]  # 최대 100개 질문 가정
        if len(filtered) > 0:
            return filtered
    return []
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


def extract_math_answer(pred_str):
    # if ('The answer is ' in pred_str):
    #    pred = pred_str.split('The answer is ')[-1].strip()
    # elif ('the answer is ' in pred_str):
    #    pred = pred_str.split('the answer is ')[-1].strip()
    # elif 'boxed' in pred_str:
    #    ans = pred_str.split('boxed')[-1]
    #    if (len(ans) and ans[0] == '{'):
    #        stack = 1
    #        a = ''
    #        for c in ans[1:]:
    #            if (c == '{'):
    #                stack += 1
    #                a += c
    #            elif (c == '}'):
    #                stack -= 1
    #                if (stack == 0): break
    #                a += c
    #            else:
    #                a += c
    #    else:
    #        a = ans.split('$')[0].strip()
    #    a = _strip_string(a)
    #    pred = a

    # else:
    #    pattern = '-?\d*\.?\d+'
    #    pred = re.findall(pattern, pred_str)
    #    if (len(pred) >= 1):
    #        # print(pred_str)
    #        pred = pred[-1]
    #    else:
    #        pred = ''
    # if pred != "":
    #    if pred[-1] == ".":
    #        pred = pred[:-1]
    #    if pred[-1] == "/":
    #        pred = pred[:-1]
    # pred = _strip_string(pred)
    try:
        if 'boxed' in pred_str:
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
        elif ('the answer is ' in pred_str):
            pred = pred_str.split('the answer is ')[-1].strip()
        elif ('The answer is ' in pred_str):
            pred = pred_str.split('The answer is ')[-1].strip()
        else:
            pattern = '-?\d*\.?\d+'
            pred = re.findall(pattern, pred_str)
            if (len(pred) >= 1):
                pred = pred[-1]
            else:
                pred = ''
            if pred != "":
                if pred[-1] == ".":
                    pred = pred[:-1]
    except:
        pred = ""
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


def find_math_answer(s):
    assert ('boxed' in s)
    # s = s.replace(",", "")
    ans = s.split('boxed')[-1]
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

    if "\\text{" in a:
        a = re.search(r'\\text\{(.*?)\}', a)
        if a:
            a = a.group(1)

    return a


def ESC(input_list):
    esc_count = 1
    esc_num = 0
    while True:
        if esc_num + 4 >= len(input_list):
            break
        sub = input_list[esc_num]
        flag = 1
        for i in range(1, 5):
            if sub != input_list[esc_num+i]:
                flag = 0
                break

        if flag == 1:
            break
        else:
            esc_count += 1
            esc_num += 5

    return esc_count

def cal_spear(result_list1, result_list2):
    spearman = []
    pearson = []
    for i in range(len(result_list1)):
        pearson_corr, _ = pearsonr(result_list1[i], result_list2[i])
        spearman_corr, _ = spearmanr(result_list1[i], result_list2[i])
        spearman.append(spearman_corr)
        pearson.append(pearson_corr)

    return sum(spearman) / len(spearman), sum(pearson) / len(pearson)

def most_common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]


def sort_and_record(A):
    indexed_A = list(enumerate(A))
    sorted_A = sorted(indexed_A, key=lambda x: x[1])
    sorted_indexes = [i[0] for i in sorted_A]
    sorted_A = [i[1] for i in sorted_A]
    return sorted_A, sorted_indexes


def divide_into_batches(sorted_A, batch_size):
    avg_len = len(sorted_A) // batch_size
    out = [[] for i in range(batch_size)]
    index = 0
    while True:
        if index >= batch_size:
            break
        if index == batch_size - 1:
            out[index].extend(sorted_A[index*avg_len:])
        else:
            out[index].extend(sorted_A[index*avg_len:(index+1)*avg_len])

        index += 1

    return out


def batchify(input_list, batch_size, strategy="random", score_list=None):
    flag = True
    score = []
    for i in range(len(score_list)):
        try:
            score.append(sum(score_list[i])/len(score_list[i]))
        except:
            flag = False

    if flag == False:
        strategy = "random"

    if strategy == "random" or strategy=="permsc":
        index_list = list(range(len(input_list)))
        random.shuffle(index_list)
        batches = [index_list[i:i+batch_size] for i in range(0, len(input_list), batch_size)]
        out = []
        for j in range(len(batches)):
            batch = batches[j]
            sub = []
            if len(batch) < batch_size:
                for k in range(batch_size-len(batch)):
                    batch.append(batches[j-1][k])

            for i in batch:
                sub.append((i, input_list[i]))
            out.append(deepcopy(sub))
    else:
        sorted_score, sorted_indexes = sort_and_record(score)
        sorted_indexes_batchfy = divide_into_batches(sorted_indexes, batch_size)
        for i in range(len(sorted_indexes_batchfy)):
            random.shuffle(sorted_indexes_batchfy[i])

        out = []
        for j in range(len(sorted_indexes_batchfy[0])):
            sub = []
            for i in range(batch_size):
                sub.append((sorted_indexes_batchfy[i][j], input_list[sorted_indexes_batchfy[i][j]]))
            out.append(deepcopy(sub))

        index = 0
        for j in range(len(sorted_indexes_batchfy[0]), len(sorted_indexes_batchfy[-1])):
            sub = []
            for i in range(batch_size):
                if len(sorted_indexes_batchfy[i]) == len(sorted_indexes_batchfy[0]):
                    sub.append((sorted_indexes_batchfy[i][index], input_list[sorted_indexes_batchfy[i][index]]))
                else:
                    sub.append((sorted_indexes_batchfy[i][j], input_list[sorted_indexes_batchfy[i][j]]))
            out.append(deepcopy(sub))
            index += 1
        # batch diverse strategy
    
    return out

def mean(nums):
    return sum(nums) / len(nums)


def shuffle_list_n_times(input_list, n):
    shuffled_lists = []
    index_transforms = []

    for _ in range(n):
        index_list = list(range(len(input_list)))
        random.shuffle(index_list)
        index_transforms.append(index_list)

        shuffled_list = [input_list[i] for i in index_list]
        shuffled_lists.append(shuffled_list)

    return shuffled_lists, index_transforms

def process_task(k, Gen, perm_questions, temp, model, rank_prompt, dataset):
    gen_preference_result = Gen.calc(questions=perm_questions[k], temp=temp, n=1, model=model, rank_prompt=rank_prompt, task=dataset)[:1]
    gen_preference = gen_preference_result[0]
    # Handle both old format (string) and new format (dict)
    if isinstance(gen_preference, dict):
        generated_text = gen_preference["generated_text"]
    else:
        generated_text = gen_preference
    return extract_all_number(generated_text.split("Ranking:")[-1])

def remove_short_lists(input_list, index_list, length):
    out_list = []
    out_index_list = []
    for i in range(len(input_list)):
        if len(input_list[i]) == length:
            out_list.append(input_list[i])
            out_index_list.append(index_list[i])

    return out_list, out_index_list

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", 
                       help="Model name: Full HF model ID (e.g., Qwen/Qwen2.5-7B-Instruct, google/gemma-3-4b-it)")
    parser.add_argument("--dataset", type=str, default="MATH_all",
                       help="Dataset name: GSM8K, AIME_2024, AIME_2025, MATH500_test, MMLU-Pro, MMLU-Pro_full, GPQA-Diamond")
    parser.add_argument("--eval_dir", type=str, default="../eval",
                       help="Evaluation dataset directory (default: ../eval)")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="experiments/results",
        help="Results directory (default: experiments/results)",
    )
    parser.add_argument("--stages_config", type=str, default="40",
                       help="Stages configuration (default: 40)")
    parser.add_argument("--strategy", type=str, default="random", choices=["random", "diverse", "permsc"])
    parser.add_argument("--rank_prompt", type=str, default="short", choices=["short", "long"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--iteration", type=int, default=5)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--temp", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    args = parser.parse_args()
    
    Gen = Get(args.model, gpu_memory_utilization=args.gpu_memory_utilization)
    model = args.model
    temp = args.temp
    top_p = args.top_p
    n = args.n
    samples = []
    i = 0
    batch_size = args.batch_size
    iteration = args.iteration
    strategy = args.strategy
    dataset = args.dataset
    rank_prompt = args.rank_prompt
    # dataset_sub = dataset_sub[0]
    ori_sc_result = []
    asc_result = []

    label_all = []
    predict = []

    mismatch_count = 0
    count = 0
    error_count = 0
    # if dataset == "MATH":
    #     output_path = "result/{}_{}_gpt_{}_eval_batchsize_{}_iter_{}_n_{}_temp_{}.jsonl".format(dataset, dataset_sub, model, batch_size, iteration, n, temp)
    # elif dataset == "GSM8K":
    #     output_path = "result/{}_gpt_{}_eval_batchsize_{}_iter_{}_n_{}_temp_{}.jsonl".format(dataset, model, batch_size, iteration, n, temp)

    eval_dir = args.eval_dir
    results_dir = args.results_dir
    stages_config = args.stages_config
    
    # DatasetLoader를 사용하여 데이터셋 로드
    if dataset in ["GSM8K", "AIME_2024", "AIME_2025", "AMC23", "GPQA-Diamond", "MATH500_test", "MMLU-Pro", "MMLU-Pro_full"]:
        dataset_loader = DatasetLoader(dataset_name=dataset, eval_dir=eval_dir)
        data_list = dataset_loader.load()
        # jsonl 형식으로 변환
        data = [json.dumps(item) + "\n" for item in data_list]
    elif dataset in ["coin_flip", "last_letter", "strategy", "common", "MATH_all"]:
        # 기존 방식 유지 (로컬 파일)
        input_map = {
            "coin_flip": "coin_flip.jsonl",
            "last_letter": "last_letter.jsonl",
            "strategy": "strategy.jsonl",
            "common": "common.jsonl",
            "MATH_all": "MATH_all.jsonl",
        }
        input_path = input_map.get(dataset, f"{dataset}.jsonl")
        with open(input_path, "r") as f:
            data = f.readlines()
            f.close()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    def run_eval():
        pre = []
        answer = []
        pre_batch = []
        question = []
        error_count = 0

        for line in data:
            tem = json.loads(line)
            pre.append(tem)
            # 질문 필드 추출
            if dataset == "MATH_all":
                question.append(tem["problem"])
            elif dataset in ["GSM8K", "AIME_2024", "AIME_2025", "AMC23", "GPQA-Diamond", "MATH500_test", "MMLU-Pro", "MMLU-Pro_full"]:
                question.append(tem["question"])
            else:
                # coin_flip, last_letter, strategy, common 등의 경우
                question.append(tem.get("input", tem.get("question", "")))
            # 답변 필드 추출
            # 모든 데이터셋에 대해 동일한 정규화 로직 사용
            if dataset in ["GSM8K", "AIME_2024", "AIME_2025", "AMC23", "MATH500_test", "GPQA-Diamond", "MMLU-Pro", "MMLU-Pro_full", "MATH_all"]:
                answer.append(normalize_answer_eval(tem["answer"]))
            else:
                answer.append(normalize_answer_eval(tem.get("output", "")))
            # shu_batch = tem['completion']
            # predict_batch = [extract_gsm8k_answer(item) for item in shu_batch]
            # pre_batch.append(deepcopy(predict_batch))

        eval_all = [[] for i in range(len(pre))]
        # 난이도 분류(prepare) 토큰/시간 추적 - 모든 iteration 합계
        prepare_input_tokens_total = 0
        prepare_output_tokens_total = 0
        prepare_time_total = 0.0
        for iter_num in tqdm(range(iteration)):
            samples = []
            random_set = batchify(question, batch_size, strategy, eval_all)
            # pre_batch_sub = pre_batch[i*batch_size:(i+1)*batch_size]
            # pre_answer_sub = pre_answer[i*batch_size:(i+1)*batch_size]
            #
            # pre_sub = pre[i*batch_size:(i+1)*batch_size]
            # question_sub = question[i*batch_size:(i+1)*batch_size]
            error_gen = 0
            # 현재 iteration의 토큰/시간 추적
            prepare_input_tokens_iter = 0
            prepare_output_tokens_iter = 0
            prepare_time_iter = 0.0
            for i in tqdm(range(len(random_set))):
                # prepare 단계 전체 시간 측정 시작 (랭킹 생성부터 Easy/Hard 분류까지)
                batch_prepare_start = time.time()
                
                subset = random_set[i]
                subset_question = []
                subset_index = []
                for item in subset:
                    subset_question.append(item[1])
                    subset_index.append(item[0])
                if strategy=="permsc":
                    perm_questions, perm_index = shuffle_list_n_times(subset_question, n)
                    perm_preference, perm_preference_map = [], []
                    # 분류 토큰/시간 추적을 위해 순차 처리로 변경 (parallel은 토큰 추적이 어려움)
                    perm_preference_results = []
                    for k in range(len(perm_questions)):
                        gen_preference_result = Gen.calc(
                            questions=perm_questions[k],
                            temp=temp,
                            top_p=top_p,
                            n=1,
                            model=model,
                            rank_prompt=rank_prompt,
                            task=dataset,
                        )[:1]
                        perm_preference_results.append(gen_preference_result[0])
                        # 토큰 추적 (시간은 전체 배치 시간으로 측정)
                        if isinstance(gen_preference_result[0], dict):
                            prepare_input_tokens_iter += gen_preference_result[0].get("input_tokens", 0)
                            prepare_output_tokens_iter += gen_preference_result[0].get("output_tokens", 0)
                        gen_preference = gen_preference_result[0]
                        if isinstance(gen_preference, dict):
                            generated_text = gen_preference["generated_text"]
                        else:
                            generated_text = gen_preference
                        perm_preference.append(extract_all_number(generated_text.split("Ranking:")[-1]))

                    # parallel implement (토큰 추적 안 함)
                    # with ProcessPoolExecutor() as executor:
                    #     futures = [executor.submit(process_task, k, Gen, perm_questions, temp, model, rank_prompt, dataset) for k in range(len(perm_questions))]
                    # perm_preference = [future.result() for future in futures]

                    pre_len = len(perm_preference)
                    perm_preference, perm_index = remove_short_lists(perm_preference, perm_index, batch_size)
                    post_len = len(perm_preference)
                    error_gen += pre_len - post_len

                    for j in range(len(perm_preference)):
                        preference_completion_map_sub = [0 for k in range(len(perm_preference[j]))]
                        try:
                            for index in range(len(preference_completion_map_sub)):
                                preference_completion_map_sub[index] = perm_index[j][perm_preference[j][index] - 1] # turn 1-8 to 0-7
                            perm_preference_map.append(preference_completion_map_sub)
                        except:
                            error_count += 1

                    perm_preference_map = np.array(perm_preference_map)
                    preference_optimal = KemenyOptimalAggregator().aggregate(perm_preference_map, verbose=False)
                    rank_optimal = ranks_from_preferences(preference_optimal)
                    for j in range(len(rank_optimal)):
                        eval_all[subset_index[j]].append(rank_optimal[j])


                else:
                    rank_results = Gen.calc(
                        questions=subset_question,
                        temp=temp,
                        top_p=top_p,
                        n=n,
                        model=model,
                        rank_prompt=rank_prompt,
                        task=dataset,
                    )[:n]
                    # 새 형식(dict)에서 텍스트 추출 및 토큰 추적 (시간은 전체 배치 시간으로 측정)
                    rank = []
                    for rank_result in rank_results:
                        if isinstance(rank_result, dict):
                            prepare_input_tokens_iter += rank_result.get("input_tokens", 0)
                            prepare_output_tokens_iter += rank_result.get("output_tokens", 0)
                            rank.append(rank_result["generated_text"])
                        else:
                            rank.append(rank_result)
                    
                    # samples.append(dict(questions=subset_random, completion=rank, label=sub_label_random))
                    # write_jsonl("result/{}_{}_gpt_{}_eval_test_num_{}_n_{}_temp_{}.jsonl".format(dataset, dataset_sub, model, test_num, n, temp), samples)

                    rank_completion = []
                    for j in range(len(rank)):
                        # "Ranking:"이 있으면 그 뒤를 사용, 없으면 전체 텍스트 사용
                        if "Ranking:" in rank[j]:
                            text_to_parse = rank[j].split("Ranking:")[-1]
                        else:
                            text_to_parse = rank[j]
                        sub = extract_all_number(text_to_parse)
                        if len(sub) == batch_size:
                            rank_completion.append(sub)
                        else:
                            error_gen += 1
                            # 디버깅: 첫 번째 에러만 출력
                            if error_gen == 1 and iter_num == 0:
                                print(f"[DEBUG] Failed to extract ranking. Text: {rank[j][:200]}")
                                print(f"[DEBUG] Extracted numbers: {sub}, Expected count: {batch_size}")
                    rank_completion_map = []
                    for j in range(len(rank_completion)):
                        rank_completion_map_sub = [0 for k in range(len(rank_completion[j]))]
                        try:
                            for index in range(len(rank_completion_map_sub)):
                                rank_completion_map_sub[rank_completion[j][index] - 1] = index + 1
                            rank_completion_map.append(rank_completion_map_sub)
                        except:
                            error_count += 1

                    rank_completion_map = list(map(mean, zip(*rank_completion_map)))
                    # rank_completion_map = [item / n for item in rank_completion_map]

                    for j in range(len(rank_completion_map)):
                        eval_all[subset_index[j]].append(rank_completion_map[j])
                
                # prepare 단계 전체 시간 측정 종료 (랭킹 생성부터 Easy/Hard 분류까지)
                batch_prepare_end = time.time()
                prepare_time_iter += (batch_prepare_end - batch_prepare_start)

            error = 0
            eval_all_sub = []
            for tem in eval_all:
                try:
                    eval_all_sub.append(sum(tem) / len(tem))
                except:
                    eval_all_sub.append(0)
                    error += 1

            print("gen error:{}".format(error_gen))
            print("sum error:{}".format(error))
            # 현재 iteration의 시간을 총합에 누적
            prepare_input_tokens_total += prepare_input_tokens_iter
            prepare_output_tokens_total += prepare_output_tokens_iter
            prepare_time_total += prepare_time_iter
            # prepare 단계 시간 출력
            print(f"[Iteration {iter_num + 1}] Prepare time: {prepare_time_iter:.2f}s (this iteration), {prepare_time_total:.2f}s (total)")
            print(f"[Iteration {iter_num + 1}] Prepare tokens: input={prepare_input_tokens_iter}, output={prepare_output_tokens_iter} (this iteration)")
            print(f"[Iteration {iter_num + 1}] Prepare tokens: input={prepare_input_tokens_total}, output={prepare_output_tokens_total} (total)")
            # 난이도 분류(prepare) 토큰/시간 정보 추가 (모든 iteration 합계)
            # Step 1: Ranking 시간만 저장 (Step 2는 cmp_cal.py에서 추가됨)
            prepare_info = {
                "prepare_input_tokens": prepare_input_tokens_total,
                "prepare_output_tokens": prepare_output_tokens_total,
                "ranking_time": prepare_time_total,  # Step 1: 랭킹 생성 시간
                "prepare_time": prepare_time_total   # 총 시간 (Step 2 추가 전)
            }
            samples.append(dict(questions=question, eval=eval_all_sub, answer=answer, prepare_info=prepare_info))
            # 결과 저장 경로: experiments/results/{model_name}/stages-{n}/dsc_eval/
            model_safe = model.replace("/", "_")
            output_dir = os.path.join(results_dir, model, f"stages-{stages_config}", "dsc_eval")
            os.makedirs(output_dir, exist_ok=True)
            
            # temp 값을 정수로 변환하여 파일명 생성 (0.0 -> 0)
            temp_str = str(int(temp)) if temp == int(temp) else str(temp)
            output_file = os.path.join(output_dir, 
                "{}_{}_eval_strategy_{}_prompt_{}_batchsize_{}_iter_{}_n_{}_temp_{}.jsonl".format(
                    dataset, model_safe, strategy, rank_prompt, batch_size,
                    iter_num + 1, n, temp_str))
            write_jsonl(output_file, samples)
            # samples.append(dict(questions=question_sub, completion=rank))
            # write_jsonl("result/{}_gpt_{}_eval_batchsize_{}_n_{}_temp_{}.jsonl".format(dataset, model, batch_size, n, temp), samples)

        print("error count:{}".format(error_count))
        print("=" * 50)
        print(f"[FINAL] Total prepare time: {prepare_time_total:.2f}s ({prepare_time_total/60:.2f} minutes)")
        print(f"[FINAL] Total prepare tokens: input={prepare_input_tokens_total}, output={prepare_output_tokens_total}")
        print("=" * 50)
        print("done!")



    if dataset == "MATH":
        input_path = "MATH.jsonl"
        ori_data = []
        final_eval = []
        index = 0
        with open(input_path, "r") as f:
            data = f.readlines()
            f.close()

        type_dict = {}

        for line in data:
            tem = json.loads(line)
            ori_data.append(deepcopy(tem))
            type = tem["subject"].replace(" ", "_")
            if type not in type_dict.keys():
                type_dict[type] = {"hard_level": [], "problem": [], "answer": [], "eval_all":[], "index":[]}

            type_dict[type]["problem"].append(tem["problem"])
            type_dict[type]["answer"].append(tem["answer"])
            # shu_batch = tem['generated_answer']
            type_dict[type]["hard_level"].append(tem["level"])
            type_dict[type]["index"].append(index)

            # predict_batch = [extract_math_answer(item) for item in shu_batch]
            # type_dict[type]["generated_answer"].append(deepcopy(predict_batch))
            type_dict[type]["eval_all"].append([])
            final_eval.append(0)
            index += 1


        for key in type_dict.keys():
            print(key)
            count = 0
            error_count = 0
            answer = type_dict[key]["answer"]
            # pre_batch = type_dict[key]["generated_answer"]
            question = type_dict[key]["problem"]
            eval_all = type_dict[key]["eval_all"]
            hard_level = type_dict[key]["hard_level"]
            sub_index = type_dict[key]["index"]

            # 난이도 분류(prepare) 토큰/시간 추적 - 모든 iteration 합계
            prepare_input_tokens_total = 0
            prepare_output_tokens_total = 0
            prepare_time_total = 0.0
            for iter_num in tqdm(range(iteration)):
                samples = []
                random_set = batchify(question, batch_size, strategy, eval_all)
                # pre_batch_sub = pre_batch[i*batch_size:(i+1)*batch_size]
                # pre_answer_sub = pre_answer[i*batch_size:(i+1)*batch_size]
                #
                # pre_sub = pre[i*batch_size:(i+1)*batch_size]
                # question_sub = question[i*batch_size:(i+1)*batch_size]
                error_gen = 0
                # 현재 iteration의 토큰/시간 추적
                prepare_input_tokens_iter = 0
                prepare_output_tokens_iter = 0
                prepare_time_iter = 0.0
                for i in tqdm(range(len(random_set))):
                    # prepare 단계 전체 시간 측정 시작 (랭킹 생성부터 Easy/Hard 분류까지)
                    batch_prepare_start = time.time()
                    
                    subset = random_set[i]
                    subset_question = []
                    subset_index = []
                    for item in subset:
                        subset_question.append(item[1])
                        subset_index.append(item[0])

                    if strategy == "permsc":
                        perm_questions, perm_index = shuffle_list_n_times(subset_question, n)
                        perm_preference, perm_preference_map = [], []
                        # 분류 토큰/시간 추적을 위해 순차 처리로 변경
                        for k in range(len(perm_questions)):
                            gen_preference_result = Gen.calc(
                                questions=perm_questions[k],
                                temp=temp,
                                top_p=top_p,
                                n=1,
                                model=model,
                                rank_prompt=rank_prompt,
                                task=dataset,
                            )[:1]
                            gen_preference = gen_preference_result[0]
                            # 토큰 추적 (시간은 전체 배치 시간으로 측정)
                            if isinstance(gen_preference, dict):
                                prepare_input_tokens_iter += gen_preference.get("input_tokens", 0)
                                prepare_output_tokens_iter += gen_preference.get("output_tokens", 0)
                                generated_text = gen_preference["generated_text"]
                            else:
                                generated_text = gen_preference
                            perm_preference.append(extract_all_number(generated_text.split("Ranking:")[-1]))

                        # parallel implement (토큰 추적 안 함)
                        # with ProcessPoolExecutor() as executor:
                        #     futures = [
                        #         executor.submit(process_task, k, Gen, perm_questions, temp, model, rank_prompt, dataset)
                        #         for k in range(len(perm_questions))]
                        # perm_preference = [future.result() for future in futures]

                        pre_len = len(perm_preference)
                        perm_preference, perm_index = remove_short_lists(perm_preference, perm_index, batch_size)
                        post_len = len(perm_preference)
                        error_gen += pre_len - post_len

                        for j in range(len(perm_preference)):
                            preference_completion_map_sub = [0 for k in range(len(perm_preference[j]))]
                            try:
                                for index in range(len(preference_completion_map_sub)):
                                    preference_completion_map_sub[index] = perm_index[j][perm_preference[j][index] - 1]  # turn 1-8 to 0-7
                                perm_preference_map.append(preference_completion_map_sub)
                            except:
                                error_count += 1

                        perm_preference_map = np.array(perm_preference_map)
                        preference_optimal = KemenyOptimalAggregator().aggregate(perm_preference_map, verbose=False)
                        rank_optimal = ranks_from_preferences(preference_optimal)
                        for j in range(len(rank_optimal)):
                            eval_all[subset_index[j]].append(rank_optimal[j])

                    else:
                        rank_results = Gen.calc(
                            questions=subset_question,
                            temp=temp,
                            top_p=top_p,
                            n=n,
                            model=model,
                            rank_prompt=rank_prompt,
                            task=dataset,
                        )[:n]
                        # 새 형식(dict)에서 텍스트 추출 및 토큰 추적 (시간은 전체 배치 시간으로 측정)
                        rank = []
                        for rank_result in rank_results:
                            if isinstance(rank_result, dict):
                                prepare_input_tokens_iter += rank_result.get("input_tokens", 0)
                                prepare_output_tokens_iter += rank_result.get("output_tokens", 0)
                                rank.append(rank_result["generated_text"])
                            else:
                                rank.append(rank_result)
                        
                        # samples.append(dict(questions=subset_random, completion=rank, label=sub_label_random))
                        # write_jsonl("result/{}_{}_gpt_{}_eval_test_num_{}_n_{}_temp_{}.jsonl".format(dataset, dataset_sub, model, test_num, n, temp), samples)

                        rank_completion = []
                        for j in range(len(rank)):
                            sub = extract_all_number(rank[j].split("Ranking:")[-1])
                            if len(sub) == batch_size:
                                rank_completion.append(extract_all_number(rank[j].split("Ranking:")[-1]))
                            else:
                                error_gen += 1
                        rank_completion_map = []
                        for j in range(len(rank_completion)):
                            rank_completion_map_sub = [0 for k in range(len(rank_completion[j]))]
                            try:
                                for index in range(len(rank_completion_map_sub)):
                                    rank_completion_map_sub[rank_completion[j][index] - 1] = index
                                rank_completion_map.append(rank_completion_map_sub)
                            except:
                                error_count += 1

                        rank_completion_map = list(map(mean, zip(*rank_completion_map)))
                        # rank_completion_map = [item / n for item in rank_completion_map]

                        for j in range(len(rank_completion_map)):
                            eval_all[subset_index[j]].append(rank_completion_map[j])
                    
                    # prepare 단계 전체 시간 측정 종료 (랭킹 생성부터 Easy/Hard 분류까지)
                    batch_prepare_end = time.time()
                    prepare_time_iter += (batch_prepare_end - batch_prepare_start)

                error = 0
                eval_all_sub = []
                for tem in eval_all:
                    try:
                        eval_all_sub.append(sum(tem) / len(tem))
                    except:
                        eval_all_sub.append(0)
                        error += 1

                print("gen error:{}".format(error_gen))
                print("sum error:{}".format(error))
                # 현재 iteration의 시간을 총합에 누적
                prepare_input_tokens_total += prepare_input_tokens_iter
                prepare_output_tokens_total += prepare_output_tokens_iter
                prepare_time_total += prepare_time_iter
                # prepare 단계 시간 출력
                print(f"[Iteration {iter_num + 1}] Prepare time: {prepare_time_iter:.2f}s (this iteration), {prepare_time_total:.2f}s (total)")
                print(f"[Iteration {iter_num + 1}] Prepare tokens: input={prepare_input_tokens_iter}, output={prepare_output_tokens_iter} (this iteration)")
                print(f"[Iteration {iter_num + 1}] Prepare tokens: input={prepare_input_tokens_total}, output={prepare_output_tokens_total} (total)")
                # 난이도 분류(prepare) 토큰/시간 정보 추가 (모든 iteration 합계)
                # Step 1: Ranking 시간만 저장 (Step 2는 cmp_cal.py에서 추가됨)
                prepare_info = {
                    "prepare_input_tokens": prepare_input_tokens_total,
                    "prepare_output_tokens": prepare_output_tokens_total,
                    "ranking_time": prepare_time_total,  # Step 1: 랭킹 생성 시간
                    "prepare_time": prepare_time_total   # 총 시간 (Step 2 추가 전)
                }
                samples.append(dict(type=key, questions=question, eval=eval_all_sub, answer=answer, hard_level=hard_level, prepare_info=prepare_info))
                # 결과 저장 경로: experiments/results/{model_name}/stages-{n}/dsc_eval/
                model_safe = model.replace("/", "_")
                output_dir = os.path.join(results_dir, model, f"stages-{stages_config}", "dsc_eval")
                os.makedirs(output_dir, exist_ok=True)
                
                # temp 값을 정수로 변환하여 파일명 생성 (0.0 -> 0)
                temp_str = str(int(temp)) if temp == int(temp) else str(temp)
                output_file = os.path.join(output_dir,
                    "{}_{}_eval_strategy_{}_prompt_{}_batchsize_{}_iter_{}_n_{}_temp_{}.jsonl".format(
                        dataset, key, model_safe, strategy, rank_prompt, batch_size,
                        iter_num + 1, n, temp_str))
                write_jsonl(output_file, samples)
                # samples.append(dict(questions=question_sub, completion=rank))
                # write_jsonl("result/{}_gpt_{}_eval_batchsize_{}_n_{}_temp_{}.jsonl".format(dataset, model, batch_size, n, temp), samples)

            print("error count:{}".format(error_count))
            print("done!")
            for i in range(len(eval_all)):
                ori_data[sub_index[i]]["eval"] = sum(eval_all[i]) / len(eval_all[i])


        # 결과 저장 경로: experiments/results/{model_name}/stages-{n}/dsc_eval/
        model_safe = model.replace("/", "_")
        output_dir = os.path.join(results_dir, model, f"stages-{stages_config}", "dsc_eval")
        os.makedirs(output_dir, exist_ok=True)
        
        # temp 값을 정수로 변환하여 파일명 생성 (0.0 -> 0)
        temp_str = str(int(temp)) if temp == int(temp) else str(temp)
        out_path = os.path.join(output_dir,
            "{}_{}_eval_strategy_{}_prompt_{}_batchsize_{}_iter_{}_n_{}_temp_{}.jsonl".format(
                dataset, model_safe, strategy, rank_prompt, batch_size,
                iteration, n, temp_str))
        with open(out_path, 'w') as f:
            for item in ori_data:
                f.write(json.dumps(item) + '\n')
            f.close()

        print("saved!")



    #
    # if dataset == "MATH":
    #     if os.path.exists(output_path):
    #         with open(output_path, "r") as f:
    #             for line in f:
    #                 tem = json.loads(line)
    #
    #                 rank_completion = []
    #                 for j in range(len(tem["completion"])):
    #                     rank_completion.append(extract_all_number(tem["completion"][j].split("Ranking:")[-1]))
    #
    #                 rank_completion_map = []
    #                 for j in range(len(rank_completion)):
    #                     rank_completion_map_sub = [0 for k in range(len(rank_completion[j]))]
    #                     for index in range(len(rank_completion_map_sub)):
    #                         rank_completion_map_sub[rank_completion[j][index] - 1] = index + 1
    #                     rank_completion_map.append(rank_completion_map_sub)
    #
    #                 rank_completion_map = list(map(sum, zip(*rank_completion_map)))
    #                 rank_completion_map = [item/n for item in rank_completion_map]
    #
    #                 predict.append(rank_completion_map)
    #                 label_all.append(tem["label"])
    #             f.close()
    #     else:
    #         input_path = "MATH/{}".format(dataset_sub)
    #         json_files = [pos_json for pos_json in os.listdir(input_path) if pos_json.endswith('.json')]
    #
    #         data = []
    #         for json_file in json_files:
    #             json_file_path = os.path.join(input_path, json_file)
    #             with open(json_file_path, 'r') as j:
    #                 data.append(json.load(j))
    #         hard_level = {"1": [], "2": [], "3": [], "4": [], "5": []}
    #         for item in data:
    #             sub_level = item["level"][-1]
    #             hard_level[sub_level].append(item["problem"])
    #
    #
    #         for i in tqdm(range(repeat)):
    #             subset = []
    #             sub_label = []
    #             for key in ["1", "2", "3", "4", "5"]:
    #                 subset += random.sample(hard_level[key], test_num)
    #                 sub_label += [int(key) for j in range(test_num)]
    #
    #             random_indices = random.sample(range(len(subset)), len(subset))
    #             subset_random = [0 for j in range(len(subset))]
    #             sub_label_random = [0 for j in range(len(subset))]
    #
    #             for j in range(len(random_indices)):
    #                 subset_random[j] = subset[random_indices[j]]
    #                 sub_label_random[j] = sub_label[random_indices[j]]
    #
    #             rank = Gen.calc(questions=subset_random, temp=temp, n=n, model=model, rank_prompt=rank_prompt, task=dataset)[:n]
    #             samples.append(dict(questions=subset_random, completion=rank, label=sub_label_random))
    #             write_jsonl("result/{}_{}_gpt_{}_eval_test_num_{}_n_{}_temp_{}.jsonl".format(dataset, dataset_sub, model, test_num, n, temp), samples)

    run_eval()
    #
    #             rank_completion = []
    #             for j in range(len(rank)):
    #                 rank_completion.append(extract_all_number(rank[j].split("Ranking:")[-1]))
    #
    #             rank_completion_map = []
    #             for j in range(len(rank_completion)):
    #                 rank_completion_map_sub = [0 for k in range(len(rank_completion[j]))]
    #                 for index in range(len(rank_completion_map_sub)):
    #                     rank_completion_map_sub[rank_completion[j][index]-1] = index + 1
    #                 rank_completion_map.append(rank_completion_map_sub)
    #
    #             rank_completion_map = list(map(sum, zip(*rank_completion_map)))
    #             rank_completion_map = [item/n for item in rank_completion_map]
    #
    #             predict.append(rank_completion_map)
    #             label_all.append(sub_label_random)



        # spearman_avg, pearson_avg = cal_spear(predict, label_all)
        #
        # print("spearman: {}".format(spearman_avg))
        # print("pearson: {}".format(pearson_avg))