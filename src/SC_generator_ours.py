import argparse
import os
import pandas as pd
from tqdm import tqdm
import json
import re
from typing import Optional, List, Dict, Any
from utils import _strip_string
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from fewshot import (
    AIME_fewshot,
    MATH_fewshot,
    MMLU_fewshot,
    GPQA_fewshot,
    GSM8K_fewshot,
)
from confidence_extractor import (
    VLLMRunner,
    BINARY_FT01,
    generate_confidence,
    likelihood_to_confidence,
    find_answer_span,
)

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

EVAL_DIR = '../eval'
DATASET_FEWSHOT_MAP: Dict[str, List[Dict[str, Any]]] = {
    'AIME_2024': AIME_fewshot,
    'AIME_2025': AIME_fewshot,
    'MATH500_test': MATH_fewshot,
    'MMLU-Pro': MMLU_fewshot,
    'GPQA-Diamond': GPQA_fewshot,
    'GSM8K': GSM8K_fewshot,
}


def load_jsonl(file_path):
    """Load JSONL file and return as list of dictionaries"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def extract_math_answer(pred_str):
    if pred_str is None:
        return ''
    pred_str = str(pred_str).strip()
    if not pred_str:
        return ''
    
    if ('The answer is ' in pred_str):
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

def extract_answer_gpqa(content: Optional[str]) -> Optional[str]:
    """Extracts the final answer letter (A, B, C, or D) from the content."""
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

def extract_answer_mmlu_pro(content: Optional[str]) -> Optional[str]:
    """Extracts the final answer letter (A–J) from the model output."""
    if content is None:
        return None

    ANSWER_PATTERN = r"((?:[A-J]\b))"

    matches = list(re.finditer(ANSWER_PATTERN, content, flags=re.IGNORECASE))

    if matches:
        last = matches[-1]
        return last.group(1).upper()

    return None


def format_options(options: Optional[List[str]]) -> str:
    if not options:
        return ''
    option_lines = []
    for idx, opt in enumerate(options):
        option_label = chr(65 + idx)
        option_lines.append(f"{option_label}. {opt}")
    return "Options:\n" + "\n".join(option_lines)


def build_fewshot_context(dataset_key: str, disabled: bool) -> str:
    if disabled:
        return ''
    examples = DATASET_FEWSHOT_MAP.get(dataset_key, [])
    formatted_examples = []
    for idx, example in enumerate(examples, start=1):
        parts = [f"### Example {idx}"]
        question_block = example.get('question', '')
        
        # MMLU-Pro의 경우 question 필드에 이미 "Question:"이 포함되어 있을 수 있음
        if dataset_key == 'MMLU-Pro' and question_block.startswith('Question:'):
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

def save_json(row, model_name, jsonl_file, enable_thinking=False):
    """Save row as JSON format (array of JSON objects)"""
    # Qwen3 또는 Nemotron 모델인지 확인
    is_qwen3 = is_qwen3_model(model_name)
    is_nemotron = is_nemotron_model(model_name)
    
    # 저장 경로 구성: Qwen3 또는 Nemotron 모델이면 모드별 폴더 분리
    if is_qwen3 or is_nemotron:
        mode_subdir = "thinking" if enable_thinking else "non-thinking"
        storage_dir = os.path.join(EVAL_DIR, f'Evaluation_CoTs/{model_name}/{mode_subdir}')
    else:
        storage_dir = os.path.join(EVAL_DIR, f'Evaluation_CoTs/{model_name}')
    
    if not os.path.exists(storage_dir):
        os.makedirs(storage_dir)
    jsonl_basename = os.path.splitext(jsonl_file)[0]  # 확장자 제거
    file_path = os.path.join(storage_dir, f'{jsonl_basename}.json')
    
    # Convert row (Series) to dictionary
    row_dict = row.to_dict()

    # Group CoT/Final Answer pairs under "Reasoning Paths"
    cot_indices = []
    for key in row_dict.keys():
        match = re.match(r'^CoT_(\d+)$', key)
        if match:
            cot_indices.append(int(match.group(1)))
    cot_indices.sort()

    reasoning_paths = []
    for idx in cot_indices:
        cot_key = f'CoT_{idx}'
        final_key = f'Final Answer_{idx}'
        path_entry = {
            cot_key: row_dict.get(cot_key),
            final_key: row_dict.get(final_key)
        }
        token_key = f'Tokens_{idx}'
        if token_key in row_dict:
            path_entry[token_key] = row_dict.get(token_key)
        # Confidence 정보 추가 (있는 경우)
        confidence_key = f'Confidence_{idx}'
        if confidence_key in row_dict:
            path_entry[confidence_key] = row_dict.get(confidence_key)
        reasoning_paths.append(path_entry)
        row_dict.pop(cot_key, None)
        row_dict.pop(final_key, None)
        row_dict.pop(token_key, None)
        row_dict.pop(confidence_key, None)

    row_dict['Reasoning Paths'] = reasoning_paths
    
    # Read existing data if file exists
    if os.path.isfile(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = []
    
    # Append new row
    data.append(row_dict)
    
    # Write entire array back to file (with indentation for readability)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
def is_qwen3_model(model_name: str) -> bool:
    """Check if the model is a Qwen3 series model"""
    model_name_lower = model_name.lower()
    return 'qwen3' in model_name_lower or 'qwen-3' in model_name_lower

def is_nemotron_model(model_name: str) -> bool:
    """Check if the model is a Nemotron series model"""
    model_name_lower = model_name.lower()
    return 'nemotron' in model_name_lower or 'nvidia-nemotron' in model_name_lower

def main(args):
    jsonl_basename = os.path.splitext(args.jsonl_file)[0]
    
    # Qwen3 또는 Nemotron 모델인지 확인
    is_qwen3 = is_qwen3_model(args.model_name)
    is_nemotron = is_nemotron_model(args.model_name)
    
    # 저장 경로 구성: Qwen3 또는 Nemotron 모델이면 모드별 폴더 분리
    if is_qwen3 or is_nemotron:
        mode_subdir = "thinking" if args.enable_thinking else "non-thinking"
        storage_dir = os.path.join(EVAL_DIR, f'Evaluation_CoTs/{args.model_name}/{mode_subdir}')
    else:
        storage_dir = os.path.join(EVAL_DIR, f'Evaluation_CoTs/{args.model_name}')
    
    output_path = os.path.join(storage_dir, f'{jsonl_basename}.json')

    # 기존 결과 파일이 있으면 이미 처리된 항목 ID 확인 (resume 기능)
    processed_ids = set()
    if os.path.isfile(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                for item in existing_data:
                    item_id = item.get('id')
                    if item_id:
                        processed_ids.add(item_id)
            print(f"[RESUME] 기존 결과 파일 발견: {output_path}")
            print(f"[RESUME] 이미 처리된 항목 수: {len(processed_ids)}개")
        except Exception as e:
            print(f"[WARNING] 기존 결과 파일 읽기 실패: {e}")
            print(f"[WARNING] 처음부터 다시 시작합니다.")
    
    # Qwen3 또는 Nemotron 모델의 경우 thinking/non-thinking 모드에 따라 다른 샘플링 파라미터 사용
    if is_qwen3:
        if args.enable_thinking:
            # Thinking mode: Temperature=0.6, TopP=0.95, TopK=20, MinP=0
            sampling_params = SamplingParams(
                n=args.num_of_paths,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                min_p=0.0,
                max_tokens=args.max_tokens,
            )
            print("[INFO] Qwen3 Thinking 모드 사용: Temperature=0.6, TopP=0.95, TopK=20, MinP=0")
        else:
            # Non-thinking mode: Temperature=0.7, TopP=0.8, TopK=20, MinP=0
            sampling_params = SamplingParams(
                n=args.num_of_paths,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                min_p=0.0,
                max_tokens=args.max_tokens,
            )
            print("[INFO] Qwen3 Non-thinking 모드 사용: Temperature=0.7, TopP=0.8, TopK=20, MinP=0")
    elif is_nemotron:
        if args.enable_thinking:
            # Thinking mode: Temperature=0.6, TopP=0.95, TopK=20, MinP=0
            sampling_params = SamplingParams(
                n=args.num_of_paths,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                min_p=0.0,
                max_tokens=args.max_tokens,
            )
            print("[INFO] Nemotron Thinking 모드 사용: Temperature=0.6, TopP=0.95, TopK=20, MinP=0")
        else:
            # Non-thinking mode: Temperature=0.7, TopP=0.8, TopK=20, MinP=0
            sampling_params = SamplingParams(
                n=args.num_of_paths,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                min_p=0.0,
                max_tokens=args.max_tokens,
            )
            print("[INFO] Nemotron Non-thinking 모드 사용: Temperature=0.7, TopP=0.8, TopK=20, MinP=0")
    else:
        # 기존 모델의 경우 사용자 지정 파라미터 사용 (self_consistency.py와 동일하게 top_k, min_p 추가)
        sampling_params = SamplingParams(
            n=args.num_of_paths,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=20,
            min_p=0.0,
            max_tokens=args.max_tokens,
        )

    # JSONL 파일 로드
    jsonl_path = os.path.join(EVAL_DIR, args.jsonl_file)
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")
    
    data_list = load_jsonl(jsonl_path)
    data_subset = data_list[:args.num_of_samples]
    
    # 이미 처리된 항목 제외 (resume 기능)
    if processed_ids:
        original_count = len(data_subset)
        data_subset = [item for item in data_subset if item.get('id') not in processed_ids]
        remaining_count = len(data_subset)
        skipped_count = original_count - remaining_count
        print(f"[RESUME] 처리 대상: {original_count}개 중 {skipped_count}개 건너뛰고 {remaining_count}개 처리")
        
        if remaining_count == 0:
            print(f"[INFO] 모든 항목이 이미 처리되었습니다. 종료합니다.")
            return

    # LLM / Tokenizer 로드
    print(f"Loading model: {args.model_name}")
    # gemma3 모델은 float16을 지원하지 않으므로 bfloat16 or float32 사용
    if 'gemma' in args.model_name.lower():
        dtype = 'bfloat16'
    else:
        dtype = 'float16'
    llm_kwargs = {
        'dtype': dtype,
        'enable_prefix_caching': True,
        'enable_chunked_prefill': False,
        'trust_remote_code': True,
        'disable_log_stats': True,  # vLLM 통계 로그 비활성화
    }
    if args.gpu_memory_utilization is not None:
        llm_kwargs['gpu_memory_utilization'] = args.gpu_memory_utilization
    if args.max_model_len is not None:
        llm_kwargs['max_model_len'] = args.max_model_len
    llm = LLM(model=args.model_name, **llm_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print("Model loaded successfully!")
    
    # Confidence 추출을 위한 Runner 초기화 (활성화된 경우에만)
    # vLLM을 사용하므로 이미 로드된 llm과 tokenizer를 재사용
    confidence_runner = None
    if args.enable_confidence:
        print("Initializing vLLM-based confidence extraction runner...")
        try:
            confidence_runner = VLLMRunner(llm=llm, tokenizer=tokenizer)
            print("Confidence extraction runner initialized successfully!")
        except Exception as e:
            print(f"[WARNING] Confidence extraction runner 초기화 실패: {e}")
            print("[WARNING] Confidence 추출을 비활성화하고 계속 진행합니다.")
            args.enable_confidence = False

    # Prompt template 로드
    with open(args.prompt_link, 'r', encoding='utf-8') as f:
        prompt_template = json.load(f)
    
    batch_size = args.batch_size
    fewshot_context = build_fewshot_context(jsonl_basename, args.disable_fewshot)

    # 데이터셋에 따라 적절한 파싱 함수 선택 (배치 루프 밖에서 한 번만 실행)
    jsonl_basename = os.path.splitext(args.jsonl_file)[0]
    if jsonl_basename in ['AIME_2024', 'AIME_2025', 'MATH500_test', "GSM8K"]:
        extract_func = extract_math_answer
    elif jsonl_basename == 'GPQA-Diamond':
        extract_func = extract_answer_gpqa
    elif jsonl_basename == 'MMLU-Pro':
        extract_func = extract_answer_mmlu_pro
    else:
        # 기본값으로 math answer 추출 사용
        extract_func = extract_math_answer

    # 데이터셋 및 모델(Qwen3 여부)에 따라 적절한 system 프롬프트 선택
    system_template_dict = prompt_template.get('system', {})
    if isinstance(system_template_dict, dict):
        # Qwen3 모델이면 dataset_Qwen3 키를 우선 시도
        if is_qwen3:
            qwen3_key = f"{jsonl_basename}_Qwen3"
            system_template = system_template_dict.get(
                qwen3_key,
                system_template_dict.get(jsonl_basename, system_template_dict.get('default', ''))
            )
        else:
            system_template = system_template_dict.get(jsonl_basename, system_template_dict.get('default', ''))
    else:
        # 기존 형식 (문자열)인 경우 하위 호환성 유지
        system_template = system_template_dict if isinstance(system_template_dict, str) else ''

    # 여러 문제를 batch 단위로 vLLM에 넣어서 병렬 처리
    for start_idx in tqdm(range(0, len(data_subset), batch_size), desc="Generating", colour='blue'):
        batch = data_subset[start_idx:start_idx + batch_size]

        # chat template으로 변환
        prompts = []
        for data in batch:
            question = data.get('question', '')
            
            # 모든 데이터셋에 대해 "Question:" 형식으로 통일
            if not question.startswith('Question:'):
                question = f"Question: {question}"
            
            # MMLU-Pro 및 GPQA-Diamond 같은 multiple-choice 문제의 경우 options 포함
            if jsonl_basename in ['MMLU-Pro', 'GPQA-Diamond']:
                options = data.get('options')
                if options:
                    options_str = format_options(options)
                    question = f"{question}\n{options_str}"
            
            # 모든 데이터셋에 대해 "Answer: " 추가 (MMLU-Pro 형식 통일)
            question = f"{question}\nAnswer: "
            
            # Prompt template에서 system과 human 메시지 가져오기
            system_content = system_template.format(subject=args.subject)
            human_template = prompt_template.get('human', '{question}')
            if '{fewshot}' in human_template:
                human_content = human_template.format(
                    question=question,
                    fewshot=fewshot_context
                ).strip()
            elif fewshot_context:
                human_content = f"{fewshot_context}\n\n{human_template.format(question=question).strip()}"
            else:
                human_content = human_template.format(question=question).strip()
            
            # Nemotron 모델의 경우 thinking/non-thinking 모드에 따라 토큰 추가
            if is_nemotron:
                if args.enable_thinking:
                    # Thinking mode: /think 토큰을 system 또는 user 메시지에 추가
                    # user 메시지 앞에 추가하는 방식으로 구현
                    human_content = f"/think\n{human_content}"
                else:
                    # Non-thinking mode: /no_think 토큰 추가
                    human_content = f"/no_think\n{human_content}"
            
            messages = [
                {
                    "role": "system",
                    "content": system_content
                },
                {
                    "role": "user",
                    "content": human_content
                }
            ]
            
            # Qwen3 모델의 경우 enable_thinking 파라미터 추가
            apply_template_kwargs = {
                'conversation': messages,
                'tokenize': False,
                'add_generation_prompt': True
            }
            if is_qwen3:
                apply_template_kwargs['enable_thinking'] = args.enable_thinking
            
            prompt_string = tokenizer.apply_chat_template(**apply_template_kwargs)
            prompts.append(prompt_string)
            
            # 디버깅: 첫 번째 배치의 첫 번째 샘플만 출력
            if start_idx == 0 and len(prompts) == 1:
                print("\n" + "="*80)
                print("[DEBUG] 첫 번째 샘플의 입력 프롬프트 확인")
                print("="*80)
                print(f"\n[System Content]:\n{system_content}")
                print(f"\n[Human Content]:\n{human_content}")
                print("="*80 + "\n")

        # vLLM에 prompts 리스트를 한 번에 넣어서 병렬 생성
        # outputs: len(outputs) == len(prompts) == len(batch)
        outputs = llm.generate(prompts, sampling_params)

        # 각 출력에 대해 reasoning path 및 최종 정답 파싱
        for data, output, prompt in zip(batch, outputs, prompts):
            # 원본 데이터를 pandas Series로 변환 (저장을 위해)
            row = pd.Series(data.copy())
            
            # 입력 토큰 수 측정
            input_token_ids = tokenizer.encode(prompt)
            row['input_tokens'] = len(input_token_ids)
            
            for i, generation in enumerate(output.outputs):
                CoT = generation.text.strip()
                pred = extract_func(CoT)
                # None을 빈 문자열로 변환 (기존 코드와 호환성)
                if pred is None:
                    pred = ''
                token_ids = getattr(generation, "token_ids", None)
                if token_ids is not None:
                    tokens_used = len(token_ids)
                else:
                    # token_ids가 없는 경우 (gemma 모델 등) 생성된 텍스트를 토크나이저로 인코딩하여 계산
                    tokens_used = len(tokenizer.encode(CoT, add_special_tokens=False))
                
                # pandas Series에 추가
                row[f'CoT_{i}'] = CoT
                row[f'Final Answer_{i}'] = pred
                row[f'Tokens_{i}'] = tokens_used
                
                # Confidence 추출 (활성화된 경우)
                if args.enable_confidence and confidence_runner is not None:
                    try:
                        # answer_span 찾기 (CoT에서 답변 부분 위치 찾기)
                        answer_span = find_answer_span(CoT)
                        
                        # Confidence 추출
                        # 원래 BINARY_FT01 방식: CoT 전체(답변 부분까지)를 사용하여
                        # 각 path의 reasoning 과정이 confidence에 반영되도록 함
                        likelihoods = generate_confidence(
                            original_prompt=prompt,
                            original_response=CoT,
                            answer_span=answer_span,
                            runner=confidence_runner,
                            config=BINARY_FT01,
                        )
                        
                        # Likelihood를 confidence 점수로 변환
                        confidence_score = likelihood_to_confidence(likelihoods)
                        row[f'Confidence_{i}'] = confidence_score
                    except Exception as e:
                        print(f"[WARNING] Confidence 추출 실패 (path {i}): {e}")
                        row[f'Confidence_{i}'] = None

            # Save as JSON
            save_json(row, args.model_name, args.jsonl_file, args.enable_thinking)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-3B-Instruct", type=str,
                        help="Model name for vLLM (e.g., 'Qwen/Qwen2.5-3B-Instruct')")
    parser.add_argument("--num_of_paths", default=40, type=int,
                        help="Number of reasoning paths per question (Self-Consistency)")
    parser.add_argument("--num_of_samples", default=500, type=int,
                        help="Number of samples to process")
    parser.add_argument("--subject", default="math", type=str,
                        help="Subject for the prompt template")
    parser.add_argument("--batch_size", default=32, type=int,
                        help="Batch size for vLLM processing")
    parser.add_argument("--jsonl_file", default="MMLU-Pro.jsonl", type=str,
                        help="JSONL file name in eval directory")
    parser.add_argument("--prompt_link", default="prompt_template.json", type=str,
                        help="Path to prompt template JSON file")
    parser.add_argument("--temperature", default=0.7, type=float,
                        help="Sampling temperature")
    parser.add_argument("--max_tokens", default=10240, type=int,
                        help="Maximum tokens to generate")
    parser.add_argument("--top_p", default=0.95, type=float,
                        help="Top-p sampling parameter")
    parser.add_argument("--gpu_memory_utilization", default=0.9, type=float,
                        help="GPU memory utilization ratio (0.0 to 1.0)")
    parser.add_argument("--max_model_len", default=16384, type=int,
                        help="Maximum model length (reduce if KV cache memory is insufficient)")
    parser.add_argument("--disable_fewshot", default=False, action="store_true",
                        help="Disable prepending few-shot exemplars to the human prompt")
    parser.add_argument("--enable_thinking", default=False, action="store_true",
                        help="Enable thinking mode for Qwen3 models (Temperature=0.6, TopP=0.95, TopK=20, MinP=0). "
                             "If False, uses non-thinking mode (Temperature=0.7, TopP=0.8, TopK=20, MinP=0)")
    parser.add_argument("--enable_confidence", default=False, action="store_true",
                        help="Enable BINARY_FT01 confidence extraction for each reasoning path. "
                             "When enabled, extracts confidence score (0-1) for each generated path.")
    
    args = parser.parse_args()
    main(args)
