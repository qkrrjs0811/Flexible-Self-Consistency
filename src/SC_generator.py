from LLM_agent import *
import os
import pandas as pd
from tqdm import tqdm
import json

DATA_DIR = '../data'

# Experiment Config

DF_NAME = 'GSM8K'
DIFFICULTY = 'easy'

NUM_OF_SAMPLES = 2
NUM_OF_COT = 3
llm_config = {
    # change these three together
    'llm_type': 'huggingface',  # openai, ollama, anthropic, huggingface
    'api_key_link': 'api_key_yw.txt',
    'model': "Qwen/Qwen2.5-3B-Instruct",  # see llm_model.txt, for huggingface: e.g., "meta-llama/Llama-2-7b-chat-hf"
    # change these two together
    'prompt_link': 'prompt_template.json',
    'parser_template': CoT,
    # change as needed (논문 설정: temperature=0.7, max_tokens=1024, top_p=0.95)
    'temperature': 0.7,
    'max_tokens': 1024,
    'top_p': 0.95,
    # Hugging Face specific settings (only used when llm_type == 'huggingface')
    'cache_dir': '~/.cache/huggingface',  # 모델 캐시 디렉토리
    'device_map': 'auto',  # 'auto', 'cpu', 'cuda:0' 등
}


def save_csv(row):
    storage_dir = os.path.join(DATA_DIR, f'Evaluation_CoTs/{llm_config["model"]}')
    if not os.path.exists(storage_dir):
        os.makedirs(storage_dir)
    file_path = os.path.join(storage_dir, f'{DF_NAME}_{DIFFICULTY}.csv')

    if not os.path.isfile(file_path):
        # The file does not exist, write with header
        row.to_frame().T.to_csv(file_path, mode='a', index=False, header=True)
    else:
        # The file exists, append without header
        row.to_frame().T.to_csv(file_path, mode='a', index=False, header=False)

def save_json(row):
    """Save row as JSON format (array of JSON objects)"""
    storage_dir = os.path.join(DATA_DIR, f'Evaluation_CoTs/{llm_config["model"]}')
    if not os.path.exists(storage_dir):
        os.makedirs(storage_dir)
    file_path = os.path.join(storage_dir, f'{DF_NAME}_{DIFFICULTY}.json')
    
    # Convert row (Series) to dictionary
    row_dict = row.to_dict()
    
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
        
if __name__ == '__main__':
    with open(llm_config['api_key_link'], 'r') as f:
        api_key = f.read()
    df = pd.read_csv(os.path.join(DATA_DIR, f'{DF_NAME}/{DF_NAME}_{DIFFICULTY}.csv'))
    df_subset = df[:NUM_OF_SAMPLES]

    # Self-Consistency 방법론: 동일한 모델로 동일한 질문에 대해 여러 번의 독립적인 추론 생성
    # 따라서 모델은 한 번만 로드하고 모든 round에서 재사용해야 함
    print(f"Loading model: {llm_config['model']} (type: {llm_config['llm_type']})")
    if llm_config['llm_type'] == 'huggingface':
        cache_dir_path = os.path.expanduser(llm_config.get('cache_dir', '~/.cache/huggingface'))
        print(f"Cache directory: {cache_dir_path}")
        print("Note: If model exists in cache, it will be loaded without downloading.")
    
    shared_cot_agent = LLM_agent(
        llm_type=llm_config['llm_type'], 
        api_key=api_key, 
        model=llm_config['model'],
        temperature=llm_config['temperature'],
        max_tokens=llm_config.get('max_tokens', 1024),
        top_p=llm_config.get('top_p', 0.95),
        cache_dir=os.path.expanduser(llm_config.get('cache_dir', '~/.cache/huggingface')) if llm_config['llm_type'] == 'huggingface' else None,
        device_map=llm_config.get('device_map', 'auto') if llm_config['llm_type'] == 'huggingface' else 'auto'
    )
    shared_cot_agent.set_prompt(llm_config['prompt_link'])
    if llm_config['llm_type'] == 'openai' or llm_config['llm_type'] == 'anthropic' or llm_config['llm_type'] == 'huggingface':
        shared_cot_agent.set_parser(llm_config['parser_template'])
    print("Model loaded successfully! Will be reused for all rounds.")

    # Data collection
    START_INDEX = 0
    for row_idx in tqdm(range(START_INDEX, len(df_subset)), colour='blue', desc='Sample Progress', position=0):
        row = df_subset.iloc[row_idx].copy()  # copy()를 사용하여 SettingWithCopyWarning 방지
        subject = row['Category']
        question = row['Question']

        arguments_dict = {
            'subject': subject,
            'question': question
        }

        # Self-Consistency: 동일한 질문에 대해 NUM_OF_COT번의 독립적인 추론 경로 생성
        for round in tqdm(range(NUM_OF_COT), colour='green', desc='Round', position=1):
            instruction_violation_count = []
            parse_error = True
            parse_error_attempt = 0
            while parse_error and parse_error_attempt < 3:
                # 모든 모델 타입에서 동일한 모델 인스턴스 재사용 (Self-Consistency 방법론)
                cot_agent = shared_cot_agent
                response, attempts = cot_agent.involk(arguments_dict)
                
                # 각 LLM 타입별로 response 처리
                if llm_config['llm_type'] == 'ollama':
                    # Ollama는 AIMessage 객체 반환
                    if hasattr(response, 'content'):
                        response = response.content
                    else:
                        response = str(response)
                # HuggingFace, OpenAI, Anthropic은 이미 파싱된 딕셔너리 반환 (변환 불필요)
                try:
                    CoT = response['CoT']
                    Final_Answer = response['Final_answer']
                    parse_error = False
                except:
                    CoT = response
                    Final_Answer = 'error'
                    parse_error = True
                    parse_error_attempt += 1
                    # Ollama는 JSON 파싱 실패가 정상 (raw 텍스트 반환)
                    if llm_config['llm_type'] == 'ollama':
                        Final_Answer = None
                        parse_error = False                                            
                instruction_violation_count.append((attempts,parse_error_attempt))
            row[f'CoT_{round}'] = CoT
            row[f'Final Answer_{round}'] = Final_Answer
            row[f'Instruction Violation_{round}'] = instruction_violation_count

        save_csv(row)  # Save as CSV
        save_json(row)  # Save as JSON
