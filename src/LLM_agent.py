from langchain_openai import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import json
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage
from typing import List, Optional
import re
from langchain_community.chat_models import ChatOllama
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os
from Parsers import *

class LLM_agent:
    def __init__(self,api_key = None, llm_type = 'openai',model="gpt-3.5-turbo-0125",temperature=0.7, max_tokens=1024, top_p=0.95, cache_dir=None, device_map='auto'):
        self.api_key = api_key
        if llm_type == 'openai':
            self.llm = ChatOpenAI(openai_api_key=self.api_key, model_name=model, temperature=temperature, max_tokens=max_tokens)
        elif llm_type == 'anthropic':
            self.llm = ChatAnthropic(model=model,
                                  anthropic_api_key=self.api_key,
                                  temperature=temperature,
                                  max_tokens=max_tokens)
        elif llm_type == 'ollama':
            self.llm = ChatOllama(model=model)
        elif llm_type == 'huggingface':
            # Hugging Face 모델을 로컬 캐시에서 로드
            if cache_dir is None:
                cache_dir = os.path.expanduser('~/.cache/huggingface')
            os.makedirs(cache_dir, exist_ok=True)
            
            # from_pretrained는 cache_dir에 모델이 이미 존재하면 자동으로 다운로드하지 않고 로드함
            # local_files_only=False (기본값)이면 먼저 cache_dir을 확인하고, 없으면 다운로드 후 저장
            tokenizer = AutoTokenizer.from_pretrained(
                model, 
                cache_dir=cache_dir,
                local_files_only=False  # False면 캐시 확인 후 없으면 다운로드
            )
            hf_model = AutoModelForCausalLM.from_pretrained(
                model, 
                cache_dir=cache_dir,
                device_map=device_map,
                torch_dtype='auto',
                local_files_only=False  # False면 캐시 확인 후 없으면 다운로드
            )
            
            # Pipeline 생성 (논문 설정: temperature=0.7, max_tokens=1024, top_p=0.95)
            pipe = pipeline(
                "text-generation",
                model=hf_model,
                tokenizer=tokenizer,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_tokens,
                return_full_text=False,
                do_sample=True  # top_p를 사용하려면 샘플링 필요
            )
            
            self.llm = HuggingFacePipeline(pipeline=pipe)
        self.llm_type = llm_type
        self.chat_prompt = None

    def involk(self,var_dict):
        if self.llm_type == 'openai':
            output_parser = self.parser
        elif self.llm_type == 'anthropic':
            output_parser = extract_json

        
        # 각 LLM 타입별로 chain 구성
        if self.llm_type == 'ollama':
            # Ollama는 ChatModel이므로 파서 없이 사용
            chain = self.chat_prompt | self.llm 
        elif self.llm_type == 'huggingface':
            # HuggingFace는 LLM이므로 파서 없이 사용
            chain = self.chat_prompt | self.llm 
        else:
            # OpenAI, Anthropic은 JSON 파서 사용
            chain = self.chat_prompt | self.llm | output_parser
        success = False
        attempts = 0
        response = None
        while (not success) and attempts < 3:
            try:
                response = chain.invoke(var_dict)
                # OpenAI, Anthropic은 파싱된 딕셔너리 반환
                if self.llm_type == 'openai' or self.llm_type == 'anthropic':
                    if len(response) == self.num_of_llm_output:
                        success = True
                    else:
                        print("Response length does not match the expected length.")
                # Ollama, HuggingFace는 raw 응답 반환
                else:
                    success = True
            except Exception as e:
                print("An exception occurred:", str(e))
                attempts += 1
                success = False

            if attempts == 3:
                print('Maximum attempts reached.')
                chain = self.chat_prompt | self.llm
                response = chain.invoke(var_dict)

        return response,attempts

    def get_llm(self):
        return self.llm
    def set_prompt(self,prompt_json_link):
        with open(prompt_json_link) as f:
            message = []
            for key,val in json.load(f).items():
                # OpenAI, Anthropic, HuggingFace에 format_instructions 추가 (JSON 파서 사용)
                if self.llm_type == 'openai' or self.llm_type == 'anthropic' or self.llm_type == 'huggingface':
                    if key == 'system':
                        val += '\n{format_instructions}'
                message.append((key,val))

        chat_prompt = ChatPromptTemplate.from_messages(message)
        self.chat_prompt = chat_prompt

    def get_prompt(self):
        return self.chat_prompt

    def set_parser(self, parser_obj):
        self.parser = JsonOutputParser(pydantic_object=parser_obj)
        self.num_of_llm_output = len(parser_obj.__fields__)
        self.chat_prompt = self.chat_prompt.partial(format_instructions=self.parser.get_format_instructions())
    def get_parser(self):
        return self.parser

def extract_json(message: AIMessage) -> List[dict]:
    """Extracts JSON content from a string where JSON is embedded between ```json and ``` tags.

    Parameters:
        text (str): The text containing the JSON content.

    Returns:
        list: A list of extracted JSON strings.
    """
    text = message.content
    # Define the regular expression pattern to match JSON blocks
    pattern = r'"([^"]+)":\s*"([^"]+)"'

    # Find all non-overlapping matches of the pattern in the string
    matches = re.findall(pattern, text)

    # Return the list of matched JSON strings, stripping any leading or trailing whitespace
    try:
        json_string = json.dumps(dict(matches), indent=4)
        return dict(matches)
    except Exception:
        raise ValueError(f"Failed to parse: {message}")

def extract_json_from_string(text: str) -> dict:
    """
    Extracts a JSON block from a string (including markdown) and parses it.
    """
    
    # 1. 마크다운 ```json ... ``` 블록 찾기
    json_match = re.search(r'```json\s*([\s\S]*?)\s*```', text)
    
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        # 2. 마크다운이 없다면, 첫 번째 '{'와 마지막 '}' 사이의 내용 찾기
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            json_str = json_match.group(0)
        else:
            # 3. JSON을 찾지 못한 경우
            print(f"[Parsing Error] No JSON block found in response: {text}")
            return {} # 빈 딕셔너리 반환

    # 4. 찾은 문자열을 JSON으로 파싱
    try:
        # 줄바꿈 문자 등 이스케이프 문자 처리
        json_str = json_str.replace(r'\n', '\\n').replace(r'\"', '\\"')
        parsed_json = json.loads(json_str)
        if isinstance(parsed_json, dict):
            return parsed_json
        else:
            # print(f"[Parsing Error] Parsed content is not a dictionary: {json_str}")
            return {}
            
    except json.JSONDecodeError as e:
        # print(f"[Parsing Error] Failed to decode JSON: {e}")
        # print(f"[Parsing Error] Raw JSON string was: {json_str}")
        return {} # 파싱 실패 시 빈 딕셔너리 반환
    
if __name__ == '__main__':
    print(1)