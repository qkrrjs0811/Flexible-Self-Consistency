"""
Reasoning Path Generator 모듈

이 모듈은 Multi-stage RSC에서 사용되는 reasoning path(Chain-of-Thought)를 생성합니다.
기존 src/SC_generator_ours.py의 vLLM 기반 로직을 활용합니다.
"""

from typing import List, Dict, Any, Optional, Tuple
import sys
import os
import json
import random
from collections import defaultdict

# 기존 코드 경로 추가 (src 디렉토리)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from rsc.aggregator import SCAggregator


def _prompt_template_dataset_key(dataset_name: Optional[str]) -> Optional[str]:
    """prompt.json / refinement.json 조회용 키 (MMLU-Pro_full은 MMLU-Pro와 동일 템플릿)."""
    if dataset_name == "MMLU-Pro_full":
        return "MMLU-Pro"
    return dataset_name


def _is_multiple_choice_dataset(dataset_name: Optional[str]) -> bool:
    if not dataset_name:
        return False
    return dataset_name in ("MMLU-Pro", "MMLU-Pro_full", "GPQA-Diamond")


class ReasoningPathGenerator:
    """
    Reasoning path를 생성하는 클래스
    
    Stage 1: 원본 질문 q에 대해 n개의 reasoning path 생성
    Stage t (t ≥ 2): 질문 q + 이전 Stage reasoning paths에 대해 refined reasoning path 생성
    
    기존 SC_generator_ours.py의 vLLM 기반 로직을 활용합니다.
    """
    
    def __init__(self, model_name: str, tokenizer: Optional[AutoTokenizer] = None,
                 llm: Optional[LLM] = None, device: str = 'cuda',
                 gpu_memory_utilization: float = 0.85, max_model_len: int = 16384,
                 enable_thinking: bool = False,
                 prompt_template_path: str = None,
                 refinement_template_path: str = None):
        """
        Args:
            model_name: 모델 이름 (HuggingFace 경로)
            tokenizer: 토크나이저 (None이면 자동 로드)
            llm: vLLM LLM 인스턴스 (None이면 자동 로드)
            device: 실행 디바이스 ('cuda' 또는 'cpu')
            gpu_memory_utilization: GPU 메모리 사용률
            max_model_len: 최대 모델 길이
            enable_thinking: Qwen3 모델의 thinking 모드 활성화 여부
            prompt_template_path: 프롬프트 템플릿 파일 경로 (None이면 기본 경로 사용)
            refinement_template_path: Refinement 프롬프트 템플릿 파일 경로 (None이면 기본 경로 사용)
        """
        self.model_name = model_name
        self.device = device
        self.enable_thinking = enable_thinking
        self.is_qwen3 = self._is_qwen3_model(model_name)
        self.is_nemotron = self._is_nemotron_model(model_name)
        self.is_qwen3_or_nemotron = self.is_qwen3 or self.is_nemotron
        
        # 프롬프트 템플릿 로드
        if prompt_template_path is None:
            prompt_template_path = os.path.join(os.path.dirname(__file__), 'prompt', 'prompt.json')
        self.prompt_template = self._load_prompt_template(prompt_template_path)
        
        # Refinement 템플릿 로드
        if refinement_template_path is None:
            refinement_template_path = os.path.join(os.path.dirname(__file__), 'prompt', 'refinement.json')
        self.refinement_template = self._load_refinement_template(refinement_template_path)
        
        # vLLM 모델 로드 (기존 코드 로직 활용)
        if llm is None:
            if 'gemma' in model_name.lower():
                dtype = 'bfloat16'
            else:
                dtype = 'float16'
            llm_kwargs = {
                'dtype': dtype,
                'gpu_memory_utilization': gpu_memory_utilization,
                'max_model_len': max_model_len,
                'trust_remote_code': True,
            }
            self.llm = LLM(model=model_name, **llm_kwargs)
        else:
            self.llm = llm
        
        # Tokenizer 로드
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
        else:
            self.tokenizer = tokenizer
    
    def _is_qwen3_model(self, model_name: str) -> bool:
        """Qwen3 모델인지 확인 (기존 코드 로직)"""
        model_name_lower = model_name.lower()
        return 'qwen3' in model_name_lower or 'qwen-3' in model_name_lower
    
    def _is_nemotron_model(self, model_name: str) -> bool:
        """Nemotron 모델인지 확인"""
        model_name_lower = model_name.lower()
        return 'nemotron' in model_name_lower or 'nvidia-nemotron' in model_name_lower
    
    def _load_prompt_template(self, template_path: str) -> Dict[str, Any]:
        """
        프롬프트 템플릿 로드
        
        Args:
            template_path: 템플릿 파일 경로
            
        Returns:
            Dict[str, Any]: 프롬프트 템플릿 딕셔너리
        """
        if os.path.exists(template_path):
            with open(template_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # 기본값 반환
            return {
                'system': {'default': 'You are a helpful assistant.'},
                'human': '{question}'
            }
    
    def _load_refinement_template(self, template_path: str) -> Dict[str, Any]:
        """
        Refinement 프롬프트 템플릿 로드
        
        Args:
            template_path: 템플릿 파일 경로
            
        Returns:
            Dict[str, Any]: Refinement 템플릿 딕셔너리
        """
        if os.path.exists(template_path):
            with open(template_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # 기본값 반환
            return {
                'collective': {
                    'template': '질문: {question}\n\n이전 추론 과정들:\n{previous_reasonings}\n\n새로운 추론:',
                    'format_reasoning': '[추론 {index}]\n{path}'
                },
                'individual': {
                    'template': '질문: {question}\n\n이전 추론 과정:\n[추론 {path_index}]\n{previous_reasoning}\n\n새로운 추론:'
                }
            }
    
    def generate_stage1_paths(self, question: str, n_paths: int = 5, 
                              temperature: float = 0.7, max_tokens: int = 10240,
                              top_p: float = 0.95, prompt_template: Dict = None,
                              fewshot_context: str = '', dataset_name: str = None,
                              subject: str = 'math', options: List[str] = None) -> Tuple[List[str], Dict[str, Any]]:
        """
        Stage 1: 원본 질문에 대해 n1개의 reasoning path 생성
        
        Args:
            question: 입력 질문
            n_paths: 생성할 reasoning path 개수
            temperature: 생성 시 사용할 temperature
            max_tokens: 최대 생성 토큰 수
            top_p: Top-p 샘플링 파라미터
            prompt_template: 프롬프트 템플릿 딕셔너리
            fewshot_context: Few-shot 예제 컨텍스트
            
        Returns:
            Tuple[List[str], Dict]: (생성된 reasoning path 리스트, 토큰 정보)
                토큰 정보: {'input_tokens': int, 'output_tokens': List[int]}
        """
        # Qwen3 또는 Nemotron 모델의 경우 thinking/non-thinking 모드에 따라 다른 파라미터 사용 (기존 코드 로직)
        if self.is_qwen3_or_nemotron:
            if self.enable_thinking:
                sampling_params = SamplingParams(
                    n=n_paths,
                    temperature=0.6,
                    top_p=0.95,
                    top_k=20,
                    min_p=0.0,
                    max_tokens=max_tokens,
                )
            else:
                sampling_params = SamplingParams(
                    n=n_paths,
                    temperature=0.7,
                    top_p=0.8,
                    top_k=20,
                    min_p=0.0,
                    max_tokens=max_tokens,
                )
        else:
            sampling_params = SamplingParams(
                n=n_paths,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
        
        # 프롬프트 구성 (기존 코드 로직 활용)
        prompt = self._build_prompt(question, prompt_template, fewshot_context, 
                                   dataset_name, subject, options)
        
        # 입력 토큰 수 계산 (기존 코드와 동일)
        input_token_ids = self.tokenizer.encode(prompt)
        input_tokens = len(input_token_ids)
        
        # vLLM으로 생성
        outputs = self.llm.generate([prompt], sampling_params)
        
        # 결과 추출 (토큰 정보 포함)
        paths = []
        token_info = {
            'input_tokens': input_tokens,
            'output_tokens': []
        }
        if outputs and len(outputs) > 0:
            for output in outputs[0].outputs:
                generated_text = output.text.strip()
                paths.append(generated_text)
                # 각 출력의 토큰 수 계산
                token_ids = getattr(output, "token_ids", None)
                if token_ids is not None:
                    tokens_used = len(token_ids)
                else:
                    # token_ids가 없는 경우 (gemma 모델 등) 생성된 텍스트를 토크나이저로 인코딩하여 계산
                    tokens_used = len(self.tokenizer.encode(generated_text, add_special_tokens=False))
                token_info['output_tokens'].append(tokens_used)
        
        return paths, token_info
    
    def generate_stage1_paths_batch(self, questions: List[str], n_paths: int = 5,
                                   temperature: float = 0.7, max_tokens: int = 10240,
                                   top_p: float = 0.95, prompt_template: Dict = None,
                                   fewshot_context: str = '', dataset_name: str = None,
                                   subject: str = 'math', options_list: List[List[str]] = None,
                                   batch_size: int = 32,
                                   batch_callback: callable = None,
                                   print_first_prompt: bool = False) -> Tuple[List[List[str]], List[Dict[str, Any]]]:
        """
        Stage 1: 배치 처리 버전 - 여러 질문에 대해 n1개의 reasoning path 생성
        SC_generator_ours.py의 배치 처리 방식 참고
        
        Args:
            questions: 입력 질문 리스트
            n_paths: 생성할 reasoning path 개수
            temperature: 생성 시 사용할 temperature
            max_tokens: 최대 생성 토큰 수
            top_p: Top-p 샘플링 파라미터
            prompt_template: 프롬프트 템플릿 딕셔너리
            fewshot_context: Few-shot 예제 컨텍스트
            dataset_name: 데이터셋 이름
            subject: 주제
            options_list: 선택지 리스트 (multiple-choice 문제용, 각 샘플별)
            batch_size: vLLM 배치 크기
            batch_callback: 배치 완료 시 호출할 콜백 함수 (batch_start_idx, batch_paths, batch_token_infos) -> None
            
        Returns:
            Tuple[List[List[str]], List[Dict]]: (각 샘플별 reasoning path 리스트, 각 샘플별 토큰 정보)
        """
        # Qwen3 모델의 경우 thinking/non-thinking 모드에 따라 다른 파라미터 사용
        if self.is_qwen3:
            if self.enable_thinking:
                sampling_params = SamplingParams(
                    n=n_paths,
                    temperature=0.6,
                    top_p=0.95,
                    top_k=20,
                    min_p=0.0,
                    max_tokens=max_tokens,
                )
            else:
                sampling_params = SamplingParams(
                    n=n_paths,
                    temperature=0.7,
                    top_p=0.8,
                    top_k=20,
                    min_p=0.0,
                    max_tokens=max_tokens,
                )
        else:
            sampling_params = SamplingParams(
                n=n_paths,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
        
        all_paths = []  # List[List[str]] - 각 샘플별 paths
        all_token_infos = []  # List[Dict] - 각 샘플별 token_info
        
        # 배치 단위로 처리 (SC_generator_ours.py 방식)
        for start_idx in range(0, len(questions), batch_size):
            batch_end = min(start_idx + batch_size, len(questions))
            batch_questions = questions[start_idx:batch_end]
            batch_options = (options_list[start_idx:batch_end] if options_list else [None] * len(batch_questions))
            
            # 배치 프롬프트 구성
            batch_prompts = []
            for i, question in enumerate(batch_questions):
                options = batch_options[i] if i < len(batch_options) else None
                prompt = self._build_prompt(
                    question, prompt_template, fewshot_context,
                    dataset_name, subject, options
                )
                batch_prompts.append(prompt)
                
                # 첫 번째 배치의 첫 번째 샘플에 대해서만 프롬프트 출력
                if print_first_prompt and start_idx == 0 and i == 0:
                    print("\n" + "="*80)
                    print(f"[프롬프트 출력] 데이터셋: {dataset_name or 'Unknown'}, 첫 번째 샘플")
                    print("="*80)
                    print(prompt)
                    print("="*80 + "\n")
            
            # vLLM에 배치로 전달 (SC_generator_ours.py 522줄 방식)
            # 디버그: 실제로 몇 개의 프롬프트를 한 번에 전달하는지 확인
            if len(batch_prompts) > 0:
                print(f"[배치 처리] vLLM에 {len(batch_prompts)}개 프롬프트를 한 번에 전달 (각 프롬프트마다 {n_paths}개 path 생성)")
            outputs = self.llm.generate(batch_prompts, sampling_params)
            print(f"[배치 처리 완료] {len(outputs)}개 출력 수신 (예상: {len(batch_prompts)}개)")
            
            # 배치 결과 저장
            batch_paths = []
            batch_token_infos = []
            
            # 각 출력 처리
            for output, prompt in zip(outputs, batch_prompts):
                paths = []
                input_token_ids = self.tokenizer.encode(prompt)
                input_tokens = len(input_token_ids)
                
                token_info = {
                    'input_tokens': input_tokens,
                    'output_tokens': []
                }
                
                if output and len(output.outputs) > 0:
                    for gen_output in output.outputs:
                        generated_text = gen_output.text.strip()
                        paths.append(generated_text)
                        token_ids = getattr(gen_output, "token_ids", None)
                        if token_ids is not None:
                            tokens_used = len(token_ids)
                        else:
                            # token_ids가 없는 경우 (gemma 모델 등) 생성된 텍스트를 토크나이저로 인코딩하여 계산
                            tokens_used = len(self.tokenizer.encode(generated_text, add_special_tokens=False))
                        token_info['output_tokens'].append(tokens_used)
                
                batch_paths.append(paths)
                batch_token_infos.append(token_info)
                all_paths.append(paths)
                all_token_infos.append(token_info)
            
            # 배치 완료 콜백 호출 (중간 저장용)
            if batch_callback:
                batch_callback(start_idx, batch_paths, batch_token_infos)
        
        return all_paths, all_token_infos
    
    def generate_stage_t_paths(self, question: str, previous_stage_paths: List[str],
                                stage_number: int = 2, n_paths: int = 5, 
                                temperature: float = 0.7, max_tokens: int = 10240,
                                top_p: float = 0.95, refinement_prompt: str = None,
                                mode: str = 'collective', dataset_name: str = None,
                                subject: str = 'math', collective_num_paths: int = None):
        """
        Stage t (t ≥ 2): 이전 Stage의 paths를 참고하여 reflective reasoning path 생성
        일반화된 버전으로, 임의의 Stage t에 대해 사용 가능
        
        Args:
            question: 원본 질문
            previous_stage_paths: 이전 Stage에서 생성된 reasoning path 리스트 (R^{t-1})
            stage_number: 현재 Stage 번호 (로깅 및 프롬프트용, 기본값 2)
            n_paths: 생성할 refined reasoning path 개수
                - collective 모드: 이 파라미터 사용 (이전 chain들을 참고하여 새로운 n_paths개 생성)
                - individual 모드: 무시됨 (이전 Stage의 chain 개수만큼 자동 생성, 각 chain마다 1:1 refinement)
            temperature: 생성 시 사용할 temperature
            max_tokens: 최대 생성 토큰 수
            top_p: Top-p 샘플링 파라미터
            refinement_prompt: Refinement 프롬프트 (None이면 기본 사용)
            mode: Stage 방식 ('collective' 또는 'individual')
                - 'collective': 이전 Stage chain들을 보여주고 n_paths개의 새로운 chain 생성
                - 'individual': 각 이전 Stage chain마다 개별적으로 refinement (결과 개수 = 이전 Stage chain 개수)
            dataset_name: 데이터셋 이름
            subject: 주제
            collective_num_paths: Collective 모드에서 사용할 이전 Stage path 개수 (None이면 전체 사용)
            
        Returns:
            Tuple[List[str], Dict]: (생성된 refined reasoning path 리스트, 토큰 정보)
        """
        if mode == 'individual':
            return self._generate_reflective_paths_individual(
                question, previous_stage_paths, stage_number, temperature, max_tokens, top_p,
                dataset_name, subject
            )
        else:  # collective (기본값)
            return self._generate_reflective_paths_collective(
                question, previous_stage_paths, stage_number, n_paths, temperature, 
                max_tokens, top_p, refinement_prompt, dataset_name, subject, collective_num_paths
            )
    
    def _generate_reflective_paths_collective(self, question: str, previous_stage_paths: List[str],
                                             stage_number: int = 2, n_paths: int = 5,
                                             temperature: float = 0.7, max_tokens: int = 10240,
                                             top_p: float = 0.95, refinement_prompt: str = None,
                                             dataset_name: str = None, subject: str = 'math',
                                             collective_num_paths: int = None) -> Tuple[List[str], Dict[str, Any]]:
        """
        Reflective paths 생성 (Collective 방식): 이전 Stage chain들을 보여주고 n_paths개의 새로운 chain 생성
        일반화된 버전으로, 임의의 Stage t에 대해 사용 가능
        
        Args:
            question: 원본 질문
            previous_stage_paths: 이전 Stage에서 생성된 reasoning path 리스트
            stage_number: 현재 Stage 번호
            n_paths: 생성할 refined reasoning path 개수
            temperature: 생성 시 사용할 temperature
            max_tokens: 최대 생성 토큰 수
            top_p: Top-p 샘플링 파라미터
            refinement_prompt: Refinement 프롬프트 (None이면 기본 사용)
            dataset_name: 데이터셋 이름
            subject: 주제
            collective_num_paths: Collective 모드에서 사용할 이전 Stage path 개수 (None이면 전체 사용)
            
        Returns:
            Tuple[List[str], Dict]: (생성된 refined reasoning path 리스트, 토큰 정보)
        """
        # Collective 모드에서 사용할 이전 Stage paths 선택
        if collective_num_paths is not None and collective_num_paths < len(previous_stage_paths):
            # 답변의 다양성을 고려한 샘플링
            selected_paths = self._sample_paths_by_answer_diversity(
                previous_stage_paths, collective_num_paths, dataset_name
            )
        else:
            # 전체 사용
            selected_paths = previous_stage_paths
        
        # Qwen3 모델의 경우 thinking/non-thinking 모드에 따라 다른 파라미터 사용
        if self.is_qwen3:
            if self.enable_thinking:
                sampling_params = SamplingParams(
                    n=n_paths,
                    temperature=0.6,
                    top_p=0.95,
                    top_k=20,
                    min_p=0.0,
                    max_tokens=max_tokens,
                )
            else:
                sampling_params = SamplingParams(
                    n=n_paths,
                    temperature=0.7,
                    top_p=0.8,
                    top_k=20,
                    min_p=0.0,
                    max_tokens=max_tokens,
                )
        else:
            sampling_params = SamplingParams(
                n=n_paths,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
        
        # Refinement 프롬프트 구성
        if refinement_prompt is None:
            # 기본 refinement 프롬프트 생성 (선택된 이전 Stage paths 포함)
            refinement_prompt = self._build_refinement_prompt(question, selected_paths, dataset_name, subject, stage_number)
        
        # 입력 토큰 수 계산 (collective: 하나의 프롬프트)
        input_token_ids = self.tokenizer.encode(refinement_prompt)
        input_tokens = len(input_token_ids)
        
        # vLLM으로 생성
        outputs = self.llm.generate([refinement_prompt], sampling_params)
        
        # 결과 추출 (토큰 정보 포함)
        paths = []
        token_info = {
            'input_tokens': input_tokens,
            'output_tokens': []
        }
        if outputs and len(outputs) > 0:
            for output in outputs[0].outputs:
                paths.append(output.text.strip())
                # 각 출력의 토큰 수 계산
                token_ids = getattr(output, "token_ids", None)
                tokens_used = len(token_ids) if token_ids is not None else None
                token_info['output_tokens'].append(tokens_used)
        
        return paths, token_info
    
    def _generate_reflective_paths_individual(self, question: str, previous_stage_paths: List[str],
                                             stage_number: int = 2, temperature: float = 0.7,
                                             max_tokens: int = 10240, top_p: float = 0.95,
                                             dataset_name: str = None, subject: str = 'math') -> Tuple[List[str], Dict[str, Any]]:
        """
        Reflective paths 생성 (Individual 방식): 각 이전 Stage chain마다 개별적으로 refinement
        배치 처리로 최적화: 모든 refinement 프롬프트를 한 번에 처리
        일반화된 버전으로, 임의의 Stage t에 대해 사용 가능
        
        Args:
            question: 원본 질문
            previous_stage_paths: 이전 Stage에서 생성된 reasoning path 리스트
            stage_number: 현재 Stage 번호
            temperature: 생성 시 사용할 temperature
            max_tokens: 최대 생성 토큰 수
            top_p: Top-p 샘플링 파라미터
            dataset_name: 데이터셋 이름
            subject: 주제
            
        Returns:
            Tuple[List[str], Dict]: (생성된 refined reasoning path 리스트, 토큰 정보)
        """
        # Qwen3 모델의 경우 thinking/non-thinking 모드에 따라 다른 파라미터 사용
        if self.is_qwen3:
            if self.enable_thinking:
                sampling_params = SamplingParams(
                    n=1,  # 각 프롬프트마다 1개씩 생성
                    temperature=0.6,
                    top_p=0.95,
                    top_k=20,
                    min_p=0.0,
                    max_tokens=max_tokens,
                )
            else:
                sampling_params = SamplingParams(
                    n=1,  # 각 프롬프트마다 1개씩 생성
                    temperature=0.7,
                    top_p=0.8,
                    top_k=20,
                    min_p=0.0,
                    max_tokens=max_tokens,
                )
        else:
            sampling_params = SamplingParams(
                n=1,  # 각 프롬프트마다 1개씩 생성
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
        
        # 모든 이전 Stage path에 대한 refinement 프롬프트를 먼저 구성 (배치 처리)
        refinement_prompts = []
        input_tokens_list = []
        for i, previous_path in enumerate(previous_stage_paths):
            refinement_prompt = self._build_refinement_prompt_individual(
                question, previous_path, i+1, dataset_name, subject, stage_number
            )
            refinement_prompts.append(refinement_prompt)
            # 각 프롬프트의 입력 토큰 수 계산 (individual: 각각 다른 프롬프트)
            input_token_ids = self.tokenizer.encode(refinement_prompt)
            input_tokens_list.append(len(input_token_ids))
        
        # vLLM에 모든 refinement 프롬프트를 한 번에 배치로 전달 (1번 호출)
        outputs = self.llm.generate(refinement_prompts, sampling_params)
        
        # 결과 추출 (토큰 정보 포함)
        refined_paths = []
        token_info = {
            'input_tokens': input_tokens_list,  # individual: 각 프롬프트별 토큰 수 리스트
            'output_tokens': []
        }
        for i, output in enumerate(outputs):
            if output and len(output.outputs) > 0:
                refined_path = output.outputs[0].text.strip()
                refined_paths.append(refined_path)
                # 각 출력의 토큰 수 계산
                token_ids = getattr(output.outputs[0], "token_ids", None)
                if token_ids is not None:
                    tokens_used = len(token_ids)
                else:
                    # token_ids가 없는 경우 (gemma 모델 등) 생성된 텍스트를 토크나이저로 인코딩하여 계산
                    tokens_used = len(self.tokenizer.encode(refined_path, add_special_tokens=False))
                token_info['output_tokens'].append(tokens_used)
            else:
                # 생성 실패 시 원본 path 사용
                refined_paths.append(previous_stage_paths[i])
                token_info['output_tokens'].append(None)
        
        return refined_paths, token_info
    
    def generate_stage_t_paths_batch(self, questions: List[str], previous_stage_paths_list: List[List[str]],
                                     stage_number: int = 2, n_paths: int = 5,
                                     temperature: float = 0.7, max_tokens: int = 10240,
                                     top_p: float = 0.95, mode: str = 'collective',
                                     dataset_name: str = None, subject: str = 'math',
                                     collective_num_paths: int = None,
                                     batch_size: int = 32) -> Tuple[List[List[str]], List[Dict[str, Any]]]:
        """
        Stage t (t ≥ 2): 배치 처리 버전 - 여러 샘플에 대해 reflective reasoning path 생성
        
        Args:
            questions: 원본 질문 리스트
            previous_stage_paths_list: 각 샘플별 이전 Stage reasoning path 리스트 (List[List[str]])
            stage_number: 현재 Stage 번호
            n_paths: 생성할 refined reasoning path 개수
                - collective 모드: 이 파라미터 사용 (이전 chain들을 참고하여 새로운 n_paths개 생성)
                - individual 모드: 무시됨 (이전 Stage의 chain 개수만큼 자동 생성, 각 chain마다 1:1 refinement)
            temperature: 생성 시 사용할 temperature
            max_tokens: 최대 생성 토큰 수
            top_p: Top-p 샘플링 파라미터
            mode: Stage 방식 ('collective' 또는 'individual')
            dataset_name: 데이터셋 이름
            subject: 주제
            collective_num_paths: Collective 모드에서 사용할 이전 Stage path 개수
            batch_size: vLLM 배치 크기
            
        Returns:
            Tuple[List[List[str]], List[Dict]]: (각 샘플별 refined reasoning path 리스트, 각 샘플별 토큰 정보)
        """
        if mode == 'individual':
            return self._generate_reflective_paths_individual_batch(
                questions, previous_stage_paths_list, stage_number, temperature, max_tokens, top_p,
                dataset_name, subject, batch_size
            )
        else:  # collective
            return self._generate_reflective_paths_collective_batch(
                questions, previous_stage_paths_list, stage_number, n_paths, temperature,
                max_tokens, top_p, dataset_name, subject, collective_num_paths, batch_size
            )
    
    def _generate_reflective_paths_collective_batch(self, questions: List[str],
                                                    previous_stage_paths_list: List[List[str]],
                                                    stage_number: int = 2, n_paths: int = 5,
                                                    temperature: float = 0.7, max_tokens: int = 10240,
                                                    top_p: float = 0.95, dataset_name: str = None,
                                                    subject: str = 'math', collective_num_paths: int = None,
                                                    batch_size: int = 32) -> Tuple[List[List[str]], List[Dict[str, Any]]]:
        """Collective 모드 배치 처리"""
        # Qwen3 모델의 경우 thinking/non-thinking 모드에 따라 다른 파라미터 사용
        if self.is_qwen3:
            if self.enable_thinking:
                sampling_params = SamplingParams(
                    n=n_paths,
                    temperature=0.6,
                    top_p=0.95,
                    top_k=20,
                    min_p=0.0,
                    max_tokens=max_tokens,
                )
            else:
                sampling_params = SamplingParams(
                    n=n_paths,
                    temperature=0.7,
                    top_p=0.8,
                    top_k=20,
                    min_p=0.0,
                    max_tokens=max_tokens,
                )
        else:
            sampling_params = SamplingParams(
                n=n_paths,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
        
        all_paths = []
        all_token_infos = []
        
        # 배치 단위로 처리
        for start_idx in range(0, len(questions), batch_size):
            batch_end = min(start_idx + batch_size, len(questions))
            batch_questions = questions[start_idx:batch_end]
            batch_previous_paths = previous_stage_paths_list[start_idx:batch_end]
            
            # 각 샘플별 refinement 프롬프트 구성
            batch_prompts = []
            for question, previous_paths in zip(batch_questions, batch_previous_paths):
                # Collective 모드에서 사용할 이전 Stage paths 선택
                if collective_num_paths is not None and collective_num_paths < len(previous_paths):
                    selected_paths = self._sample_paths_by_answer_diversity(
                        previous_paths, collective_num_paths, dataset_name
                    )
                else:
                    selected_paths = previous_paths
                
                refinement_prompt = self._build_refinement_prompt(
                    question, selected_paths, dataset_name, subject, stage_number
                )
                batch_prompts.append(refinement_prompt)
            
            # vLLM에 배치로 전달
            # 디버그: 실제로 몇 개의 프롬프트를 한 번에 전달하는지 확인
            if len(batch_prompts) > 0:
                print(f"[배치 처리] Stage {stage_number} (Collective): vLLM에 {len(batch_prompts)}개 프롬프트를 한 번에 전달 (각 프롬프트마다 {n_paths}개 path 생성)")
            outputs = self.llm.generate(batch_prompts, sampling_params)
            print(f"[배치 처리 완료] Stage {stage_number} (Collective): {len(outputs)}개 출력 수신 (예상: {len(batch_prompts)}개)")
            
            # 출력 개수 검증
            if len(outputs) != len(batch_prompts):
                raise ValueError(
                    f"Stage {stage_number} (Collective): 출력 개수 불일치! "
                    f"예상: {len(batch_prompts)}개, 실제: {len(outputs)}개. "
                    f"vLLM generate가 예상보다 적은 출력을 반환했습니다."
                )
            
            # 각 출력 처리
            for output, prompt in zip(outputs, batch_prompts):
                paths = []
                input_token_ids = self.tokenizer.encode(prompt)
                input_tokens = len(input_token_ids)
                
                token_info = {
                    'input_tokens': input_tokens,
                    'output_tokens': []
                }
                
                if output and len(output.outputs) > 0:
                    for gen_output in output.outputs:
                        paths.append(gen_output.text.strip())
                        token_ids = getattr(gen_output, "token_ids", None)
                        tokens_used = len(token_ids) if token_ids is not None else None
                        token_info['output_tokens'].append(tokens_used)
                
                all_paths.append(paths)
                all_token_infos.append(token_info)
        
        return all_paths, all_token_infos
    
    def _generate_reflective_paths_individual_batch(self, questions: List[str],
                                                    previous_stage_paths_list: List[List[str]],
                                                    stage_number: int = 2, temperature: float = 0.7,
                                                    max_tokens: int = 10240, top_p: float = 0.95,
                                                    dataset_name: str = None, subject: str = 'math',
                                                    batch_size: int = 32) -> Tuple[List[List[str]], List[Dict[str, Any]]]:
        """Individual 모드 배치 처리"""
        # Qwen3 모델의 경우 thinking/non-thinking 모드에 따라 다른 파라미터 사용
        if self.is_qwen3:
            if self.enable_thinking:
                sampling_params = SamplingParams(
                    n=1,
                    temperature=0.6,
                    top_p=0.95,
                    top_k=20,
                    min_p=0.0,
                    max_tokens=max_tokens,
                )
            else:
                sampling_params = SamplingParams(
                    n=1,
                    temperature=0.7,
                    top_p=0.8,
                    top_k=20,
                    min_p=0.0,
                    max_tokens=max_tokens,
                )
        else:
            sampling_params = SamplingParams(
                n=1,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
        
        all_paths = []
        all_token_infos = []
        
        # 배치 단위로 처리: 여러 샘플을 한 번에 처리
        for start_idx in range(0, len(questions), batch_size):
            batch_end = min(start_idx + batch_size, len(questions))
            batch_questions = questions[start_idx:batch_end]
            batch_previous_paths_list = previous_stage_paths_list[start_idx:batch_end]
            
            # 모든 샘플의 모든 refinement 프롬프트를 하나의 배치로 구성
            batch_prompts = []
            batch_input_tokens_list = []  # 각 프롬프트별 input_tokens
            sample_boundaries = []  # 각 샘플이 시작하는 인덱스
            
            for question, previous_paths in zip(batch_questions, batch_previous_paths_list):
                sample_start_idx = len(batch_prompts)
                sample_boundaries.append(sample_start_idx)
                
                for i, previous_path in enumerate(previous_paths):
                    refinement_prompt = self._build_refinement_prompt_individual(
                        question, previous_path, i+1, dataset_name, subject, stage_number
                    )
                    batch_prompts.append(refinement_prompt)
                    input_token_ids = self.tokenizer.encode(refinement_prompt)
                    batch_input_tokens_list.append(len(input_token_ids))
            
            # vLLM에 배치로 전달 (모든 샘플의 모든 프롬프트를 한 번에)
            if len(batch_prompts) > 0:
                print(f"[배치 처리] Stage {stage_number} (Individual): vLLM에 {len(batch_prompts)}개 프롬프트를 한 번에 전달 ({len(batch_questions)}개 샘플, 샘플당 평균 {len(batch_prompts)//len(batch_questions)}개 path)")
            outputs = self.llm.generate(batch_prompts, sampling_params)
            print(f"[배치 처리 완료] Stage {stage_number} (Individual): {len(outputs)}개 출력 수신 (예상: {len(batch_prompts)}개)")
            
            # 출력 개수 검증
            if len(outputs) != len(batch_prompts):
                raise ValueError(
                    f"Stage {stage_number} (Individual): 출력 개수 불일치! "
                    f"예상: {len(batch_prompts)}개, 실제: {len(outputs)}개. "
                    f"vLLM generate가 예상보다 적은 출력을 반환했습니다."
                )
            
            # 결과를 샘플별로 분리
            for sample_idx, (sample_start_idx, question, previous_paths) in enumerate(zip(sample_boundaries, batch_questions, batch_previous_paths_list)):
                sample_end_idx = sample_start_idx + len(previous_paths)
                if sample_end_idx > len(outputs):
                    raise ValueError(
                        f"Stage {stage_number} (Individual): 샘플 {sample_idx}의 출력 인덱스 범위 초과! "
                        f"sample_end_idx={sample_end_idx}, outputs 길이={len(outputs)}"
                    )
                sample_outputs = outputs[sample_start_idx:sample_end_idx]
                sample_input_tokens = batch_input_tokens_list[sample_start_idx:sample_end_idx]
                
                # 결과 추출
                refined_paths = []
                output_tokens_list = []
                for output in sample_outputs:
                    if output and len(output.outputs) > 0:
                        refined_path = output.outputs[0].text.strip()
                        refined_paths.append(refined_path)
                        token_ids = getattr(output.outputs[0], "token_ids", None)
                        tokens_used = len(token_ids) if token_ids is not None else None
                        output_tokens_list.append(tokens_used)
                    else:
                        refined_paths.append("")
                        output_tokens_list.append(None)
                
                all_paths.append(refined_paths)
                all_token_infos.append({
                    'input_tokens': sample_input_tokens,
                    'output_tokens': output_tokens_list
                })
        
        return all_paths, all_token_infos
    
    def _build_prompt(self, question: str, prompt_template: Dict = None,
                     fewshot_context: str = '', dataset_name: str = None,
                     subject: str = 'math', options: List[str] = None) -> str:
        """
        프롬프트 구성 (기존 SC_generator_ours.py 로직 활용)
        
        Args:
            question: 질문
            prompt_template: 프롬프트 템플릿 딕셔너리
            fewshot_context: Few-shot 컨텍스트
            dataset_name: 데이터셋 이름 (예: 'AIME_2024', 'MMLU-Pro')
            subject: 주제 (예: 'math', 'science', 'general')
            options: 선택지 리스트 (multiple-choice 문제용)
            
        Returns:
            str: 구성된 프롬프트
        """
        # 질문 포맷팅 (기존 코드 로직)
        # 모든 데이터셋에 대해 "Question:" 형식으로 통일
        if not question.startswith('Question:'):
            question = f"Question: {question}"
        
        # MMLU-Pro / MMLU-Pro_full 및 GPQA-Diamond 같은 multiple-choice 문제의 경우 options 포함
        if _is_multiple_choice_dataset(dataset_name) and options:
            from rsc.fewshot_utils import format_options
            options_str = format_options(options)
            question = f"{question}\n{options_str}"
        
        # 모든 데이터셋에 대해 "Answer: " 추가 (MMLU-Pro 형식 통일)
        question = f"{question}\nAnswer: "
        
        # prompt_template이 None이면 기본 템플릿 사용
        if prompt_template is None:
            prompt_template = self.prompt_template
        
        # 프롬프트 템플릿이 없으면 기본 형식 사용
        if prompt_template is None or not prompt_template:
            human_content = f"{fewshot_context}\n\n{question}" if fewshot_context else question
            messages = [
                {"role": "user", "content": human_content}
            ]
        else:
            # 기존 코드의 프롬프트 구성 로직 활용 (SC_generator_ours.py 410-424줄)
            system_template_dict = prompt_template.get('system', {})
            tmpl_key = _prompt_template_dataset_key(dataset_name)
            if isinstance(system_template_dict, dict):
                # Qwen3 모델이면 dataset_Qwen3 키를 우선 시도
                if self.is_qwen3 and tmpl_key:
                    qwen3_key = f"{tmpl_key}_Qwen3"
                    system_template = system_template_dict.get(
                        qwen3_key,
                        system_template_dict.get(tmpl_key, system_template_dict.get('default', ''))
                    )
                elif tmpl_key:
                    system_template = system_template_dict.get(
                        tmpl_key,
                        system_template_dict.get('default', '')
                    )
                else:
                    system_template = system_template_dict.get('default', '')
            else:
                # 기존 형식 (문자열)인 경우 하위 호환성 유지
                system_template = system_template_dict if isinstance(system_template_dict, str) else ''
            
            # Prompt template에서 system과 human 메시지 가져오기 (기존 코드 450-460줄)
            system_content = system_template.format(subject=subject)
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
        
        # Chat template 적용 (기존 코드 473-481줄)
        apply_template_kwargs = {
            'conversation': messages,
            'tokenize': False,
            'add_generation_prompt': True
        }
        if self.is_qwen3:
            apply_template_kwargs['enable_thinking'] = self.enable_thinking
        
        prompt_string = self.tokenizer.apply_chat_template(**apply_template_kwargs)
        return prompt_string
    
    def _build_refinement_prompt(self, question: str, previous_stage_paths: List[str],
                                 dataset_name: str = None, subject: str = 'math',
                                 stage_number: int = 2) -> str:
        """
        Refinement 프롬프트 구성 (Collective 방식: 모든 이전 Stage paths 포함)
        refinement.json의 collective 템플릿 사용 (데이터셋별 템플릿 지원)
        
        Args:
            question: 원본 질문
            previous_stage_paths: 이전 Stage reasoning paths
            dataset_name: 데이터셋 이름 (예: 'AIME_2024', 'MMLU-Pro')
            subject: 주제 (예: 'math', 'science', 'general')
            stage_number: 현재 Stage 번호
            
        Returns:
            str: 구성된 refinement 프롬프트
        """
        # refinement.json의 collective 템플릿 사용
        template = self.refinement_template.get('collective', {})
        
        # 데이터셋별 템플릿 선택 (prompt.json과 동일한 로직)
        tmpl_key = _prompt_template_dataset_key(dataset_name)
        if tmpl_key and tmpl_key in template:
            template_str = template[tmpl_key]
        else:
            template_str = template.get('default', '질문: {question}\n\n이전 추론 과정들:\n{previous_reasonings}\n\n새로운 추론:')
        
        format_reasoning = template.get('format_reasoning', '[추론 {index}]\n{path}')
        
        # 이전 Stage paths 포맷팅
        previous_reasonings = "\n\n".join([
            format_reasoning.format(index=i+1, path=path) for i, path in enumerate(previous_stage_paths)
        ])
        
        # 템플릿에 값 삽입
        refinement_content = template_str.format(
            question=question,
            previous_reasonings=previous_reasonings,
            subject=subject
        )
        
        messages = [
            {"role": "user", "content": refinement_content}
        ]
        
        apply_template_kwargs = {
            'conversation': messages,
            'tokenize': False,
            'add_generation_prompt': True
        }
        if self.is_qwen3:
            apply_template_kwargs['enable_thinking'] = self.enable_thinking
        
        prompt_string = self.tokenizer.apply_chat_template(**apply_template_kwargs)
        return prompt_string
    
    def _build_refinement_prompt_individual(self, question: str, previous_stage_path: str, 
                                            path_index: int, dataset_name: str = None, 
                                            subject: str = 'math', stage_number: int = 2) -> str:
        """
        Refinement 프롬프트 구성 (Individual 방식: 단일 이전 Stage path만 포함)
        refinement.json의 individual 템플릿 사용 (데이터셋별 템플릿 지원)
        
        Args:
            question: 원본 질문
            previous_stage_path: 이전 Stage reasoning path (단일)
            path_index: Path 인덱스 (표시용)
            dataset_name: 데이터셋 이름 (예: 'AIME_2024', 'MMLU-Pro')
            subject: 주제 (예: 'math', 'science', 'general')
            stage_number: 현재 Stage 번호
            
        Returns:
            str: 구성된 refinement 프롬프트
        """
        # refinement.json의 individual 템플릿 사용
        template = self.refinement_template.get('individual', {})
        
        # 데이터셋별 템플릿 선택 (prompt.json과 동일한 로직)
        tmpl_key = _prompt_template_dataset_key(dataset_name)
        if tmpl_key and tmpl_key in template:
            template_str = template[tmpl_key]
        else:
            template_str = template.get('default', '질문: {question}\n\n이전 추론 과정:\n[추론 {path_index}]\n{previous_reasoning}\n\n새로운 추론:')
        
        # 템플릿에 값 삽입
        refinement_content = template_str.format(
            question=question,
            path_index=path_index,
            previous_reasoning=previous_stage_path,
            subject=subject
        )
        
        messages = [
            {"role": "user", "content": refinement_content}
        ]
        
        apply_template_kwargs = {
            'conversation': messages,
            'tokenize': False,
            'add_generation_prompt': True
        }
        if self.is_qwen3:
            apply_template_kwargs['enable_thinking'] = self.enable_thinking
        
        prompt_string = self.tokenizer.apply_chat_template(**apply_template_kwargs)
        return prompt_string
    
    def _sample_paths_by_answer_diversity(self, previous_stage_paths: List[str], num_paths: int,
                                         dataset_name: str = None) -> List[str]:
        """
        답변의 다양성을 고려하여 이전 Stage paths 샘플링
        
        각 path에서 답변을 추출하고, 답변별로 그룹화한 후
        각 답변 그룹에서 균등하게 샘플링하여 다양성 확보
        
        Args:
            previous_stage_paths: 이전 Stage reasoning path 리스트
            num_paths: 선택할 path 개수
            dataset_name: 데이터셋 이름 (답변 추출 함수 선택용)
            
        Returns:
            List[str]: 선택된 reasoning path 리스트
        """
        # Aggregator를 사용하여 각 path에서 답변 추출
        aggregator = SCAggregator(dataset_name=dataset_name)
        
        # 답변별로 paths 그룹화
        answer_to_paths = defaultdict(list)
        for i, path in enumerate(previous_stage_paths):
            answer = aggregator.extract_answer(path)
            # 답변이 없거나 빈 문자열인 경우 "unknown"으로 처리
            answer_key = str(answer) if answer else "unknown"
            answer_to_paths[answer_key].append((i, path))
        
        # 답변 그룹이 비어있는 경우 (답변 추출 실패) 랜덤 샘플링
        if not answer_to_paths:
            return random.sample(previous_stage_paths, min(num_paths, len(previous_stage_paths)))
        
        # 각 답변 그룹에서 균등하게 샘플링 (라운드로빈 방식)
        selected_paths = []
        selected_indices = set()
        
        group_keys = list(answer_to_paths.keys())
        random.shuffle(group_keys)  # 그룹 순서 랜덤화
        
        # 라운드로빈 방식으로 각 그룹에서 하나씩 선택
        round_idx = 0
        while len(selected_paths) < num_paths:
            # 현재 라운드에서 선택할 수 있는 그룹들
            available_groups = [
                key for key in group_keys 
                if round_idx < len(answer_to_paths[key])
            ]
            
            if not available_groups:
                # 더 이상 선택할 수 있는 그룹이 없으면 종료
                break
            
            # 각 그룹에서 하나씩 선택
            for answer_key in available_groups:
                if len(selected_paths) >= num_paths:
                    break
                
                group = answer_to_paths[answer_key]
                if round_idx < len(group):
                    idx, path = group[round_idx]
                    if idx not in selected_indices:
                        selected_paths.append(path)
                        selected_indices.add(idx)
            
            round_idx += 1
        
        # 아직 선택하지 못한 경우 나머지는 랜덤으로 채우기
        if len(selected_paths) < num_paths:
            remaining_paths = [(i, path) for i, path in enumerate(previous_stage_paths) if i not in selected_indices]
            if remaining_paths:
                num_needed = num_paths - len(selected_paths)
                additional = random.sample(remaining_paths, min(num_needed, len(remaining_paths)))
                selected_paths.extend([path for _, path in additional])
        
        return selected_paths
