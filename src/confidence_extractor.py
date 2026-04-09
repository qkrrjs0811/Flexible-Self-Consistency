"""
BINARY_FT01 Confidence Extraction 모듈

use_p_true.md에 설명된 BINARY_FT01 방식의 confidence 추출을 구현합니다.
"""
import dataclasses
import torch
import numpy as np
import re
from typing import Optional, List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
import abc


@dataclasses.dataclass
class ConfidenceExtractionConfig:
    """Confidence 추출 설정"""
    prompt: str  # Confidence 추출을 위한 프롬프트
    confidence_options: List[str] | None = None  # 선택 가능한 옵션들 (예: ["0", "1"])
    temperature: float = 0.0
    num_tokens: int = 10
    verbal_eos_pattern: str = ")"


# BINARY_FT01 기본 설정
BINARY_FT01 = ConfidenceExtractionConfig(
    prompt="""
Do you believe the proposed answer is accurate? Select either
  (0) if you think the proposed answer may be incorrect
  (1) if you are certain the proposed answer is correct
Selection: (""",
    confidence_options=["0", "1"],
)


class Runner(abc.ABC):
    """모델 실행을 위한 인터페이스"""
    
    @abc.abstractmethod
    def get_completion_likelihoods(
        self,
        prefixes: List[str],
        completions: List[str],
        enable_formatting: bool = False,
    ) -> List[List[float]]:
        """각 prefix에 대해 각 completion의 likelihood 반환
        
        Args:
            prefixes: 입력 프롬프트 리스트
            completions: 평가할 completion 토큰들 (예: ["0", "1"])
            enable_formatting: 포맷팅 활성화 여부
            
        Returns:
            각 prefix에 대한 likelihood 리스트
            예: [[log_prob_0, log_prob_1], ...]
        """
        raise NotImplementedError()


class HuggingFaceRunner(Runner):
    """HuggingFace 모델을 사용한 Runner 구현"""
    
    def __init__(self, model_name: str, device: str = "cuda"):
        """Runner 초기화
        
        Args:
            model_name: HuggingFace 모델 이름 또는 경로
            device: 사용할 디바이스 ("cuda" 또는 "cpu")
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left",
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        # pad_token_id 설정
        if self.tokenizer.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id = (
                self.tokenizer.eos_token_id
            )
        self.device = device
    
    def get_completion_likelihoods(
        self,
        prefixes: List[str],
        completions: List[str],
        enable_formatting: bool = False,
    ) -> List[List[float]]:
        """각 completion 토큰의 log probability 계산"""
        # 완료는 단일 토큰이어야 함
        completion_ids = []
        for completion in completions:
            # 토큰 ID로 변환 시도
            completion_id = self.tokenizer.convert_tokens_to_ids(completion)
            if completion_id == self.tokenizer.unk_token_id:
                # 직접 인코딩 시도
                tokenized = self.tokenizer.encode(completion, add_special_tokens=False)
                if len(tokenized) == 1:
                    completion_id = tokenized[0]
                else:
                    raise ValueError(
                        f"completion '{completion}' must be a single valid token. "
                        f"Tokenized to {len(tokenized)} tokens: {tokenized}"
                    )
            completion_ids.append(completion_id)
        
        # 한 토큰만 생성하여 scores 얻기
        results = []
        for prefix in prefixes:
            inputs = self.tokenizer(
                prefix,
                return_tensors="pt",
                add_special_tokens=False,
            )
            if torch.cuda.is_available() and self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[0, -1, :]  # 마지막 토큰의 logits
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                
                # 각 completion ID의 log probability 추출
                likelihoods = [log_probs[id].item() for id in completion_ids]
                results.append(likelihoods)
        
        return results


class VLLMRunner(Runner):
    """vLLM을 사용한 Runner 구현 - logprobs 기능 활용"""
    
    def __init__(self, llm, tokenizer):
        """Runner 초기화
        
        Args:
            llm: vLLM의 LLM 객체
            tokenizer: 토크나이저 객체
        """
        self.llm = llm
        self.tokenizer = tokenizer
    
    def get_completion_likelihoods(
        self,
        prefixes: List[str],
        completions: List[str],
        enable_formatting: bool = False,
    ) -> List[List[float]]:
        """각 completion 토큰의 log probability 계산
        
        vLLM의 generate 메서드의 logprobs 기능을 사용합니다.
        """
        from vllm import SamplingParams
        
        # completion 토큰들의 ID 확인
        completion_ids = []
        for completion in completions:
            # 토큰 ID로 변환 시도
            try:
                completion_id = self.tokenizer.convert_tokens_to_ids(completion)
            except:
                completion_id = None
            
            if completion_id is None or completion_id == self.tokenizer.unk_token_id:
                # 직접 인코딩 시도
                tokenized = self.tokenizer.encode(completion, add_special_tokens=False)
                if len(tokenized) == 1:
                    completion_id = tokenized[0]
                else:
                    raise ValueError(
                        f"completion '{completion}' must be a single valid token. "
                        f"Tokenized to {len(tokenized)} tokens: {tokenized}"
                    )
            completion_ids.append(completion_id)
        
        # 각 prefix에 대해 vLLM generate를 사용하여 logprobs 얻기
        # logprobs 파라미터를 completion 개수만큼 설정 (최소 2개)
        logprobs_num = max(len(completion_ids), 2)
        
        # Confidence 추출을 위한 샘플링 파라미터
        # temperature=0.0으로 설정하여 확정적인 확률 계산
        confidence_params = SamplingParams(
            logprobs=logprobs_num,
            temperature=0.0,
            max_tokens=1,  # 첫 번째 토큰만 생성
        )
        
        results = []
        for prefix in prefixes:
            try:
                # vLLM generate 호출
                outputs = self.llm.generate([prefix], confidence_params)
                
                if not outputs or not outputs[0].outputs:
                    raise RuntimeError("vLLM generate가 출력을 생성하지 못했습니다.")
                
                # 첫 번째 출력의 첫 번째 토큰의 logprobs 추출
                output = outputs[0]
                generation = output.outputs[0]
                
                # logprobs가 있는지 확인
                if not hasattr(generation, 'logprobs') or generation.logprobs is None:
                    raise RuntimeError("vLLM 출력에 logprobs가 없습니다. logprobs 파라미터를 확인하세요.")
                
                # logprobs는 리스트 형태: [TokenLogprob, ...]
                # 각 TokenLogprob는 토큰 정보와 logprob를 포함
                logprobs_list = generation.logprobs
                
                if not logprobs_list or len(logprobs_list) == 0:
                    raise RuntimeError("logprobs 리스트가 비어있습니다.")
                
                # 첫 번째 토큰의 logprobs 사용
                first_token_logprobs = logprobs_list[0]
                
                # vLLM의 logprobs 구조에 따라 다르게 처리
                # 일반적으로 TokenLogprob 객체이거나 딕셔너리 형태
                logprob_dict = {}
                
                # 방법 1: TokenLogprob 객체인 경우 - logprob_dict 속성 확인
                if hasattr(first_token_logprobs, 'logprob_dict'):
                    logprob_dict = first_token_logprobs.logprob_dict
                # 방법 2: 직접 딕셔너리인 경우
                elif isinstance(first_token_logprobs, dict):
                    logprob_dict = first_token_logprobs
                # 방법 3: TokenLogprob 객체의 다른 속성들 확인
                else:
                    # 객체의 속성들을 확인하여 logprob 정보 추출
                    # vLLM의 TokenLogprob는 보통 token_id와 logprob를 포함
                    if hasattr(first_token_logprobs, '__dict__'):
                        attrs = first_token_logprobs.__dict__
                        # logprob 관련 속성 찾기
                        for key, value in attrs.items():
                            if isinstance(value, dict):
                                logprob_dict = value
                                break
                    
                    # 여전히 찾지 못한 경우, 객체의 직접 속성 확인
                    if not logprob_dict:
                        # decoded_token과 logprob 속성이 있는 경우
                        if hasattr(first_token_logprobs, 'decoded_token') and hasattr(first_token_logprobs, 'logprob'):
                            token_str = first_token_logprobs.decoded_token
                            try:
                                token_id = self.tokenizer.convert_tokens_to_ids(token_str)
                                logprob_dict[token_id] = first_token_logprobs.logprob
                            except:
                                pass
                
                # completion ID들의 log probability 추출
                likelihoods = []
                for comp_id in completion_ids:
                    if comp_id in logprob_dict:
                        logprob_value = logprob_dict[comp_id]
                        # Logprob 객체인 경우 실제 float 값 추출
                        if hasattr(logprob_value, 'logprob'):
                            logprob_value = logprob_value.logprob
                        elif hasattr(logprob_value, 'value'):
                            logprob_value = logprob_value.value
                        elif hasattr(logprob_value, '__float__'):
                            logprob_value = float(logprob_value)
                        # 이미 float인 경우 그대로 사용
                        if not isinstance(logprob_value, (int, float)):
                            # 여전히 객체인 경우 에러
                            raise RuntimeError(
                                f"logprob 값이 숫자로 변환할 수 없습니다: {type(logprob_value)}, "
                                f"value: {logprob_value}, attrs: {dir(logprob_value) if hasattr(logprob_value, '__dict__') else 'N/A'}"
                            )
                        likelihoods.append(float(logprob_value))
                    else:
                        # logprob_dict에 없으면 -inf 사용
                        # (해당 토큰이 top-k에 포함되지 않았을 수 있음)
                        likelihoods.append(float('-inf'))
                
                # 모든 값이 -inf인 경우 에러
                if all(l == float('-inf') for l in likelihoods):
                    # 디버깅 정보 수집
                    debug_parts = [
                        f"logprobs type: {type(first_token_logprobs)}",
                    ]
                    if hasattr(first_token_logprobs, '__dict__'):
                        debug_parts.append(f"attrs: {list(first_token_logprobs.__dict__.keys())}")
                    if logprob_dict:
                        debug_parts.append(f"logprob_dict keys (first 10): {list(logprob_dict.keys())[:10]}")
                    else:
                        debug_parts.append("logprob_dict: empty")
                    
                    raise RuntimeError(
                        f"Completion 토큰들 {completion_ids}의 logprob를 찾을 수 없습니다. "
                        f"{'; '.join(debug_parts)}. "
                        f"logprobs={logprobs_num}을 더 크게 설정하거나 토큰 ID를 확인하세요."
                    )
                
                results.append(likelihoods)
            except Exception as e:
                raise RuntimeError(
                    f"vLLM generate에서 logprobs 추출 실패: {e}. "
                    f"Error type: {type(e).__name__}. "
                    "vLLM 버전이나 logprobs 파라미터를 확인하세요."
                )
        
        return results


def softmax(nums: List[float] | np.ndarray, temp: float = 1.0) -> np.ndarray:
    """Softmax 함수"""
    if temp == 0:
        raise ValueError("Temperature cannot be 0.")
    nums = np.array(nums) / temp
    
    # 수치 안정성을 위한 조정
    nums = nums - np.max(nums)
    
    e = np.exp(nums)
    s = e.sum()
    if s == 0:
        raise ValueError("Sum of exp(nums) is 0.")
    return e / s


def likelihood_to_confidence(likelihoods: List[float] | None) -> float | None:
    """Likelihood를 confidence 점수로 변환
    
    Args:
        likelihoods: 각 옵션의 log probability [log_prob_0, log_prob_1]
        
    Returns:
        confidence: "1" 옵션의 확률 (0~1 사이 값)
    """
    if likelihoods is None or len(likelihoods) != 2:
        return None
    
    # Softmax 적용하여 확률로 변환
    probs = softmax(likelihoods, temp=1.0)
    
    # "1" (정확하다고 믿는) 옵션의 확률 반환
    return float(probs[1])


def find_answer_span(response: str) -> Tuple[int, int] | None:
    """응답에서 답변 부분의 위치를 찾습니다.
    
    extract_math_answer와 동일한 로직을 사용하여 답변 위치를 찾습니다.
    
    Args:
        response: 모델이 생성한 응답
        
    Returns:
        (start, end) 튜플 또는 None
    """
    if not response:
        return None
    
    response_str = str(response).strip()
    if not response_str:
        return None
    
    # extract_math_answer와 동일한 순서로 패턴 찾기
    # 1. "The answer is " 패턴
    if 'The answer is ' in response_str:
        idx = response_str.find('The answer is ')
        start = idx + len('The answer is ')
        # 답변 부분의 끝 찾기 (줄바꿈이나 마침표까지)
        end = len(response_str)
        for stop_char in ['\n', '.', '\n\n']:
            stop_idx = response_str.find(stop_char, start)
            if stop_idx != -1:
                end = min(end, stop_idx)
        return (start, end)
    
    # 2. "the answer is " 패턴
    elif 'the answer is ' in response_str:
        idx = response_str.find('the answer is ')
        start = idx + len('the answer is ')
        end = len(response_str)
        for stop_char in ['\n', '.', '\n\n']:
            stop_idx = response_str.find(stop_char, start)
            if stop_idx != -1:
                end = min(end, stop_idx)
        return (start, end)
    
    # 3. "boxed" 패턴 (가장 중요 - extract_math_answer의 주요 로직)
    elif 'boxed' in response_str:
        idx = response_str.find('boxed')
        # "boxed" 이후의 내용을 찾기
        ans_start = idx + len('boxed')
        ans = response_str[ans_start:]
        
        if len(ans) and ans[0] == '{':
            # 중괄호 안의 내용 찾기
            stack = 1
            end_offset = 1  # '{' 다음부터 시작
            for i, c in enumerate(ans[1:], start=1):
                if c == '{':
                    stack += 1
                elif c == '}':
                    stack -= 1
                    if stack == 0:
                        end_offset = i + 1  # '}' 포함
                        break
            # boxed{...} 전체를 포함
            end = ans_start + end_offset
            return (idx, end)  # "boxed" 시작부터 답변 끝까지
        else:
            # '$' 또는 줄바꿈까지
            dollar_idx = ans.find('$')
            newline_idx = ans.find('\n')
            if dollar_idx != -1 and newline_idx != -1:
                end_offset = min(dollar_idx, newline_idx)
            elif dollar_idx != -1:
                end_offset = dollar_idx
            elif newline_idx != -1:
                end_offset = newline_idx
            else:
                end_offset = len(ans)
            end = ans_start + end_offset
            return (idx, end)  # "boxed" 시작부터 답변 끝까지
    
    # 4. 숫자 패턴 (extract_math_answer의 fallback)
    else:
        pattern = r'-?\d*\.?\d+'
        matches = re.findall(pattern, response_str)
        if len(matches) >= 1:
            # 마지막 숫자의 위치 찾기
            last_match = matches[-1]
            last_idx = response_str.rfind(last_match)
            if last_idx != -1:
                start = last_idx
                end = last_idx + len(last_match)
                return (start, end)
    
    # 패턴을 찾지 못한 경우 전체 응답 반환
    return (0, len(response_str) - 1)


def generate_confidence(
    original_prompt: str,
    original_response: str,
    answer_span: Tuple[int, int] | None,
    runner: Runner,
    config: ConfidenceExtractionConfig,
) -> List[float] | None:
    """Confidence 추출 메인 함수
    
    Args:
        original_prompt: 원본 프롬프트
        original_response: 모델이 생성한 원본 응답 (CoT 전체)
        answer_span: 응답에서 답변의 위치 (start, end). None이면 자동으로 찾음
        runner: 모델 실행을 위한 Runner
        config: Confidence 추출 설정
        
    Returns:
        confidence_likelihoods: 각 옵션에 대한 log probability 리스트
        예: [log_prob_0, log_prob_1]
    """
    # answer_span이 None이면 자동으로 찾기
    if answer_span is None:
        answer_span = find_answer_span(original_response)
        if answer_span is None:
            return None
    
    # 답변까지만 포함 (CoT 전체를 사용하되, 답변 부분까지만)
    # 이렇게 하면 각 path의 reasoning 과정이 confidence에 반영됨
    answer_text = original_response[: answer_span[1] + 1]
    
    # Full prompt 구성
    full_prompt = (
        original_prompt
        + answer_text
        + config.prompt
    )
    
    # Likelihood 계산
    if config.confidence_options is not None:
        try:
            likelihoods = runner.get_completion_likelihoods(
                [full_prompt],
                completions=config.confidence_options,
                enable_formatting=False,
            )[0]
            return likelihoods
        except Exception as e:
            print(f"[WARNING] Confidence 추출 실패: {e}")
            return None
    
    return None

