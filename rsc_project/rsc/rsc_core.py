"""
RSC (Recursive Self-Consistency) Core Orchestrator 모듈

이 모듈은 RSC 알고리즘의 전체 오케스트레이션을 담당합니다.
Multi-stage RSC를 지원하며, k개의 Stage를 반복적으로 실행합니다.
Stage 1: 기본 Self-Consistency
Stage t (t ≥ 2): Reflective Self-Consistency
최종 답변 결정
"""

from typing import List, Dict, Any, Optional, Union
from .generator import ReasoningPathGenerator
from .aggregator import SCAggregator


class RSCEngine:
    """
    RSC 알고리즘의 핵심 엔진 클래스
    
    전체 RSC 파이프라인을 실행합니다:
    - Stage 1: 기본 Self-Consistency (질문 q에 대해 n₁개의 reasoning chain 생성)
    - Stage t (t ≥ 2): Reflective Self-Consistency (이전 Stage의 chain 집합을 입력으로 사용)
    - 최종 답변 결정 (다양한 전략 지원)
    
    Multi-stage 지원: n × k 형태 (예: 10 × 4 = 10개 chain × 4 stage)
    """
    
    def __init__(self, generator: ReasoningPathGenerator, 
                 aggregator: SCAggregator = None,
                 final_answer_mode: str = 'stage_k_only'):
        """
        Args:
            generator: ReasoningPathGenerator 인스턴스
            aggregator: SCAggregator 인스턴스 (None이면 기본 생성)
            final_answer_mode: 최종 답변 결정 모드
                - 'stage_k_only': 마지막 Stage(k)의 majority vote 결과 사용 (기본값)
                - 'stage1_stagek_agreement': Stage1과 Stage k가 일치하면 그 답, 다르면 Stage k
                - 'majority': 모든 Stage 답변의 majority vote
        """
        self.generator = generator
        self.aggregator = aggregator or SCAggregator()
        self.final_answer_mode = final_answer_mode
        
        if final_answer_mode not in ['stage_k_only', 'stage1_stagek_agreement', 'majority']:
            raise ValueError(
                "final_answer_mode는 'stage_k_only', 'stage1_stagek_agreement', 또는 'majority'여야 합니다."
            )
    
    def run(self, question: str, 
            stages: List[Dict[str, Any]],
            temperature: float = 0.7, max_tokens: int = 10240,
            top_p: float = 0.95, prompt_template: Dict = None,
            fewshot_context: str = '', dataset_name: str = None,
            subject: str = 'math', options: List[str] = None) -> Dict[str, Any]:
        """
        RSC 알고리즘 실행 (Multi-stage 지원)
        
        Args:
            question: 입력 질문
            stages: Stage 설정 리스트 (필수)
                예: [
                    {'n': 20},  # Stage 1
                    {'n': 5, 'mode': 'collective'},  # Stage 2
                    {'n': 5, 'mode': 'individual'},  # Stage 3
                    {'n': 5, 'mode': 'collective', 'collective_num_paths': 10}  # Stage 4
                ]
                - Stage 1: 'n'만 지정 (mode 무시)
                - Stage t (t ≥ 2): 
                    - 'n': 생성할 reasoning path 개수 (collective 모드에서만 사용)
                    - 'mode': 'collective' 또는 'individual' (필수)
                    - 'collective_num_paths': Collective 모드에서 사용할 이전 Stage path 개수 (선택)
            temperature: 생성 시 사용할 temperature
            max_tokens: 최대 생성 토큰 수
            top_p: Top-p 샘플링 파라미터
            prompt_template: 프롬프트 템플릿 딕셔너리
            fewshot_context: Few-shot 예제 컨텍스트
            dataset_name: 데이터셋 이름
            subject: 주제 (math, science, general)
            options: 선택지 리스트 (multiple-choice 문제용)
            
        Returns:
            Dict[str, Any]: RSC 실행 결과
                - question: 입력 질문
                - stages: 각 Stage별 결과 리스트
                    각 요소: {
                        'stage_number': int,
                        'paths': List[str],
                        'answer': str,
                        'token_info': Dict,
                        'distribution': Dict,
                        'mode': str (Stage 2 이상)
                    }
                - final_answer: 최종 답변
                - final_answer_mode: 최종 답변 결정 모드
                - metadata: 추가 메타데이터
        """
        if not stages or len(stages) == 0:
            raise ValueError("stages 리스트가 비어있습니다. 최소 1개의 Stage가 필요합니다.")
        
        # Stage 1: 기본 Self-Consistency
        stage1_config = stages[0]
        stage1_n = stage1_config['n']
        
        stage1_paths, stage1_token_info = self.generator.generate_stage1_paths(
            question, n_paths=stage1_n, temperature=temperature,
            max_tokens=max_tokens, top_p=top_p,
            prompt_template=prompt_template, fewshot_context=fewshot_context,
            dataset_name=dataset_name, subject=subject, options=options
        )
        stage1_answer = self.aggregator.aggregate(stage1_paths)
        
        # Stage 결과 저장
        stage_results = [
            {
                'stage_number': 1,
                'paths': stage1_paths,
                'answer': stage1_answer,
                'token_info': stage1_token_info,
                'distribution': self.aggregator.get_answer_distribution(stage1_paths)
            }
        ]
        
        # Stage t (t ≥ 2): Reflective Self-Consistency
        previous_paths = stage1_paths
        for stage_idx, stage_config in enumerate(stages[1:], start=2):
            if 'mode' not in stage_config:
                raise ValueError(f"Stage {stage_idx}: 'mode' 필드가 필요합니다 ('collective' 또는 'individual')")
            
            mode = stage_config['mode']
            
            # Individual 모드: 이전 Stage의 chain 개수만큼만 생성 (각 chain마다 1:1 refinement)
            # Collective 모드: n_paths 파라미터 사용 (이전 chain들을 참고하여 새로운 n_paths개 생성)
            if mode == 'individual':
                n_paths = len(previous_paths)  # Individual 모드는 무조건 이전 Stage paths 개수와 동일
            else:  # collective
                n_paths = stage_config.get('n', len(previous_paths))  # 기본값: 이전 Stage path 개수
            
            stage_collective_num_paths = stage_config.get('collective_num_paths', None)
            
            # Reflective generation
            stage_paths, stage_token_info = self.generator.generate_stage_t_paths(
                question=question,
                previous_stage_paths=previous_paths,
                stage_number=stage_idx,
                n_paths=n_paths,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                refinement_prompt=None,
                mode=mode,
                dataset_name=dataset_name,
                subject=subject,
                collective_num_paths=stage_collective_num_paths
            )
            stage_answer = self.aggregator.aggregate(stage_paths)
            
            # Stage 결과 저장
            stage_results.append({
                'stage_number': stage_idx,
                'paths': stage_paths,
                'answer': stage_answer,
                'token_info': stage_token_info,
                'distribution': self.aggregator.get_answer_distribution(stage_paths),
                'mode': mode
            })
            
            # 다음 Stage를 위한 업데이트
            previous_paths = stage_paths
        
        # 최종 답변 결정
        final_answer = self._determine_final_answer_multi_stage(stage_results)
        
        # 결과 반환
        result = {
            'question': question,
            'stages': stage_results,
            'final_answer': final_answer,
            'final_answer_mode': self.final_answer_mode,
            'metadata': {
                'num_stages': len(stage_results),
                'temperature': temperature,
                'max_tokens': max_tokens,
                'top_p': top_p,
            }
        }
        
        return result
    
    def _determine_final_answer_multi_stage(self, stage_results: List[Dict[str, Any]]) -> str:
        """
        Multi-stage 최종 답변 결정 (내부 메서드)
        
        Args:
            stage_results: 각 Stage별 결과 리스트
                각 요소: {'stage_number': int, 'answer': str, ...}
            
        Returns:
            str: 최종 답변
        """
        if not stage_results:
            raise ValueError("stage_results가 비어있습니다.")
        
        stage1_answer = stage_results[0]['answer']
        stage_k_answer = stage_results[-1]['answer']
        
        if self.final_answer_mode == 'stage_k_only':
            # 마지막 Stage(k)의 답변 사용
            return stage_k_answer
        
        elif self.final_answer_mode == 'stage1_stagek_agreement':
            # Stage 1과 Stage k가 일치하면 그 답변, 다르면 Stage k
            if stage1_answer == stage_k_answer:
                return stage1_answer
            else:
                return stage_k_answer
        
        elif self.final_answer_mode == 'majority':
            # 모든 Stage 답변의 majority vote
            from collections import Counter
            all_answers = [result['answer'] for result in stage_results]
            answer_counts = Counter(all_answers)
            return answer_counts.most_common(1)[0][0]
        
        else:
            # 기본값: 마지막 Stage 답변
            return stage_k_answer

