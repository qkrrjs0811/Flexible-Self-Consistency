"""
RSC (Recursive Self-Consistency) 프로젝트 메인 실행 파일

이 파일은 RSC 알고리즘을 실행하는 진입점입니다.
- 단일 질문에 대한 RSC inference 예제
- 전체 데이터셋 평가 실행
"""

import argparse
import yaml
import json
import os
import sys

# 스크립트가 있는 디렉토리를 Python 모듈 검색 경로에 추가
# 절대 경로로 실행할 때 상대 import가 작동하도록 함
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 경로 설정: SCRIPT_DIR을 최우선으로 추가 (중복 제거 후)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# evaluation 모듈을 먼저 import (rsc 모듈이 src 경로를 추가하기 전에)
# 이렇게 하면 evaluation 패키지와 src/evaluation.py 파일 이름 충돌 방지
from evaluation.run_eval import run_evaluation

# rsc 모듈 import (이 모듈들이 src 경로를 추가할 수 있음)
from rsc.rsc_core import RSCEngine
from rsc.generator import ReasoningPathGenerator
from rsc.aggregator import SCAggregator
from rsc.fewshot_utils import build_fewshot_context


def main():
    """
    메인 함수: 명령줄 인자를 파싱하여 적절한 모드를 실행
    """
    parser = argparse.ArgumentParser(description='RSC (Recursive Self-Consistency) 실행')
    parser.add_argument('--mode', type=str, choices=['inference', 'eval'], 
                       default='inference', help='실행 모드: inference 또는 eval')
    parser.add_argument('--config', type=str, default=None,
                       help='설정 파일 경로 (필수)')
    parser.add_argument('--question', type=str, default=None,
                       help='단일 질문 (inference 모드에서 사용)')
    parser.add_argument('--dataset', type=str, default=None,
                       help='데이터셋 이름 (inference 모드에서 사용, 예: AIME_2024, MMLU-Pro)')
    
    args = parser.parse_args()
    
    # 설정 파일 검증
    if not args.config:
        parser.error('설정 파일을 지정하세요 (--config <path>). run_all_experiments.sh 를 사용하면 자동으로 생성됩니다.')
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {args.config}")
    
    if args.mode == 'inference':
        # 단일 질문에 대한 RSC inference 예제
        run_single_inference(args.config, args.question, args.dataset)
    elif args.mode == 'eval':
        # 전체 데이터셋 평가
        run_evaluation(args.config)


def run_single_inference(config_path: str = 'experiments/config.yaml',
                        question: str = None, dataset_name: str = None):
    """
    단일 질문에 대한 RSC inference 예제 실행
    
    Args:
        config_path: 설정 파일 경로
        question: 질문 (None이면 기본 예제 사용)
        dataset_name: 데이터셋 이름 (None이면 설정 파일에서 가져옴)
    """
    # 설정 파일 로드
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 데이터셋 이름 결정
    if dataset_name is None:
        dataset_name = config['dataset']['name']
    
    # 데이터셋 이름과 subject 매핑 (기존 main_exp.sh 로직)
    DATASET_SUBJECT_MAP = {
        'AIME_2024': 'math',
        'AIME_2025': 'math',
        'MATH500_test': 'math',
        'MATH_train': 'math',
        'MATH2500': 'math',
        'AMC23': 'math',
        'GPQA-Diamond': 'science',
        'MMLU-Pro': 'general',
        'MMLU-Pro_full': 'general',
    }
    subject = config.get('subject') or DATASET_SUBJECT_MAP.get(dataset_name, 'math')
    
    # Few-shot 컨텍스트 구성
    fewshot_context = build_fewshot_context(
        dataset_name,
        disabled=config.get('disable_fewshot', False)
    )
    
    # 프롬프트 템플릿 로드
    prompt_template_path = config.get('prompt_template_path', None)
    if prompt_template_path and os.path.exists(prompt_template_path):
        with open(prompt_template_path, 'r', encoding='utf-8') as f:
            prompt_template = json.load(f)
    else:
        prompt_template = None
    
    # Refinement 템플릿 경로
    refinement_template_path = config.get('refinement_template_path', None)
    
    # RSC 엔진 초기화
    print(f"모델 로딩 중: {config['model']['name']}")
    generator = ReasoningPathGenerator(
        model_name=config['model']['name'],
        device=config['model']['device'],
        gpu_memory_utilization=config['model'].get('gpu_memory_utilization', 0.85),
        max_model_len=config['model'].get('max_model_len', 16384),
        enable_thinking=config['model'].get('enable_thinking', False),
        prompt_template_path=prompt_template_path,
        refinement_template_path=refinement_template_path
    )
    print("모델 로딩 완료!")
    
    aggregator = SCAggregator(dataset_name=dataset_name)
    
    rsc_engine = RSCEngine(
        generator=generator,
        aggregator=aggregator,
        final_answer_mode=config['rsc']['final_answer_mode']
    )
    
    # 질문 결정
    if question is None:
        # 기본 예제 질문
        if dataset_name in ['AIME_2024', 'AIME_2025', 'MATH500_test', 'MATH_train', 'MATH2500', 'AMC23']:
            question = "If $(2^3)^{(2^3)} = 2^N$, what is the value of $N$?"
        elif dataset_name in ('MMLU-Pro', 'MMLU-Pro_full'):
            question = "Question: What is the capital of France?\nOptions:\nA. London\nB. Berlin\nC. Paris\nD. Madrid"
        elif dataset_name == 'GPQA-Diamond':
            question = "What is the chemical formula for water?\nA. H2O\nB. CO2\nC. NaCl\nD. O2"
        else:
            question = "What is 2 + 2?"
    
    # Multiple-choice 문제의 경우 options 추출
    options = None
    if dataset_name in ['MMLU-Pro', 'MMLU-Pro_full', 'GPQA-Diamond']:
        # 질문에서 options 추출 (간단한 파싱)
        if 'Options:' in question or '\nA.' in question:
            lines = question.split('\n')
            options = []
            for line in lines:
                if line.strip().startswith(('A.', 'B.', 'C.', 'D.', 'E.')):
                    options.append(line.strip()[2:].strip())
    
    # stages 설정 로드
    if 'stages' in config['rsc']:
        stages = config['rsc']['stages']
    else:
        raise ValueError("config['rsc']['stages']가 필요합니다. stages 리스트를 사용하세요.")
    
    # RSC 실행
    print("\n" + "="*80)
    print("RSC Inference 실행")
    print("="*80)
    print(f"질문: {question}")
    print(f"데이터셋: {dataset_name}")
    print(f"Subject: {subject}")
    print(f"Stage 수: {len(stages)}")
    for i, stage in enumerate(stages, 1):
        if i == 1:
            print(f"  Stage {i}: n={stage['n']}")
        else:
            mode = stage.get('mode', 'collective')
            print(f"  Stage {i}: n={stage['n']}, mode={mode}")
    print("="*80 + "\n")
    
    result = rsc_engine.run(
        question=question,
        stages=stages,
        temperature=config['rsc']['temperature'],
        max_tokens=config['rsc'].get('max_tokens', 10240),
        top_p=config['rsc'].get('top_p', 0.95),
        prompt_template=prompt_template,
        fewshot_context=fewshot_context,
        dataset_name=dataset_name,
        subject=subject,
        options=options
    )
    
    # 결과 출력
    print("\n" + "="*80)
    print("RSC 결과")
    print("="*80)
    print(f"[최종 답변]: {result['final_answer']}")
    print(f"[최종 답변 모드]: {result['final_answer_mode']}")
    print(f"[총 Stage 수]: {len(result['stages'])}")
    
    for stage_result in result['stages']:
        stage_num = stage_result['stage_number']
        print(f"\n[Stage {stage_num} 답변]: {stage_result['answer']}")
        print(f"[Stage {stage_num} Reasoning Paths]: {len(stage_result['paths'])}개")
        print(f"[Stage {stage_num} 답변 분포]:")
        for answer, count in stage_result['distribution'].items():
            print(f"  {answer}: {count}회")
    
    # 상세 결과 출력 옵션
    print("\n" + "="*80)
    print("상세 Reasoning Paths (각 Stage별 처음 2개만 표시)")
    print("="*80)
    
    for stage_result in result['stages']:
        stage_num = stage_result['stage_number']
        print(f"\n[Stage {stage_num} Paths (처음 2개)]:")
        for i, path in enumerate(stage_result['paths'][:2], 1):
            print(f"\n--- Stage {stage_num} Path {i} ---")
            print(path[:500] + "..." if len(path) > 500 else path)
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()

