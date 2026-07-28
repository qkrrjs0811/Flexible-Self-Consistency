"""
Evaluation 실행 스크립트

이 모듈은 전체 데이터셋을 대상으로 RSC 알고리즘을 평가합니다.
여러 최종 답변 선택 방식에 대해 평가하고 결과를 비교합니다.
기존 src/SC_generator_ours.py의 로직을 활용합니다.
"""

import json
import os
import sys
import yaml
import re
from typing import Dict, Any, List
from datetime import datetime
from tqdm import tqdm

# rsc 모듈 import (src 경로 추가 전에 먼저 import)
from rsc.rsc_core import RSCEngine
from rsc.generator import ReasoningPathGenerator
from rsc.aggregator import SCAggregator
from rsc.fewshot_utils import build_fewshot_context

# evaluation 모듈 import (src 경로 추가 전에 먼저 import)
from evaluation.datasets import DatasetLoader
from evaluation.metrics import calculate_metrics, evaluate_stage_from_json

# normalize_answer 함수는 evaluation.metrics에서 가져오기
from evaluation.metrics import normalize_answer_eval as normalize_answer

# 기존 코드 경로 추가 (필요한 경우에만 사용, evaluation 패키지와 충돌 방지)
# src 경로는 나중에 필요할 때만 추가하도록 변경
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))
def run_evaluation(config_path: str = 'experiments/config.yaml'):
    """
    설정 파일을 기반으로 RSC 평가를 실행합니다.
    - `experiments/results_cache` 디렉토리는 더 이상 사용하지 않습니다.
    - 모든 저장/재시작/스킵 로직은 `experiments/results` 기준으로 동작합니다.
    
    Args:
        config_path: 설정 파일 경로
    """
    # 설정 파일 로드
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 기본 설정
    dataset_name = config['dataset']['name']
    if 'stages' not in config['rsc']:
        raise ValueError("config['rsc']['stages']가 필요합니다. stages 리스트를 사용하세요.")
    stages = config['rsc']['stages']

    # Stage 1만 사용
    if len(stages) != 1:
        raise ValueError(
            f"현재 run_evaluation은 Stage 1만 지원합니다 (len(stages)=1 필요, 현재={len(stages)})."
        )

    stage1_config = stages[0]
    stage1_n = stage1_config['n']
    final_answer_mode = config['rsc'].get('final_answer_mode', 'stage_k_only')

    model_name = config['model']['name']
    is_qwen3 = _is_qwen3_model(model_name)
    is_nemotron = _is_nemotron_model(model_name)
    is_qwen3_or_nemotron = is_qwen3 or is_nemotron
    enable_thinking = config['model'].get('enable_thinking', False)
    mode_subdir = "thinking" if (is_qwen3_or_nemotron and enable_thinking) else (
        "non-thinking" if is_qwen3_or_nemotron else None
    )

    # 결과 폴더 구성 (results만 사용)
    results_dir = config.get('results_dir', 'experiments/results')
    base_dir = os.path.join(results_dir, model_name)
    if is_qwen3_or_nemotron:
        base_dir = os.path.join(base_dir, mode_subdir)

    # run_all_experiments.sh에서 넣어주는 slug 사용 (없으면 simple fallback)
    stages_config_slug = config.get('stages_config_slug')
    if not stages_config_slug:
        stages_config_slug = f"stages-{stage1_n}"

    config_result_dir = os.path.join(base_dir, stages_config_slug)
    stage1_result_dir = os.path.join(config_result_dir, 'Stage1')
    stage1_result_path = os.path.join(stage1_result_dir, f'{dataset_name}.json')
    eval_result_path = os.path.join(config_result_dir, 'eval_result.txt')

    # 데이터셋 로드
    dataset_loader = DatasetLoader(
        dataset_name=dataset_name,
        dataset_path=config['dataset'].get('path'),
        eval_dir=config.get('eval_dir', '../eval')
    )
    dataset = dataset_loader.load(max_samples=config['dataset'].get('max_samples'))

    def _eval_result_has_dataset_section(path: str, ds_name: str) -> bool:
        """
        eval_result.txt에 특정 데이터셋 섹션(## Dataset: {name})이 이미 기록돼 있는지 확인.
        파일이 존재만 하고 내용이 일부만 있을 때(예: GSM8K만 기록됨) 잘못 스킵되는 것을 방지한다.
        """
        if not os.path.exists(path):
            return False
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception:
            return False
        # 섹션 헤더는 _print_and_save_eval_results가 "## Dataset: {dataset_name}" 형식으로 출력한다.
        pattern = rf"^## Dataset:\s*{re.escape(ds_name)}\s*$"
        return re.search(pattern, content, flags=re.MULTILINE) is not None

    # 이미 완전한 결과가 있는지 확인 (Stage1 JSON 기준)
    existing_stage_data = None
    if os.path.exists(stage1_result_path):
        try:
            with open(stage1_result_path, 'r', encoding='utf-8') as f:
                existing_stage_data = json.load(f)
        except Exception as e:
            print(f"[경고] 기존 Stage1 결과 파일을 읽는 데 실패했습니다: {stage1_result_path}, 오류: {e}")

    if existing_stage_data is not None:
        existing_sample_count = len(existing_stage_data)
    else:
        existing_sample_count = 0

    dataset_sample_count = len(dataset)

    # JSON 생성은 스킵하되, 평가는 항상 수행하도록 수정
    # (JSON 파일의 토큰 수 등이 수정되었을 수 있으므로 평가는 재수행 필요)
    json_generation_skip = (
        existing_sample_count == dataset_sample_count and
        os.path.exists(stage1_result_path)
    )
    
    # eval_result_all.txt 파일 경로 (평가 스킵 여부에 사용)
    eval_result_all_path = os.path.join(config_result_dir, 'eval_result_all.txt')
    
    # json_generation_skip가 True여도 _eval_result_has_dataset_section으로 데이터셋별 스킵 여부 확인
    # 단, eval_result_all.txt 파일이 없으면 평가를 스킵하지 않고 진행
    #
    # SINGLE_RUN 등에서 완전히 새로운 실험을 강제로 재평가하고 싶을 수 있어,
    # RSC_FORCE_EVAL=1이면 스킵 로직을 무시한다.
    force_eval = os.getenv("RSC_FORCE_EVAL", "").strip() == "1"
    if not force_eval:
        if _eval_result_has_dataset_section(eval_result_path, dataset_name) and os.path.exists(eval_result_all_path):
            # eval_result_all.txt 내부에도 해당 데이터셋의 섹션이 존재하는지 확인
            if _eval_result_has_dataset_section(eval_result_all_path, dataset_name):
                print("\n" + "=" * 80)
                print(f"[평가 스킵] {eval_result_path} 및 {eval_result_all_path}에 이미 {dataset_name} 결과가 있습니다.")
                print("=" * 80 + "\n")
                return
        
    if json_generation_skip:
        print("\n" + "=" * 80)
        print(f"[JSON 생성 스킵] Stage1 JSON이 이미 존재합니다.")
        print(f"  구성: {stages_config_slug}")
        print(f"  결과 폴더: {config_result_dir}")
        print(f"  Stage1 JSON 파일이 존재하여 JSON 생성은 스킵하고 평가만 수행합니다.")
        print("=" * 80 + "\n")

    # 데이터셋 이름과 subject 매핑 (기존 main_exp.sh 로직)
    DATASET_SUBJECT_MAP = {
        'AIME_2024': 'math',
        'AIME_2025': 'math',
        'MATH500_test': 'math',
        'MATH_train': 'math',
        'MATH2500': 'math',
        'AMC23': 'math',
        'GSM8K': 'math',
        'DEEPMATH_2600': 'math',
        'GPQA-Diamond': 'science',
        'MMLU-Pro': 'general',
        'MMLU-Pro_full': 'general',
    }
    subject = config.get('subject') or DATASET_SUBJECT_MAP.get(dataset_name, 'math')

    aggregator = SCAggregator(dataset_name=dataset_name)
    batch_size = config.get('evaluation', {}).get('batch_size', 1)
    esc_window_size = config.get('evaluation', {}).get('esc_window_size', 5)
    ac_beta = config.get('evaluation', {}).get('ac_beta', 0.95)
    num_trials = config.get('evaluation', {}).get('num_trials', 10)
    bootstrap_seed = config.get('evaluation', {}).get('bootstrap_seed', 42)

    print("\n" + "=" * 80)
    print("=== RSC 평가 시작 (Stage 1 only) ===")
    print("=" * 80)
    print(f"데이터셋: {dataset_name}")
    print(f"총 샘플 수: {len(dataset)}")
    print(f"Stage 1: n={stage1_n}")
    print(f"최종 답변 모드: {final_answer_mode}")
    print("=" * 80)

    # 모든 샘플의 질문/옵션
    all_questions = [sample.get('question', '') for sample in dataset]
    if dataset_name in ['MMLU-Pro', 'MMLU-Pro_full', 'GPQA-Diamond']:
        all_options = [sample.get('options') for sample in dataset]
    else:
        all_options = [None] * len(dataset)

    # 재시작: 이미 저장된 Stage1 JSON 기준으로 처리된 샘플 찾기
    processed_indices = set()
    existing_data_by_id = {}
    if existing_stage_data:
        for row in existing_stage_data:
            sid = row.get('id')
            if sid is not None:
                existing_data_by_id[sid] = row

        for idx, sample in enumerate(dataset):
            sid = sample.get('id')
            if sid in existing_data_by_id:
                processed_indices.add(idx)

        if processed_indices:
            print(f"[재시작 감지] Stage 1: {len(processed_indices)}개 샘플이 이미 처리됨. 스킵합니다.")

    remaining_indices = [i for i in range(len(dataset)) if i not in processed_indices]

    sample_stage_results = {}  # {sample_idx: {1: {...}}}

    # 이미 처리된 샘플을 메모리로 가져오기
    if existing_stage_data:
        for sample_idx, sample in enumerate(dataset):
            sid = sample.get('id')
            if sid in existing_data_by_id:
                row_dict = existing_data_by_id[sid]
                # Reasoning Paths 형식에서 paths/tokens 추출
                if 'Reasoning Paths' in row_dict:
                    reasoning_paths = row_dict['Reasoning Paths']
                    paths = []
                    output_tokens_list = []
                    for path_entry in reasoning_paths:
                        cot_key = None
                        for key in path_entry.keys():
                            if key.startswith('CoT_'):
                                cot_key = key
                                break
                        if cot_key:
                            paths.append(path_entry[cot_key] or '')
                            token_key = cot_key.replace('CoT_', 'Tokens_')
                            output_tokens_list.append(path_entry.get(token_key))
                else:
                    cot_count = 0
                    while f'CoT_{cot_count}' in row_dict:
                        cot_count += 1
                    paths = [row_dict.get(f'CoT_{i}', '') for i in range(cot_count)]
                    output_tokens_list = [row_dict.get(f'Tokens_{i}') for i in range(cot_count)]

                answer = aggregator.aggregate(paths) if paths else ''
                sample_stage_results.setdefault(sample_idx, {})[1] = {
                    'paths': paths,
                    'answer': answer,
                    'token_info': {
                        'input_tokens': row_dict.get('input_tokens'),
                        'output_tokens': output_tokens_list,
                    },
                    'distribution': aggregator.get_answer_distribution(paths),
                }

    generator = None
    prompt_template = None
    fewshot_context = None

    # 처리되지 않은 샘플 생성 (남은 샘플이 있을 때만 모델 로딩)
    if remaining_indices:
        # Few-shot 컨텍스트 구성 (생성 시에만 필요)
        fewshot_context = build_fewshot_context(
            dataset_name,
            disabled=config.get('disable_fewshot', False)
        )

        # 프롬프트 템플릿 로드 (생성 시에만 필요)
        prompt_template_path = config.get('prompt_template_path', None)
        if prompt_template_path and os.path.exists(prompt_template_path):
            with open(prompt_template_path, 'r', encoding='utf-8') as f:
                prompt_template = json.load(f)
        else:
            prompt_template = None

        refinement_template_path = config.get('refinement_template_path', None)

        print(f"[모델 로딩] {model_name} (Stage 1용)")
        generator = ReasoningPathGenerator(
            model_name=config['model']['name'],
            device=config['model']['device'],
            gpu_memory_utilization=config['model'].get('gpu_memory_utilization', 0.85),
            max_model_len=config['model'].get('max_model_len', 16384),
            enable_thinking=config['model'].get('enable_thinking', False),
            prompt_template_path=prompt_template_path,
            refinement_template_path=refinement_template_path
        )
        print("[모델 로딩 완료]")

        # - "누적 처리 완료 수 / 총 샘플 수"와 "다음 처리 범위(전체 인덱스), 남은 샘플 수"를 함께 보여준다.
        already_processed = len(processed_indices)
        total_samples = len(dataset)
        remaining_count = len(remaining_indices)
        print(f"[처리 시작] Stage 1: {already_processed}개 샘플 처리 완료 (총 {total_samples}개 중)")
        if remaining_indices:
            next_start = remaining_indices[0]
            next_end = remaining_indices[min(batch_size - 1, len(remaining_indices) - 1)]
            print(f"(전체 dataset 인덱스 {next_start}~{next_end}, 남은 샘플 개수: {remaining_count}개)")

        # results 기준으로만 저장
        os.makedirs(stage1_result_dir, exist_ok=True)

        remaining_questions = [all_questions[i] for i in remaining_indices]
        remaining_options = [all_options[i] for i in remaining_indices] if all_options else None
        remaining_dataset = [dataset[i] for i in remaining_indices]

        # 배치 완료 콜백: results/Stage1/{dataset}.json에 중간 저장
        saved_new_total = 0  # 이번 실행에서 새로 저장된 샘플 누적

        def batch_callback(batch_start_idx, batch_paths, batch_token_infos):
            actual_start_idx = remaining_indices[batch_start_idx] if batch_start_idx < len(remaining_indices) else batch_start_idx
            actual_end_idx = remaining_indices[min(
                batch_start_idx + len(batch_paths) - 1,
                len(remaining_indices) - 1
            )] if batch_start_idx + len(batch_paths) <= len(remaining_indices) else actual_start_idx + len(batch_paths) - 1

            nonlocal saved_new_total
            saved_count = _save_stage_json_incremental(
                list(zip(batch_paths, batch_token_infos)),
                stage1_result_path,
                remaining_dataset,
                batch_start_idx,
                dataset_name,
                aggregator,
                actual_start_idx=actual_start_idx,
                actual_end_idx=actual_end_idx,
            )
            saved_new_total += saved_count

            # 진행 로그: 누적 처리 완료 / 남은 샘플을 명확히 출력
            processed_total = already_processed + saved_new_total
            remaining_total = max(total_samples - processed_total, 0)
            print(f"[처리 진행] Stage 1: {processed_total}개 샘플 처리 완료 (총 {total_samples}개 중)")
            print(f"(전체 dataset 인덱스 {actual_start_idx}~{actual_end_idx}, 남은 샘플 개수: {remaining_total}개)")

        print_first_prompt = len(remaining_questions) > 0

        stage1_paths_list_partial, stage1_token_infos_partial = generator.generate_stage1_paths_batch(
            questions=remaining_questions,
            n_paths=stage1_n,
            temperature=config['rsc']['temperature'],
            max_tokens=config['rsc'].get('max_tokens', 10240),
            top_p=config['rsc'].get('top_p', 0.95),
            prompt_template=prompt_template,
            fewshot_context=fewshot_context,
            dataset_name=dataset_name,
            subject=subject,
            options_list=remaining_options,
            batch_size=batch_size,
            batch_callback=batch_callback,
            print_first_prompt=print_first_prompt,
        )

        # 새로 생성한 샘플을 sample_stage_results에 반영
        for local_idx, global_idx in enumerate(remaining_indices):
            paths = stage1_paths_list_partial[local_idx]
            token_info = stage1_token_infos_partial[local_idx]
            answer = aggregator.aggregate(paths) if paths else ''

            sample_stage_results.setdefault(global_idx, {})[1] = {
                'paths': paths,
                'answer': answer,
                'token_info': token_info,
                'distribution': aggregator.get_answer_distribution(paths),
            }

        print(f"[추가 생성 완료] Stage 1: {len(remaining_indices)}개 샘플 생성 완료")

        # 전체 Stage1 데이터를 stage1_result_path에서 다시 로드 (최종 저장본 기준)
        if os.path.exists(stage1_result_path):
            with open(stage1_result_path, 'r', encoding='utf-8') as f:
                stage1_data_list = json.load(f)
        else:
            # fallback: 메모리에서 구성
            stage1_data_list = []
            for sample_idx, sample in enumerate(dataset):
                if sample_idx in sample_stage_results and 1 in sample_stage_results[sample_idx]:
                    res = sample_stage_results[sample_idx][1]
                    stage_row = _prepare_stage_data(sample, res['paths'], res['answer'], dataset_name, res['token_info'])
                    stage1_data_list.append(stage_row)
            _save_stage_json(stage1_data_list, stage1_result_path, append=False)
    else:
        # 남은 샘플이 없으면 모델 로딩 없이 Stage1 결과로만 처리
        print(f"[완료] Stage 1: 모든 샘플이 이미 처리되었습니다. results에서 로드합니다. (모델 로딩 스킵)")
        if not os.path.exists(stage1_result_path):
            raise FileNotFoundError(
                f"Stage1 결과 파일이 존재하지 않습니다. remaining_indices=0인데 파일이 없을 수 없습니다: {stage1_result_path}"
            )
        with open(stage1_result_path, 'r', encoding='utf-8') as f:
            stage1_data_list = json.load(f)

    # 결과 리스트/메트릭 계산
    results = []
    for sample_idx, sample in enumerate(dataset):
        if sample_idx not in sample_stage_results or 1 not in sample_stage_results[sample_idx]:
            raise ValueError(f"샘플 {sample_idx}에 대한 Stage 1 결과가 없습니다.")

        stage_res = sample_stage_results[sample_idx][1]
        stages_metadata = [{
            'stage_number': 1,
            'paths': stage_res['paths'],
            'answer': stage_res['answer'],
            'token_info': stage_res['token_info'],
            'distribution': stage_res['distribution'],
            'mode': None,
        }]

        # Stage 1 only에서는 최종 답변 = Stage1 답변
        final_answer = stage_res['answer']

        results.append({
            'id': sample.get('id', ''),
            'question': sample.get('question', ''),
            'ground_truth': str(sample.get('answer', '')),
            'prediction': final_answer,
            'metadata': {
                'dataset_name': dataset_name,
                'num_stages': 1,
                'final_answer_mode': final_answer_mode,
                'stages': stages_metadata,
                'from_cache': False,
            },
        })

    metrics = calculate_metrics(results)

    # Stage1 상세 메트릭 (STAGES_CONFIG 기반 동적 평가 규칙 적용)
    stage1_detailed_metrics = None
    if os.path.exists(stage1_result_path):
        # 사용자가 원하는 규칙:
        # - pool_size = STAGES_CONFIG (stages_config_slug에서 추출)
        # - CoT: pool에서 1개 랜덤 샘플링 (metrics.py에서 처리)
        # - SC: pool에서 40개 랜덤 샘플링 후 majority voting
        # - ESC/AC: pool에서 40개 랜덤 샘플링 후 알고리즘 적용
        sc_n = config.get('evaluation', {}).get('sc_n', 40)
        subset_n = config.get('evaluation', {}).get('subset_n', 40)
        stage1_detailed_metrics = evaluate_stage_from_json(
            stage1_result_path, "Stage1",
            esc_window_size=esc_window_size,
            ac_beta=ac_beta,
            num_trials=num_trials,
            seed=bootstrap_seed,
            stages_config_slug=stages_config_slug,
            sc_n=sc_n,
            subset_n=subset_n,
        )

        if dataset_name in ("MMLU-Pro", "MMLU-Pro_full"):
            try:
                from evaluation.mmlu_pro_category_aggregate import merge_mmlu_pro_category_section
                from evaluation.metrics import evaluate_mmlu_pro_by_category_json_data

                with open(stage1_result_path, "r", encoding="utf-8") as _f:
                    _mmlu_data = json.load(_f)
                _cat_rsc = evaluate_mmlu_pro_by_category_json_data(
                    _mmlu_data,
                    stage_name="Stage1",
                    esc_window_size=esc_window_size,
                    ac_beta=ac_beta,
                    num_trials=num_trials,
                    seed=bootstrap_seed,
                    stages_config_slug=stages_config_slug,
                    sc_n=sc_n,
                    subset_n=subset_n,
                )
                _saved = merge_mmlu_pro_category_section(
                    config_result_dir,
                    "rsc",
                    _cat_rsc,
                    dataset_name=dataset_name,
                    extra_meta={"stages_config_slug": stages_config_slug, "source": "run_eval"},
                )
                print(f"[{dataset_name}] 카테고리별 RSC 집계 저장: {_saved}")
            except Exception as _e:
                print(f"[경고] {dataset_name} 카테고리별 RSC 집계 저장 실패: {_e}")

    all_results = {
        'rsc': {
            'description': f"Stage 1 RSC (n={stage1_n})",
            'results': results,
            'metrics': metrics,
            'stage1_detailed': stage1_detailed_metrics,
            'stages': results[0]['metadata'].get('stages') if results else None,
        }
    }

    _print_and_save_eval_results(all_results, config, dataset_name, stage1_result_path)


def save_stage_result(stage_data: List[Dict[str, Any]], stage_number: int,
                     model_name: str, dataset_name: str, enable_thinking: bool = False,
                     results_dir: str = 'experiments/results') -> str:
    """
    Stage 결과를 저장
    
    Args:
        stage_data: Stage 결과 데이터 리스트
        stage_number: Stage 번호
        model_name: 모델 이름
        dataset_name: 데이터셋 이름
        enable_thinking: Qwen3 thinking 모드 여부
        results_dir: 결과 저장 디렉토리
        
    Returns:
        str: 저장된 JSON 파일 경로
    """
    is_qwen3 = _is_qwen3_model(model_name)
    is_nemotron = _is_nemotron_model(model_name)
    is_qwen3_or_nemotron = is_qwen3 or is_nemotron
    base_dir = os.path.join(results_dir, model_name)
    if is_qwen3_or_nemotron:
        mode_subdir = "thinking" if enable_thinking else "non-thinking"
        base_dir = os.path.join(base_dir, mode_subdir)
    
    stage_dir = os.path.join(base_dir, f'Stage{stage_number}')
    os.makedirs(stage_dir, exist_ok=True)
    
    stage_file = os.path.join(stage_dir, f'{dataset_name}.json')
    _save_stage_json(stage_data, stage_file)
    print(f"[Stage {stage_number}] 결과 저장: {stage_file}")
    
    return stage_file


def _print_and_save_eval_results(all_results: Dict[str, Any], config: Dict[str, Any], 
                                  dataset_name: str, stage1_json_path: str = None):
    """
    eval_result.txt 형식으로 결과 출력 및 저장
    
    Args:
        all_results: 평가 결과
        config: 설정 딕셔너리
        dataset_name: 데이터셋 이름
        stage1_json_path: Stage 1 JSON 파일 경로 (없으면 저장 경로 구성)
    """
    # RSC 결과 가져오기
    rsc_result = all_results.get('rsc', {})
    stage1_detailed = rsc_result.get('stage1_detailed', None)
    metrics = rsc_result.get('metrics', {})
    
    # eval_result.txt 형식으로 출력 및 저장
    log_lines = []
    
    def log(message=""):
        print(message)
        # 줄바꿈을 포함하여 추가 (빈 메시지도 줄바꿈 포함)
        log_lines.append(message + "\n" if message else "\n")
    
    # JSON 파일 경로 정보
    model_name = config['model']['name']
    is_qwen3 = _is_qwen3_model(model_name)
    is_nemotron = _is_nemotron_model(model_name)
    is_qwen3_or_nemotron = is_qwen3 or is_nemotron
    results_dir = config.get('results_dir', 'experiments/results')
    base_dir = os.path.join(results_dir, model_name)
    if is_qwen3_or_nemotron:
        mode_subdir = "thinking" if config['model'].get('enable_thinking', False) else "non-thinking"
        base_dir = os.path.join(base_dir, mode_subdir)
    model_base_dir = base_dir
    
    if stage1_json_path:
        # 실제로 평가에 사용된 JSON 경로를 그대로 로그에 남긴다.
        # (상대/절대 경로 혼용 시 startswith 비교가 깨져 엉뚱한 경로가 찍히는 문제 방지)
        json_path_for_log = os.path.abspath(stage1_json_path)
    else:
        json_file = f'{dataset_name}.json'
        json_path_for_log = os.path.join(model_base_dir, 'Stage1', json_file)
    
    log("")  # 빈 줄 추가
    log("=" * 70)
    log(f"## Dataset: {dataset_name}")
    log("=" * 70)
    log(f"Loading JSON file: {json_path_for_log}")
    log(f"Loaded {metrics.get('total_samples', 0)} samples")
    log("")  # 빈 줄 추가
    
    # Stage 1 메트릭 출력 (CoT, SC, AC, ESC)
    if stage1_detailed:
        if stage1_detailed.get('stage1_latency_sample_count', 0) > 0:
            log(f"Stage1 Generation Latency: {stage1_detailed['stage1_avg_latency_sec']:.4f}s per sample "
                f"(avg over {stage1_detailed['stage1_latency_sample_count']} samples)")
            log("")
        if 'cot_accuracy' in stage1_detailed:
            log(f"CoT Accuracy: {stage1_detailed['cot_accuracy']:.4f} "
                f"(avg_input_tokens={stage1_detailed.get('cot_avg_input_tokens', 0):.2f}, "
                f"avg_output_tokens={stage1_detailed.get('cot_avg_output_tokens', 0):.2f})")
        
        if 'sc_accuracy' in stage1_detailed:
            log(f"SC Accuracy: {stage1_detailed['sc_accuracy']:.4f} "
                f"(avg_input_tokens={stage1_detailed.get('sc_avg_input_tokens', 0):.2f}, "
                f"avg_output_tokens={stage1_detailed.get('sc_avg_output_tokens', 0):.2f})")
        
        ac_beta = config.get('evaluation', {}).get('ac_beta', 0.95)
        if 'ac_accuracy' in stage1_detailed:
            log(f"AC Accuracy: {stage1_detailed['ac_accuracy']:.4f} "
                f"(beta={ac_beta}, avg_chains={stage1_detailed.get('ac_avg_chains', 0):.2f}, "
                f"avg_input_tokens={stage1_detailed.get('ac_avg_input_tokens', 0):.2f}, "
                f"avg_output_tokens={stage1_detailed.get('ac_avg_output_tokens', 0):.2f})")

        esc_window_size = config.get('evaluation', {}).get('esc_window_size', 5)
        if 'esc_esc_accuracy' in stage1_detailed:
            log(f"ESC Accuracy: {stage1_detailed['esc_esc_accuracy']:.4f} "
                f"(window={esc_window_size}, avg_chains={stage1_detailed.get('esc_esc_avg_chains', 0):.2f}, "
                f"avg_input_tokens={stage1_detailed.get('esc_esc_avg_input_tokens', 0):.2f}, "
                f"avg_output_tokens={stage1_detailed.get('esc_esc_avg_output_tokens', 0):.2f})")
    
    log("")  # blank line
    
    # 단일 실험 모드: eval_result.txt는 수정하지 않고, 결과는 eval_single_{seed}.txt에만 기록
    single_output_path = os.environ.get('RSC_EVAL_SINGLE_OUTPUT_PATH', '').strip()
    is_single_run = bool(single_output_path)
    
    # 구성별 결과 디렉토리 사용 (stages_config_slug 포함)
    stages_config_slug = config.get('stages_config_slug')
    if stages_config_slug:
        eval_result_dir = os.path.join(model_base_dir, stages_config_slug)
        if not is_single_run:
            os.makedirs(eval_result_dir, exist_ok=True)
        eval_result_path = os.path.join(eval_result_dir, 'eval_result.txt')
    else:
        # 이전 방식 호환
        eval_result_path = os.path.join(model_base_dir, 'eval_result.txt')
        if not is_single_run:
            os.makedirs(model_base_dir, exist_ok=True)
    
    if is_single_run:
        # eval_single_*.txt에서 기존 내용을 읽고, 동일 데이터셋 섹션만 제거 후 새 섹션 추가 (여러 데이터셋 누적)
        if os.path.exists(single_output_path):
            with open(single_output_path, 'r', encoding='utf-8') as f:
                existing_lines = f.readlines()
            dataset_header = f"## Dataset: {dataset_name}"
            start_idx = -1
            end_idx = -1
            for i, line in enumerate(existing_lines):
                if line.strip() == dataset_header:
                    start_idx = i
                    break
            if start_idx >= 0:
                for j in range(start_idx + 1, len(existing_lines)):
                    if existing_lines[j].startswith("## Dataset:"):
                        end_idx = j
                        break
                if end_idx < 0:
                    end_idx = len(existing_lines)
                del existing_lines[start_idx:end_idx]
            existing_content = "".join(existing_lines)
        else:
            existing_content = ""
    else:
        # 기존 파일이 있으면 읽어서, 동일 데이터셋 섹션은 제거한 뒤 새 섹션을 추가
        if os.path.exists(eval_result_path):
            with open(eval_result_path, 'r', encoding='utf-8') as f:
                existing_lines = f.readlines()

            dataset_header = f"## Dataset: {dataset_name}"
            start_idx = -1
            end_idx = -1

            for i, line in enumerate(existing_lines):
                if line.strip() == dataset_header:
                    start_idx = i
                    break

            if start_idx >= 0:
                # 다음 데이터셋 헤더까지를 이 데이터셋 섹션으로 보고 제거
                for j in range(start_idx + 1, len(existing_lines)):
                    if existing_lines[j].startswith("## Dataset:"):
                        end_idx = j
                        break
                if end_idx < 0:
                    end_idx = len(existing_lines)
                del existing_lines[start_idx:end_idx]

            existing_content = "".join(existing_lines)
        else:
            existing_content = ""
        
    def save_to_eval_all(eval_result_all_path: str, dataset_name: str, method_name: str, trials: List[float], fmt: str = ".4f"):
        if not trials:
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

            method_prefix = f"{method_name} Trials:"
            line_indices = []
            for i in range(section_start_idx, section_end_idx):
                if all_lines[i].startswith(method_prefix):
                    line_indices.append(i)

            for idx in sorted(line_indices, reverse=True):
                all_lines.pop(idx)
                section_end_idx -= 1

            insert_idx = section_end_idx
            while insert_idx > section_start_idx and all_lines[insert_idx - 1].strip() == "":
                insert_idx -= 1

            trials_str = f"{method_name} Trials: " + "[" + ", ".join(f"{x:{fmt}}" for x in trials) + "]\n"
            all_lines.insert(insert_idx, trials_str)
        else:
            new_section = f"======================================================================\n"
            new_section += f"## Dataset: {dataset_name}\n"
            new_section += f"======================================================================\n"
            new_section += f"{method_name} Trials: " + "[" + ", ".join(f"{x:{fmt}}" for x in trials) + "]\n\n"
            all_lines.extend(new_section.splitlines(keepends=True))

        with open(eval_result_all_path, 'w', encoding='utf-8') as f:
            f.writelines(all_lines)

    if stages_config_slug:
        eval_result_all_path = os.path.join(model_base_dir, stages_config_slug, 'eval_result_all.txt')
    else:
        eval_result_all_path = os.path.join(model_base_dir, 'eval_result_all.txt')

    # 단일 실험 모드에서는 eval_result_all.txt도 수정하지 않음
    if stage1_detailed and not is_single_run:
        if 'cot_trial_accuracies' in stage1_detailed:
            save_to_eval_all(eval_result_all_path, dataset_name, "CoT", stage1_detailed['cot_trial_accuracies'])
            if 'cot_trial_chains' in stage1_detailed:
                save_to_eval_all(eval_result_all_path, dataset_name, "CoT Chains", stage1_detailed['cot_trial_chains'], fmt=".2f")
            if 'cot_trial_input_tokens' in stage1_detailed:
                save_to_eval_all(eval_result_all_path, dataset_name, "CoT Input Tokens", stage1_detailed['cot_trial_input_tokens'], fmt=".2f")
            if 'cot_trial_output_tokens' in stage1_detailed:
                save_to_eval_all(eval_result_all_path, dataset_name, "CoT Output Tokens", stage1_detailed['cot_trial_output_tokens'], fmt=".2f")
        if 'sc_trial_accuracies' in stage1_detailed:
            save_to_eval_all(eval_result_all_path, dataset_name, "SC", stage1_detailed['sc_trial_accuracies'])
            if 'sc_trial_chains' in stage1_detailed:
                save_to_eval_all(eval_result_all_path, dataset_name, "SC Chains", stage1_detailed['sc_trial_chains'], fmt=".2f")
            if 'sc_trial_input_tokens' in stage1_detailed:
                save_to_eval_all(eval_result_all_path, dataset_name, "SC Input Tokens", stage1_detailed['sc_trial_input_tokens'], fmt=".2f")
            if 'sc_trial_output_tokens' in stage1_detailed:
                save_to_eval_all(eval_result_all_path, dataset_name, "SC Output Tokens", stage1_detailed['sc_trial_output_tokens'], fmt=".2f")
        if 'ac_trial_accuracies' in stage1_detailed:
            save_to_eval_all(eval_result_all_path, dataset_name, "AC", stage1_detailed['ac_trial_accuracies'])
            if 'ac_trial_chains' in stage1_detailed:
                save_to_eval_all(eval_result_all_path, dataset_name, "AC Chains", stage1_detailed['ac_trial_chains'], fmt=".2f")
            if 'ac_trial_input_tokens' in stage1_detailed:
                save_to_eval_all(eval_result_all_path, dataset_name, "AC Input Tokens", stage1_detailed['ac_trial_input_tokens'], fmt=".2f")
            if 'ac_trial_output_tokens' in stage1_detailed:
                save_to_eval_all(eval_result_all_path, dataset_name, "AC Output Tokens", stage1_detailed['ac_trial_output_tokens'], fmt=".2f")
        if 'esc_esc_trial_accuracies' in stage1_detailed:
            save_to_eval_all(eval_result_all_path, dataset_name, "ESC", stage1_detailed['esc_esc_trial_accuracies'])
            if 'esc_esc_trial_chains' in stage1_detailed:
                save_to_eval_all(eval_result_all_path, dataset_name, "ESC Chains", stage1_detailed['esc_esc_trial_chains'], fmt=".2f")
            if 'esc_esc_trial_input_tokens' in stage1_detailed:
                save_to_eval_all(eval_result_all_path, dataset_name, "ESC Input Tokens", stage1_detailed['esc_esc_trial_input_tokens'], fmt=".2f")
            if 'esc_esc_trial_output_tokens' in stage1_detailed:
                save_to_eval_all(eval_result_all_path, dataset_name, "ESC Output Tokens", stage1_detailed['esc_esc_trial_output_tokens'], fmt=".2f")
    
    # 새 결과 추가 (데이터셋별로 섹션 구분)
    # log_lines의 각 요소가 이미 줄바꿈을 포함하므로 join만 하면 됨
    new_section = "".join(log_lines)
    if not new_section.endswith("\n\n"):
        if new_section.endswith("\n"):
            new_section += "\n"
        else:
            new_section += "\n\n"
    
    if is_single_run:
        # 단일 실험: eval_result.txt는 건드리지 않고, eval_single_{seed}.txt에만 기록 (데이터셋별 섹션 누적)
        os.makedirs(os.path.dirname(single_output_path), exist_ok=True)
        if existing_content and not existing_content.endswith("\n"):
            existing_content += "\n"
        with open(single_output_path, 'w', encoding='utf-8') as f:
            f.write(existing_content)
            f.write(new_section)
        print(f"\n[eval_single] 저장 (eval_result.txt 미수정): {single_output_path}")
    else:
        # 기존 내용에 새 섹션 추가
        if existing_content and not existing_content.endswith("\n"):
            existing_content += "\n"
        with open(eval_result_path, 'w', encoding='utf-8') as f:
            f.write(existing_content)
            f.write(new_section)
        print(f"\n[eval_result.txt] 저장: {eval_result_path}")


def _is_qwen3_model(model_name: str) -> bool:
    """Check if the model is a Qwen3 series model"""
    model_name_lower = model_name.lower()
    return 'qwen3' in model_name_lower or 'qwen-3' in model_name_lower


def _is_nemotron_model(model_name: str) -> bool:
    """Check if the model is a Nemotron series model"""
    model_name_lower = model_name.lower()
    return 'nemotron' in model_name_lower or 'nvidia-nemotron' in model_name_lower


def _prepare_stage_data(sample: Dict[str, Any], paths: List[str], final_answer: str,
                       dataset_name: str, token_info: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Stage별 데이터를 기존 save_json 형식으로 준비
    
    Args:
        sample: 원본 샘플 데이터
        paths: Reasoning paths 리스트
        final_answer: 최종 답변
        dataset_name: 데이터셋 이름
        token_info: 토큰 정보 딕셔너리
            - Stage 1/Collective: {'input_tokens': int, 'output_tokens': List[int]}
            - Individual: {'input_tokens': List[int], 'output_tokens': List[int]}
        
    Returns:
        Dict: 기존 save_json 형식과 호환되는 딕셔너리
    """
    # 원본 데이터 복사
    row_dict = sample.copy()
    
    # 각 reasoning path를 CoT_i, Final Answer_i 형식으로 추가
    for i, path in enumerate(paths):
        row_dict[f'CoT_{i}'] = path
        
        # 각 path에서 답변 추출 (aggregator의 extract 함수 사용)
        from rsc.aggregator import SCAggregator
        aggregator = SCAggregator(dataset_name=dataset_name)
        extracted_answer = aggregator.extract_answer(path)
        # 정규화 적용 (metrics.py의 normalize_answer_eval과 동일한 방식)
        normalized_answer = normalize_answer(extracted_answer) if extracted_answer else ''
        row_dict[f'Final Answer_{i}'] = normalized_answer
        
        # 토큰 수 설정
        if token_info and 'output_tokens' in token_info and i < len(token_info['output_tokens']):
            row_dict[f'Tokens_{i}'] = token_info['output_tokens'][i]
        else:
            row_dict[f'Tokens_{i}'] = None
    
    # 입력 토큰 수 설정
    if token_info and 'input_tokens' in token_info:
        # Collective 모드: 단일 값
        if isinstance(token_info['input_tokens'], int):
            row_dict['input_tokens'] = token_info['input_tokens']
        # Individual 모드: 리스트 (각 chain마다 개별 저장)
        elif isinstance(token_info['input_tokens'], list) and len(token_info['input_tokens']) > 0:
            # Individual 모드: 각 프롬프트의 토큰 수를 input_tokens_{i}로 저장
            for i, input_tok in enumerate(token_info['input_tokens']):
                row_dict[f'input_tokens_{i}'] = input_tok
            # input_tokens는 모든 프롬프트의 총 토큰 수 (합계)
            row_dict['input_tokens'] = sum([t for t in token_info['input_tokens'] if t is not None])
        else:
            row_dict['input_tokens'] = None
    else:
        row_dict['input_tokens'] = None

    if token_info and token_info.get('latency_sec') is not None:
        row_dict['latency_sec'] = token_info['latency_sec']
    
    return row_dict


def _save_stage_json_incremental(batch_data_list: List[Dict[str, Any]], file_path: str, 
                                  dataset_samples: List[Dict[str, Any]], batch_start_idx: int,
                                  dataset_name: str, aggregator, actual_start_idx: int = None,
                                  actual_end_idx: int = None) -> int:
    """
    배치 단위로 Stage 데이터를 중간 저장 (append 모드)
    _save_stage_json과 동일한 형식으로 저장 (Reasoning Paths 형식)
    
    Args:
        batch_data_list: 배치 데이터 리스트 (paths, token_infos)
        file_path: 저장할 파일 경로
        dataset_samples: 원본 데이터셋 샘플 리스트
        batch_start_idx: 배치 시작 인덱스
        dataset_name: 데이터셋 이름
        aggregator: SCAggregator 인스턴스
        
    Returns:
        int: 저장된 샘플 수
    """
    # 기존 데이터 읽기
    if os.path.isfile(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
    else:
        existing_data = []
    
    # 이미 저장된 샘플 ID 추출 (중복 방지)
    saved_ids = {item.get('id') for item in existing_data if 'id' in item}
    
    # 배치 데이터를 저장 형식으로 변환
    new_items = []
    for i, (paths, token_info) in enumerate(batch_data_list):
        sample_idx = batch_start_idx + i
        if sample_idx >= len(dataset_samples):
            continue
        
        sample = dataset_samples[sample_idx]
        sample_id = sample.get('id')
        
        # 이미 저장된 샘플은 스킵
        if sample_id in saved_ids:
            continue
        
        answer = aggregator.aggregate(paths)
        # 정규화된 답변으로 저장 (aggregate는 이미 정규화된 majority를 반환하지만, 일관성을 위해 다시 정규화)
        normalized_answer = normalize_answer(answer) if answer else ''
        # _prepare_stage_data로 평면 형식 준비
        stage_row = _prepare_stage_data(sample, paths, normalized_answer, dataset_name, token_info)
        
        # _save_stage_json과 동일하게 Reasoning Paths 형식으로 변환
        row_dict = stage_row.copy()
        
        # CoT 인덱스 추출
        cot_indices = []
        for key in row_dict.keys():
            match = re.match(r'^CoT_(\d+)$', key)
            if match:
                cot_indices.append(int(match.group(1)))
        cot_indices.sort()
        
        # Reasoning Paths 그룹화
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
            # Individual 모드: input_tokens_{i}도 Reasoning Path에 포함
            input_token_key = f'input_tokens_{idx}'
            if input_token_key in row_dict:
                path_entry[input_token_key] = row_dict.get(input_token_key)
            reasoning_paths.append(path_entry)
            row_dict.pop(cot_key, None)
            row_dict.pop(final_key, None)
            row_dict.pop(token_key, None)
            row_dict.pop(input_token_key, None)
        
        row_dict['Reasoning Paths'] = reasoning_paths
        new_items.append(row_dict)
    
    if new_items:
        # 기존 데이터에 추가
        existing_data.extend(new_items)
        
        # 파일 저장
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)
        
        # 출력 메시지: remaining 기준 인덱스는 혼동을 줄 수 있어 숨기고, 전체 dataset 인덱스만 보여준다.
        if actual_start_idx is not None and actual_end_idx is not None:
            print(f"[중간 저장] {len(new_items)}개 샘플 저장 (전체 dataset 인덱스 {actual_start_idx}~{actual_end_idx})")
        else:
            print(f"[중간 저장] {len(new_items)}개 샘플 저장")
        return len(new_items)
    
    return 0


def _save_stage_json(data_list: List[Dict[str, Any]], file_path: str, append: bool = True):
    """
    Stage별 데이터를 기존 save_json 형식으로 저장
    
    Args:
        data_list: 저장할 데이터 리스트
        file_path: 저장할 파일 경로
        append: True면 기존 데이터에 추가, False면 덮어쓰기 (기본값: True)
    """
    # 기존 데이터 읽기 (append 모드이고 파일이 있으면)
    if append and os.path.isfile(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
    else:
        existing_data = []
    
    # 각 샘플을 기존 형식으로 변환
    for row_dict in data_list:
        # CoT 인덱스 추출
        cot_indices = []
        for key in row_dict.keys():
            match = re.match(r'^CoT_(\d+)$', key)
            if match:
                cot_indices.append(int(match.group(1)))
        cot_indices.sort()
        
        # Reasoning Paths 그룹화
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
            # Individual 모드: input_tokens_{i}도 Reasoning Path에 포함
            input_token_key = f'input_tokens_{idx}'
            if input_token_key in row_dict:
                path_entry[input_token_key] = row_dict.get(input_token_key)
            reasoning_paths.append(path_entry)
            row_dict.pop(cot_key, None)
            row_dict.pop(final_key, None)
            row_dict.pop(token_key, None)
            row_dict.pop(input_token_key, None)  # input_tokens_{i}도 제거
        
        row_dict['Reasoning Paths'] = reasoning_paths
        
        # 기존 데이터에 추가
        existing_data.append(row_dict)
    
    # 파일 저장
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=2)


def save_results(results: list, metrics: Dict[str, Any], config: Dict[str, Any]):
    """
    평가 결과를 파일로 저장 (메트릭 포함)
    
    Args:
        results: 평가 결과 리스트
        metrics: 계산된 메트릭
        config: 설정 딕셔너리
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = config.get('results_dir', 'experiments/results')
    os.makedirs(results_dir, exist_ok=True)
    
    # 결과 파일 저장
    results_file = os.path.join(results_dir, f'results_{timestamp}.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'config': config,
            'metrics': metrics,
            'results': results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n[평가 결과] 저장: {results_file}")


def _print_detailed_metrics(metrics: Dict[str, Any]):
    """
    상세 메트릭 출력 (src/evaluation.py 형식)
    
    Args:
        metrics: 평가 메트릭 딕셔너리
    """
    if metrics.get('stage1_latency_sample_count', 0) > 0:
        print(f"  Stage1 Generation Latency: {metrics['stage1_avg_latency_sec']:.4f}s per sample "
              f"(avg over {metrics['stage1_latency_sample_count']} samples)")

    if 'cot_accuracy' in metrics:
        print(f"  CoT Accuracy: {metrics['cot_accuracy']:.4f} "
              f"(avg_input_tokens={metrics.get('cot_avg_input_tokens', 0):.2f}, "
              f"avg_output_tokens={metrics.get('cot_avg_output_tokens', 0):.2f})")
    
    if 'sc_accuracy' in metrics:
        print(f"  SC Accuracy: {metrics['sc_accuracy']:.4f} "
              f"(avg_input_tokens={metrics.get('sc_avg_input_tokens', 0):.2f}, "
              f"avg_output_tokens={metrics.get('sc_avg_output_tokens', 0):.2f})")
    
    
    if 'ac_accuracy' in metrics:
        print(f"  AC Accuracy: {metrics['ac_accuracy']:.4f} "
              f"(beta={metrics.get('ac_avg_chains', 0):.2f}, "
              f"avg_input_tokens={metrics.get('ac_avg_input_tokens', 0):.2f}, "
              f"avg_output_tokens={metrics.get('ac_avg_output_tokens', 0):.2f})")

    if 'esc_esc_accuracy' in metrics:
        print(f"  ESC Accuracy: {metrics['esc_esc_accuracy']:.4f} "
              f"(window={metrics.get('esc_esc_avg_chains', 0):.2f}, "
              f"avg_input_tokens={metrics.get('esc_esc_avg_input_tokens', 0):.2f}, "
              f"avg_output_tokens={metrics.get('esc_esc_avg_output_tokens', 0):.2f})")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='RSC 평가 실행')
    parser.add_argument('--config', type=str, default='experiments/config.yaml',
                       help='설정 파일 경로')
    args = parser.parse_args()
    
    run_evaluation(args.config)
