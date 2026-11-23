"""
eval/{model}/sample_{k}/{dataset}.json (Stage1 형식, 반복 실행 결과)에 대해
CoT/SC/AC/ESC/DSC(+Ours)를 계산하고 eval_result.txt 형식으로 정리합니다.

- 기존 experiments/results 는 읽기만 합니다 (DSC 난이도 랭킹 dsc_eval/ 재사용, 문항 순서 검증).
- 계산은 기존 파이프라인 코드를 그대로 사용합니다.
    * CoT/SC/AC/ESC : evaluation.metrics.evaluate_stage_from_json + run_eval._print_and_save_eval_results
                      (run_all_experiments.sh 설정: pool 40, subset 40, esc_window 5, ac_beta 0.95)
    * DSC           : dsc/cmp_cal.py (run_all_cmp.sh 설정: judge_window 24, interplot 16)
    * Ours          : evaluate_ours_method.evaluate_ours_json (파일 내 number_of_paths 사용)
- 중간 산출물은 --work_dir (기본: eval/_work) 아래에 모델/샘플별 results_dir 로 구성됩니다.

출력:
    eval/{model}/sample_{k}/eval_result.txt   : 기존 eval_result.txt 와 동일 형식 (run 단위)
    eval/eval_result_summary.txt              : run 평균 ± 표준편차 (동일 형식) + 정확도 표

사용 예 (rsc_project/ 에서, conda activate rasc 후):
    python eval/aggregate_samples.py
    python eval/aggregate_samples.py --models gemma3_4b --samples sample_1 --datasets AMC23
    python eval/aggregate_samples.py --summary_only
"""

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(EVAL_DIR)  # rsc_project/
sys.path.insert(0, PROJECT_ROOT)

# eval/ 하위 폴더명 -> HF 모델 ID
MODEL_DIRS = OrderedDict([
    ("qwen2.5_3b", "Qwen/Qwen2.5-3B-Instruct"),
    ("qwen2.5_7b", "Qwen/Qwen2.5-7B-Instruct"),
    ("qwen2.5_14b", "Qwen/Qwen2.5-14B-Instruct"),
    ("gemma3_4b", "google/gemma-3-4b-it"),
])
SAMPLES = ["sample_1", "sample_2", "sample_3"]
DATASETS = ["AMC23", "AIME_2024", "AIME_2025"]
METHOD_ORDER = ["CoT", "SC", "AC", "ESC", "DSC"]

# run_all_experiments.sh / run_all_cmp.sh 와 동일한 설정
STAGES_CONFIG = "40"
STAGES_SLUG = f"stages-{STAGES_CONFIG}"
MAX_N = 40
ESC_WINDOW_SIZE = 5
AC_BETA = 0.95
JUDGE_WINDOW_SIZE = 24
INTERPLOT_SIZE = 16
NUM_TRIALS = 1


def _num_paths(data: List[dict]) -> int:
    counts = {len(item.get("Reasoning Paths", [])) for item in data}
    if len(counts) != 1:
        raise ValueError(f"샘플별 reasoning path 개수가 다릅니다: {sorted(counts)}")
    return counts.pop()


def _check_order_against_reference(data: List[dict], hf_model: str, dataset: str, src_path: str):
    """dsc_eval 랭킹은 기존 Stage1 문항 순서 기준이므로, 문항 id 순서가 같은지 확인."""
    ref_path = os.path.join(PROJECT_ROOT, "experiments", "results", hf_model, STAGES_SLUG, "Stage1", f"{dataset}.json")
    with open(ref_path, "r", encoding="utf-8") as f:
        ref = json.load(f)
    ids = [str(x.get("id")) for x in data]
    ref_ids = [str(x.get("id")) for x in ref]
    if ids != ref_ids:
        raise ValueError(f"문항 id 순서가 기준 Stage1과 다릅니다: {src_path} vs {ref_path}")


def run_one(model_dir: str, sample: str, datasets: List[str], work_dir: str,
            seed: int, with_ours: bool, ours_label: str) -> Optional[str]:
    """한 (모델, sample) 조합의 eval_result.txt 를 생성하고 eval/{model}/{sample}/ 에 복사."""
    from evaluation.metrics import evaluate_stage_from_json
    from evaluation.run_eval import _print_and_save_eval_results
    from evaluate_ours_method import evaluate_ours_json, save_eval_result

    hf_model = MODEL_DIRS[model_dir]
    sample_dir = os.path.join(EVAL_DIR, model_dir, sample)
    # 모델/샘플별 독립 results_dir (기존 experiments/results 와 분리)
    results_dir = os.path.join(work_dir, model_dir, sample)
    config_dir = os.path.join(results_dir, hf_model, STAGES_SLUG)
    if os.path.isdir(results_dir):
        shutil.rmtree(results_dir)  # 이전 실행 잔여물 제거 (work_dir 내부만)
    os.makedirs(os.path.join(config_dir, "Stage1"), exist_ok=True)

    # DSC 난이도 랭킹(문항 텍스트 기반, 샘플과 무관) 재사용: 읽기 전용 심볼릭 링크
    ref_dsc_eval = os.path.join(PROJECT_ROOT, "experiments", "results", hf_model, STAGES_SLUG, "dsc_eval")
    os.symlink(ref_dsc_eval, os.path.join(config_dir, "dsc_eval"))

    config = {
        "model": {"name": hf_model, "enable_thinking": False},
        "results_dir": results_dir,
        "stages_config_slug": STAGES_SLUG,
        "evaluation": {"esc_window_size": ESC_WINDOW_SIZE, "ac_beta": AC_BETA},
    }

    env = os.environ.copy()
    env.pop("RSC_EVAL_SINGLE_OUTPUT_PATH", None)

    for dataset in datasets:
        src_path = os.path.join(sample_dir, f"{dataset}.json")
        if not os.path.exists(src_path):
            print(f"[SKIP] {src_path} 없음")
            continue
        with open(src_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        n_paths = _num_paths(data)
        n = min(MAX_N, n_paths)  # run_all_cmp.sh: pool < 40 이면 N 축소
        _check_order_against_reference(data, hf_model, dataset, src_path)

        stage1_path = os.path.join(config_dir, "Stage1", f"{dataset}.json")
        shutil.copyfile(src_path, stage1_path)
        print(f"\n>>> {model_dir}/{sample}/{dataset} (paths={n_paths}, n={n})")

        # 1) CoT / SC / AC / ESC
        detailed = evaluate_stage_from_json(
            stage1_path, "Stage1",
            esc_window_size=ESC_WINDOW_SIZE,
            ac_beta=AC_BETA,
            num_trials=NUM_TRIALS,
            seed=seed,
            stages_config_slug=STAGES_SLUG,
            sc_n=MAX_N,
            subset_n=MAX_N,
        )
        all_results = {"rsc": {"metrics": {"total_samples": len(data)}, "stage1_detailed": detailed}}
        # Loading JSON file 줄에는 원본 sample 파일 경로를 남김
        _print_and_save_eval_results(all_results, config, dataset, src_path)

        # 2) DSC (eval_result.txt 의 AC/ESC 아래에 DSC 줄 추가)
        cmd = [
            sys.executable, "dsc/cmp_cal.py",
            "--model_name", hf_model,
            "--dataset_name", dataset,
            "--n", str(n),
            "--subset_n", str(n),
            "--judge_window_size", str(JUDGE_WINDOW_SIZE),
            "--interplot_size", str(INTERPLOT_SIZE),
            "--results_dir", results_dir,
            "--stages_config", STAGES_CONFIG,
            "--num_trials", str(NUM_TRIALS),
            "--bootstrap_seed", str(seed),
        ]
        log_path = os.path.join(results_dir, f"cmp_{dataset}.log")
        with open(log_path, "w", encoding="utf-8") as log_f:
            ret = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, stdout=log_f, stderr=subprocess.STDOUT)
        if ret.returncode != 0:
            print(f"[ERROR] DSC 실패 ({model_dir}/{sample}/{dataset}), 로그: {log_path}")
        else:
            print(f"[DSC] 완료 (로그: {log_path})")

        # 3) Ours (파일에 포함된 number_of_paths 사용)
        if with_ours:
            nop = {str(x["id"]): int(x["number_of_paths"]) for x in data if x.get("number_of_paths") is not None}
            ours = evaluate_ours_json(
                stage1_path, dataset, nop,
                pool_size=n_paths, subset_n=MAX_N, num_trials=1, max_n=MAX_N,
            )
            save_eval_result(os.path.join(config_dir, "eval_result.txt"), dataset, src_path, ours,
                             metric_name=ours_label)

    produced = os.path.join(config_dir, "eval_result.txt")
    if not os.path.exists(produced):
        print(f"[WARN] eval_result.txt 가 생성되지 않았습니다: {produced}")
        return None
    dest = os.path.join(sample_dir, "eval_result.txt")
    shutil.copyfile(produced, dest)
    print(f"\n[저장] {dest}")
    return dest


# ---------------------------------------------------------------------------
# 요약 (run 평균 ± 표준편차)
# ---------------------------------------------------------------------------
_METRIC_RE = re.compile(r"^(?P<name>[^:]+?): (?P<acc>-?[0-9.]+) \((?P<rest>.*)\)\s*$")
_PREPARE_RE = re.compile(r"^DSC Prepare \((?P<rest>.*)\)\s*$")
_KV_RE = re.compile(r"(\w+)[=:]\s*([^,]+)")


def _method_key(name: str) -> str:
    return name[:-len(" Accuracy")] if name.endswith(" Accuracy") else name


def parse_eval_result(path: str) -> Dict[str, Dict[str, dict]]:
    """{dataset: {method: {"acc": float|None, "fields": OrderedDict}}}"""
    out: Dict[str, Dict[str, dict]] = {}
    current = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("## Dataset:"):
                current = line.split(":", 1)[1].strip()
                out[current] = OrderedDict()
                continue
            if current is None or line.startswith("Loading JSON") or line.startswith("Loaded "):
                continue
            m = _PREPARE_RE.match(line)
            if m:
                fields = OrderedDict((k, v.strip()) for k, v in _KV_RE.findall(m.group("rest")))
                out[current]["DSC Prepare"] = {"acc": None, "fields": fields}
                continue
            m = _METRIC_RE.match(line)
            if m:
                fields = OrderedDict((k, v.strip()) for k, v in _KV_RE.findall(m.group("rest")))
                out[current][_method_key(m.group("name"))] = {"acc": float(m.group("acc")), "fields": fields}
    return out


def _mean_std(values: List[float]):
    mean = sum(values) / len(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def _to_float(v: str) -> Optional[float]:
    try:
        return float(v)
    except ValueError:
        return None


def write_summary(models: List[str], samples: List[str], datasets: List[str], out_path: str):
    # parsed[model][sample] = parse_eval_result(...)
    parsed: Dict[str, Dict[str, dict]] = defaultdict(dict)
    for model_dir in models:
        for sample in samples:
            p = os.path.join(EVAL_DIR, model_dir, sample, "eval_result.txt")
            if os.path.exists(p):
                parsed[model_dir][sample] = parse_eval_result(p)
            else:
                print(f"[요약 경고] 없음: {p}")

    lines: List[str] = []
    table_rows = []  # (dataset, model, {method: (mean, std, n_runs)})

    def method_sort_key(name: str):
        if name in METHOD_ORDER:
            return (METHOD_ORDER.index(name), 0, name)
        if name == "DSC Prepare":
            return (METHOD_ORDER.index("DSC"), 1, name)
        return (len(METHOD_ORDER), 0, name)  # Ours* 등

    for dataset in datasets:
        lines += ["", "=" * 70, f"## Dataset: {dataset}", "=" * 70]
        for model_dir in models:
            runs = [(s, parsed[model_dir][s][dataset]) for s in samples
                    if s in parsed[model_dir] and dataset in parsed[model_dir][s]]
            if not runs:
                continue
            lines.append(f"### Model: {MODEL_DIRS[model_dir]} "
                         f"(runs: {', '.join(s for s, _ in runs)}; mean ± std)")
            methods = sorted({m for _, r in runs for m in r}, key=method_sort_key)
            row = {}
            for method in methods:
                entries = [r[method] for _, r in runs if method in r]
                # 괄호 안 필드: 숫자는 run 평균, 비숫자(beta 등)는 첫 값 유지
                field_strs = []
                for key in entries[0]["fields"]:
                    vals = [e["fields"].get(key) for e in entries]
                    nums = [_to_float(v) for v in vals if v is not None]
                    if nums and all(x is not None for x in nums) and key != "beta" and key != "window":
                        field_strs.append(f"{key}={sum(nums) / len(nums):.2f}")
                    else:
                        field_strs.append(f"{key}={vals[0]}")
                missing = len(runs) - len(entries)
                suffix = f" [n_runs={len(entries)}]" if missing else ""
                if method == "DSC Prepare":
                    # 원본 형식: "DSC Prepare (input: 24344.00, output: 219939.00, time: 5.4757)"
                    prep = []
                    for key in entries[0]["fields"]:
                        nums = [_to_float(e["fields"].get(key, "")) for e in entries]
                        nums = [x for x in nums if x is not None]
                        prep.append(f"{key}: {sum(nums) / len(nums):{'.4f' if key == 'time' else '.2f'}}")
                    lines.append(f"DSC Prepare ({', '.join(prep)}){suffix}")
                    continue
                mean, std = _mean_std([e["acc"] for e in entries])
                label = method if method.startswith("Ours") else f"{method} Accuracy"
                lines.append(f"{label}: {mean:.4f} ± {std:.4f} ({', '.join(field_strs)}){suffix}")
                row[method] = (mean, std, len(entries))
            lines.append("")
            table_rows.append((dataset, model_dir, row))

    # 정확도 표 (%)
    all_methods = sorted({m for _, _, r in table_rows for m in r}, key=method_sort_key)
    header = ["Dataset", "Model"] + all_methods
    body = []
    for dataset, model_dir, row in table_rows:
        cells = [dataset, MODEL_DIRS[model_dir]]
        for m in all_methods:
            if m in row:
                mean, std, _ = row[m]
                cells.append(f"{mean * 100:.2f} ± {std * 100:.2f}")
            else:
                cells.append("-")
        body.append(cells)
    widths = [max(len(str(r[i])) for r in [header] + body) for i in range(len(header))]
    fmt = lambda r: "| " + " | ".join(str(c).ljust(w) for c, w in zip(r, widths)) + " |"
    lines += ["", "=" * 70, "## Accuracy Table (%, mean ± std over runs)", "=" * 70,
              fmt(header), "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    lines += [fmt(r) for r in body]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[요약 저장] {out_path}")


def main():
    parser = argparse.ArgumentParser(description="eval/ 반복 실행 결과 → eval_result.txt 형식 정리")
    parser.add_argument("--models", nargs="+", default=list(MODEL_DIRS), choices=list(MODEL_DIRS))
    parser.add_argument("--samples", nargs="+", default=SAMPLES)
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--seed", type=int, default=137, help="bootstrap seed (run_all_*.sh 기본값 137)")
    parser.add_argument("--work_dir", default=os.path.join(EVAL_DIR, "_work"),
                        help="중간 산출물(dsc_cmp, 로그 등) 디렉터리")
    parser.add_argument("--no_ours", action="store_true", help="Ours 계산 생략")
    parser.add_argument("--ours_label", default="Ours", help="eval_result.txt 에 기록할 Ours 라벨")
    parser.add_argument("--summary_only", action="store_true", help="기존 eval_result.txt 로 요약만 생성")
    parser.add_argument("--summary_path", default=os.path.join(EVAL_DIR, "eval_result_summary.txt"))
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)  # 기존 코드의 상대 경로(dsc/, evaluation/) 기준

    if not args.summary_only:
        for model_dir in args.models:
            for sample in args.samples:
                run_one(model_dir, sample, args.datasets, os.path.abspath(args.work_dir),
                        args.seed, not args.no_ours, args.ours_label)

    write_summary(args.models, args.samples, args.datasets, args.summary_path)


if __name__ == "__main__":
    main()
