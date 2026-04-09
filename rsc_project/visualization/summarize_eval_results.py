import argparse
import glob
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable


ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "experiments" / "results"
VIS_DIR = Path(__file__).resolve().parent

# Table column widths (고정 폭)
DATASET_W = 16
SAMPLES_W = 8
ACC_W = 8
CHAINS_W = 10
TOK_W = 12
EFF_W = 12


_DATASET_RE = re.compile(r"^## Dataset: (.+)$")
_LOADED_RE = re.compile(r"^Loaded (\d+) samples")
_COT_RE = re.compile(
    r"^CoT Accuracy: ([0-9.]+) \(avg_input_tokens=([0-9.]+), avg_output_tokens=([0-9.]+)\)"
)
_SC_RE = re.compile(
    r"^SC Accuracy: ([0-9.]+) \(avg_input_tokens=([0-9.]+), avg_output_tokens=([0-9.]+)\)"
)
_AC_RE = re.compile(
    r"^AC Accuracy: ([0-9.]+) \(beta=[0-9.]+, avg_chains=([0-9.]+), "
    r"avg_input_tokens=([0-9.]+), avg_output_tokens=([0-9.]+)\)"
)
_ESC_RE = re.compile(
    r"^ESC Accuracy: ([0-9.]+) \(window=[0-9.]+, avg_chains=([0-9.]+), "
    r"avg_input_tokens=([0-9.]+), avg_output_tokens=([0-9.]+)\)"
)
_DSC_RE = re.compile(
    r"^DSC Accuracy: ([0-9.]+) \(avg_chains=([0-9.]+), "
    r"avg_input_tokens=([0-9.]+), avg_output_tokens=([0-9.]+)\)"
)
_DSC_PREP_RE = re.compile(
    r"^DSC Prepare \(input: ([0-9.]+), output: ([0-9.]+), time: ([0-9.]+)\)"
)


def _make_ours_re(ours_variant: str, p: str) -> re.Pattern:
    """
    Build regex for lines like:
    Ours_math(p_0.7): 0.9182 (avg_chains=9.11, avg_input_tokens=106.07, avg_output_tokens=2855.19)
    """
    label = f"Ours_{ours_variant}(p_{p})"
    return re.compile(
        rf"^{re.escape(label)}: ([0-9.]+) "
        r"\(avg_chains=([0-9.]+), avg_input_tokens=([0-9.]+), avg_output_tokens=([0-9.]+)\)"
    )


def parse_eval_result(
    path: Path, ours_variant: str, p: str
) -> Dict[str, Dict[str, float]]:
    """
    eval_result.txt 하나를 파싱해서 dataset 별로 요약 정보를 리턴.

    리턴 형식:
    {
        "GSM8K": {
            "samples": 1316,
            "SC_acc": ...,
            "ESC_acc": ...,
            "AC_acc": ...,
            "ours_acc": ...,
            "ESC_chains": ...,
            "AC_chains": ...,
            "ours_paths": ...,
            "SC_in": ...,
            "SC_out": ...,
            "ESC_in": ...,
            "ESC_out": ...,
            "AC_in": ...,
            "AC_out": ...,
            "ours_in": ...,
            "ours_out": ...,
        },
        ...
    }
    """
    ours_re = _make_ours_re(ours_variant, p)
    datasets: Dict[str, Dict[str, float]] = {}

    current_name: str | None = None

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            m = _DATASET_RE.match(line)
            if m:
                current_name = m.group(1)
                datasets.setdefault(current_name, {})
                continue

            if current_name is None:
                continue

            m = _LOADED_RE.match(line)
            if m:
                datasets[current_name]["samples"] = int(m.group(1))
                continue

            m = _COT_RE.match(line)
            if m:
                acc, inp, out = m.groups()
                d = datasets[current_name]
                d["CoT_acc"] = float(acc)
                d["CoT_chains"] = 1.0  # CoT는 단일 체인
                d["CoT_in"] = float(inp)
                d["CoT_out"] = float(out)
                continue

            m = _SC_RE.match(line)
            if m:
                acc, inp, out = m.groups()
                d = datasets[current_name]
                d["SC_acc"] = float(acc)
                d["SC_in"] = float(inp)
                d["SC_out"] = float(out)
                continue

            m = _AC_RE.match(line)
            if m:
                acc, chains, inp, out = m.groups()
                d = datasets[current_name]
                d["AC_acc"] = float(acc)
                d["AC_chains"] = float(chains)
                d["AC_in"] = float(inp)
                d["AC_out"] = float(out)
                continue

            m = _ESC_RE.match(line)
            if m:
                acc, chains, inp, out = m.groups()
                d = datasets[current_name]
                d["ESC_acc"] = float(acc)
                d["ESC_chains"] = float(chains)
                d["ESC_in"] = float(inp)
                d["ESC_out"] = float(out)
                continue

            m = _DSC_RE.match(line)
            if m:
                acc, chains, inp, out = m.groups()
                d = datasets[current_name]
                d["DSC_acc"] = float(acc)
                d["DSC_chains"] = float(chains)
                d["DSC_in"] = float(inp)
                d["DSC_out"] = float(out)
                continue

            m = _DSC_PREP_RE.match(line)
            if m:
                inp, out, t = m.groups()
                d = datasets[current_name]
                d["DSC_pre_in"] = float(inp)
                d["DSC_pre_out"] = float(out)
                d["DSC_pre_time"] = float(t)
                continue

            m = ours_re.match(line)
            if m:
                acc, chains, inp, out = m.groups()
                d = datasets[current_name]
                d["ours_acc"] = float(acc)
                d["ours_chains"] = float(chains)
                d["ours_in"] = float(inp)
                d["ours_out"] = float(out)
                continue

    return datasets


def _format_float(x: float | None, width: int = 7, prec: int = 4) -> str:
    if x is None:
        return " " * width
    return f"{x:{width}.{prec}f}"


def _format_int(x: int | None, width: int = 6) -> str:
    if x is None:
        return " " * width
    return f"{x:{width}d}"

def _sep_line(length: int, ch: str = "=") -> str:
    return ch * length


def print_summary_tables(
    model_name: str, data: Dict[str, Dict[str, float]], ours_variant: str, p: str
) -> None:
    datasets = sorted(data.keys())

    print(f"Evaluating {model_name}...")
    print()

    # 1) Accuracy table
    print("Accuracy Summary")
    header_acc = (
        f"{'Dataset':<{DATASET_W}}"
        f" | {'Samples':>{SAMPLES_W}}"
        f" | {'SC Acc':>{ACC_W}}"
        f" | {'AC Acc':>{ACC_W}}"
        f" | {'ESC Acc':>{ACC_W}}"
        f" | {'DSC Acc':>{ACC_W}}"
        f" | {'Ours Acc':>{ACC_W}}"
    )
    print(_sep_line(len(header_acc), ch="="))
    print(header_acc)
    print(_sep_line(len(header_acc), ch="-"))

    for ds in datasets:
        d = data[ds]
        row = (
            f"{ds:<{DATASET_W}}"
            f" | {_format_int(int(d.get('samples', 0)), width=SAMPLES_W)}"
            f" | {_format_float((d.get('SC_acc') or 0.0) * 100.0, width=ACC_W, prec=1)}"
            f" | {_format_float((d.get('AC_acc') or 0.0) * 100.0, width=ACC_W, prec=1)}"
            f" | {_format_float((d.get('ESC_acc') or 0.0) * 100.0, width=ACC_W, prec=1)}"
            f" | {_format_float((d.get('DSC_acc') or 0.0) * 100.0, width=ACC_W, prec=1)}"
            f" | {_format_float((d.get('ours_acc') or 0.0) * 100.0, width=ACC_W, prec=1)}"
        )
        print(row)
    print(_sep_line(len(header_acc), ch="="))

    print()


def _make_emitter(
    lines: Optional[List[str]] = None, also_print: bool = True
) -> Callable[[str], None]:
    """
    emit(str) 호출 시:
    - also_print=True 이면 터미널(stdout)로 출력
    - lines 리스트가 주어지면 동일한 내용을 누적 (txt 저장용)
    """

    def emit(s: str = "") -> None:
        if lines is not None:
            lines.append(s)
        if also_print:
            print(s)

    return emit


def print_summary_tables_to(
    emit: Callable[[str], None],
    model_name: str,
    data: Dict[str, Dict[str, float]],
) -> None:
    # Dataset 출력 순서 고정
    priority_datasets = [
        "GSM8K",
        "MATH500_test",
        "AMC23",
        "AIME_2024",
        "AIME_2025",
        "GPQA-Diamond",
        "MMLU-Pro",
    ]
    existing = set(data.keys())
    # 우선순위 목록 중 실제로 존재하는 것만 순서대로
    ordered = [ds for ds in priority_datasets if ds in existing]
    # 그 외 데이터셋은 이름 순으로 뒤에 붙이기
    remaining = sorted(existing - set(ordered))
    datasets = ordered + remaining

    emit(f"Evaluating {model_name}...")
    emit()

    # 1) Accuracy table
    emit("Accuracy Summary")
    header_acc = (
        f"{'Dataset':<{DATASET_W}}"
        f" | {'Samples':>{SAMPLES_W}}"
        f" | {'CoT Acc':>{ACC_W}}"
        f" | {'SC Acc':>{ACC_W}}"
        f" | {'AC Acc':>{ACC_W}}"
        f" | {'ESC Acc':>{ACC_W}}"
        f" | {'DSC Acc':>{ACC_W}}"
        f" | {'Ours Acc':>{ACC_W}}"
    )
    emit(_sep_line(len(header_acc), ch="="))
    emit(header_acc)
    emit(_sep_line(len(header_acc), ch="-"))

    for ds in datasets:
        d = data[ds]
        cot = d.get("CoT_acc")
        sc = d.get("SC_acc")
        ac = d.get("AC_acc")
        esc = d.get("ESC_acc")
        dsc = d.get("DSC_acc")
        ours = d.get("ours_acc")
        row = (
            f"{ds:<{DATASET_W}}"
            f" | {_format_int(int(d.get('samples', 0)), width=SAMPLES_W)}"
            f" | {_format_float(cot * 100.0 if cot is not None else None, width=ACC_W, prec=1)}"
            f" | {_format_float(sc * 100.0 if sc is not None else None, width=ACC_W, prec=1)}"
            f" | {_format_float(ac * 100.0 if ac is not None else None, width=ACC_W, prec=1)}"
            f" | {_format_float(esc * 100.0 if esc is not None else None, width=ACC_W, prec=1)}"
            f" | {_format_float(dsc * 100.0 if dsc is not None else None, width=ACC_W, prec=1)}"
            f" | {_format_float(ours * 100.0 if ours is not None else None, width=ACC_W, prec=1)}"
        )
        emit(row)
    emit(_sep_line(len(header_acc), ch="="))
    emit()

    # 2) Chains table
    emit("Chains Summary")
    header_ch = (
        f"{'Dataset':<{DATASET_W}}"
        f" | {'CoT Chains':>{CHAINS_W}}"
        f" | {'SC Chains':>{CHAINS_W}}"
        f" | {'AC Chains':>{CHAINS_W}}"
        f" | {'ESC Chains':>{CHAINS_W}}"
        f" | {'DSC Chains':>{CHAINS_W}}"
        f" | {'Ours Chains':>{CHAINS_W}}"
    )
    emit(_sep_line(len(header_ch), ch="="))
    emit(header_ch)
    emit(_sep_line(len(header_ch), ch="-"))

    for ds in datasets:
        d = data[ds]
        row = (
            f"{ds:<{DATASET_W}}"
            f" | {_format_float(d.get('CoT_chains'), width=CHAINS_W, prec=2)}"
            f" | {_format_float(40.0, width=CHAINS_W, prec=2)}"
            f" | {_format_float(d.get('AC_chains'), width=CHAINS_W, prec=2)}"
            f" | {_format_float(d.get('ESC_chains'), width=CHAINS_W, prec=2)}"
            f" | {_format_float(d.get('DSC_chains'), width=CHAINS_W, prec=2)}"
            f" | {_format_float(d.get('ours_chains'), width=CHAINS_W, prec=2)}"
        )
        emit(row)
    emit(_sep_line(len(header_ch), ch="="))
    emit()

    # 3) Token usage table
    emit("Token Usage Summary")
    header_tok = (
        f"{'Dataset':<{DATASET_W}}"
        f" | {'CoT Input':>{TOK_W}}"
        f" | {'CoT Output':>{TOK_W}}"
        f" | {'SC Input':>{TOK_W}}"
        f" | {'SC Output':>{TOK_W}}"
        f" | {'AC Input':>{TOK_W}}"
        f" | {'AC Output':>{TOK_W}}"
        f" | {'ESC Input':>{TOK_W}}"
        f" | {'ESC Output':>{TOK_W}}"
        f" | {'DSC Input':>{TOK_W}}"
        f" | {'DSC Output':>{TOK_W}}"
        f" | {'Ours Input':>{TOK_W}}"
        f" | {'Ours Output':>{TOK_W}}"
    )
    emit(_sep_line(len(header_tok), ch="="))
    emit(header_tok)
    emit(_sep_line(len(header_tok), ch="-"))

    for ds in datasets:
        d = data[ds]
        row = (
            f"{ds:<{DATASET_W}}"
            f" | {_format_float(d.get('CoT_in'), width=TOK_W, prec=2)}"
            f" | {_format_float(d.get('CoT_out'), width=TOK_W, prec=2)}"
            f" | {_format_float(d.get('SC_in'), width=TOK_W, prec=2)}"
            f" | {_format_float(d.get('SC_out'), width=TOK_W, prec=2)}"
            f" | {_format_float(d.get('AC_in'), width=TOK_W, prec=2)}"
            f" | {_format_float(d.get('AC_out'), width=TOK_W, prec=2)}"
            f" | {_format_float(d.get('ESC_in'), width=TOK_W, prec=2)}"
            f" | {_format_float(d.get('ESC_out'), width=TOK_W, prec=2)}"
            f" | {_format_float(d.get('DSC_in'), width=TOK_W, prec=2)}"
            f" | {_format_float(d.get('DSC_out'), width=TOK_W, prec=2)}"
            f" | {_format_float(d.get('ours_in'), width=TOK_W, prec=2)}"
            f" | {_format_float(d.get('ours_out'), width=TOK_W, prec=2)}"
        )
        emit(row)
    emit(_sep_line(len(header_tok), ch="="))
    emit()

    # 4) Tokens (Input + Output) table
    emit("Total Tokens Summary (Input + Output)")
    header_tok_sum = (
        f"{'Dataset':<{DATASET_W}}"
        f" | {'CoT Tokens':>{TOK_W}}"
        f" | {'SC Tokens':>{TOK_W}}"
        f" | {'AC Tokens':>{TOK_W}}"
        f" | {'ESC Tokens':>{TOK_W}}"
        f" | {'DSC Tokens':>{TOK_W}}"
        f" | {'Ours Tokens':>{TOK_W}}"
    )
    emit(_sep_line(len(header_tok_sum), ch="="))
    emit(header_tok_sum)
    emit(_sep_line(len(header_tok_sum), ch="-"))

    for ds in datasets:
        d = data[ds]
        cot_tok = (
            (d.get("CoT_in") or 0.0) + (d.get("CoT_out") or 0.0)
            if d.get("CoT_in") is not None and d.get("CoT_out") is not None
            else None
        )
        sc_tok = (
            (d.get("SC_in") or 0.0) + (d.get("SC_out") or 0.0)
            if d.get("SC_in") is not None and d.get("SC_out") is not None
            else None
        )
        ac_tok = (
            (d.get("AC_in") or 0.0) + (d.get("AC_out") or 0.0)
            if d.get("AC_in") is not None and d.get("AC_out") is not None
            else None
        )
        esc_tok = (
            (d.get("ESC_in") or 0.0) + (d.get("ESC_out") or 0.0)
            if d.get("ESC_in") is not None and d.get("ESC_out") is not None
            else None
        )
        dsc_tok = (
            (d.get("DSC_in") or 0.0) + (d.get("DSC_out") or 0.0)
            if d.get("DSC_in") is not None and d.get("DSC_out") is not None
            else None
        )
        ours_tok = (
            (d.get("ours_in") or 0.0) + (d.get("ours_out") or 0.0)
            if d.get("ours_in") is not None and d.get("ours_out") is not None
            else None
        )

        row = (
            f"{ds:<{DATASET_W}}"
            f" | {_format_float(cot_tok, width=TOK_W, prec=2)}"
            f" | {_format_float(sc_tok, width=TOK_W, prec=2)}"
            f" | {_format_float(ac_tok, width=TOK_W, prec=2)}"
            f" | {_format_float(esc_tok, width=TOK_W, prec=2)}"
            f" | {_format_float(dsc_tok, width=TOK_W, prec=2)}"
            f" | {_format_float(ours_tok, width=TOK_W, prec=2)}"
        )
        emit(row)
    emit(_sep_line(len(header_tok_sum), ch="="))
    emit()

    # 5) DSC Prepare tokens table (총 prepare input/output 토큰)
    emit("DSC Prepare Tokens Summary (total tokens)")
    header_prep = (
        f"{'Dataset':<{DATASET_W}}"
        f" | {'Prepare Input':>{TOK_W}}"
        f" | {'Prepare Output':>{TOK_W}}"
        f" | {'Prepare Total':>{TOK_W}}"
    )
    emit(_sep_line(len(header_prep), ch="="))
    emit(header_prep)
    emit(_sep_line(len(header_prep), ch="-"))

    for ds in datasets:
        d = data[ds]
        prep_in = d.get("DSC_pre_in")
        prep_out = d.get("DSC_pre_out")
        prep_total = (
            (prep_in or 0.0) + (prep_out or 0.0)
            if prep_in is not None and prep_out is not None
            else None
        )
        row = (
            f"{ds:<{DATASET_W}}"
            f" | {_format_float(prep_in, width=TOK_W, prec=2)}"
            f" | {_format_float(prep_out, width=TOK_W, prec=2)}"
            f" | {_format_float(prep_total, width=TOK_W, prec=2)}"
        )
        emit(row)
    emit(_sep_line(len(header_prep), ch="="))
    emit()

    # 6) Relative Accuracy & Token Ratio Summary table
    emit("Relative Accuracy & Token Ratio Summary")
    header_rel = (
        f"{'Dataset':<{DATASET_W}}"
        f" | {'CoT RelAcc':>{EFF_W}}"
        f" | {'CoT RelTok':>{EFF_W}}"
        f" | {'AC RelAcc':>{EFF_W}}"
        f" | {'AC RelTok':>{EFF_W}}"
        f" | {'ESC RelAcc':>{EFF_W}}"
        f" | {'ESC RelTok':>{EFF_W}}"
        f" | {'DSC RelAcc':>{EFF_W}}"
        f" | {'DSC RelTok':>{EFF_W}}"
        f" | {'Ours RelAcc':>{EFF_W}}"
        f" | {'Ours RelTok':>{EFF_W}}"
    )
    emit(_sep_line(len(header_rel), ch="="))
    emit(header_rel)
    emit(_sep_line(len(header_rel), ch="-"))

    for ds in datasets:
        d = data[ds]
        sc_acc = d.get("SC_acc")
        sc_in = d.get("SC_in")
        sc_out = d.get("SC_out")
        sc_tokens = None
        if sc_in is not None and sc_out is not None:
            sc_tokens = sc_in + sc_out

        def _efficiency_rel(
            key_acc: str, key_in: str, key_out: str
        ) -> tuple[Optional[float], Optional[float], Optional[float]]:
            acc = d.get(key_acc)
            tin = d.get(key_in)
            tout = d.get(key_out)
            if (
                acc is None
                or sc_acc is None
                or sc_acc == 0
                or tin is None
                or tout is None
                or sc_tokens is None
                or sc_tokens == 0
            ):
                return None, None, None
            # RelAcc = Acc_method / Acc_SC * 100
            rel_acc = (acc / sc_acc) * 100.0
            # RelTok = Tokens_method / Tokens_SC * 100
            tokens = tin + tout
            token_ratio = (tokens / sc_tokens) * 100.0
            # Eff = RelAcc / RelTok
            eff = rel_acc / token_ratio if token_ratio > 0 else None
            return rel_acc, token_ratio, eff

        cot_rel, cot_tr, _ = _efficiency_rel("CoT_acc", "CoT_in", "CoT_out")
        ac_rel, ac_tr, _ = _efficiency_rel("AC_acc", "AC_in", "AC_out")
        esc_rel, esc_tr, _ = _efficiency_rel("ESC_acc", "ESC_in", "ESC_out")
        dsc_rel, dsc_tr, _ = _efficiency_rel("DSC_acc", "DSC_in", "DSC_out")
        ours_rel, ours_tr, _ = _efficiency_rel("ours_acc", "ours_in", "ours_out")

        row = (
            f"{ds:<{DATASET_W}}"
            f" | {_format_float(cot_rel, width=EFF_W, prec=1)}"
            f" | {_format_float(cot_tr, width=EFF_W, prec=1)}"
            f" | {_format_float(ac_rel, width=EFF_W, prec=1)}"
            f" | {_format_float(ac_tr, width=EFF_W, prec=1)}"
            f" | {_format_float(esc_rel, width=EFF_W, prec=1)}"
            f" | {_format_float(esc_tr, width=EFF_W, prec=1)}"
            f" | {_format_float(dsc_rel, width=EFF_W, prec=1)}"
            f" | {_format_float(dsc_tr, width=EFF_W, prec=1)}"
            f" | {_format_float(ours_rel, width=EFF_W, prec=1)}"
            f" | {_format_float(ours_tr, width=EFF_W, prec=1)}"
        )
        emit(row)
    emit(_sep_line(len(header_rel), ch="="))
    emit()

    # 7) Efficiency Score Summary table
    emit("Efficiency Score Summary")
    header_eff = (
        f"{'Dataset':<{DATASET_W}}"
        f" | {'CoT Eff':>{EFF_W}}"
        f" | {'AC Eff':>{EFF_W}}"
        f" | {'ESC Eff':>{EFF_W}}"
        f" | {'DSC Eff':>{EFF_W}}"
        f" | {'Ours Eff':>{EFF_W}}"
    )
    emit(_sep_line(len(header_eff), ch="="))
    emit(header_eff)
    emit(_sep_line(len(header_eff), ch="-"))

    for ds in datasets:
        d = data[ds]
        sc_acc = d.get("SC_acc")
        sc_in = d.get("SC_in")
        sc_out = d.get("SC_out")
        sc_tokens = None
        if sc_in is not None and sc_out is not None:
            sc_tokens = sc_in + sc_out

        def _efficiency_eff(
            key_acc: str, key_in: str, key_out: str
        ) -> tuple[Optional[float], Optional[float], Optional[float]]:
            acc = d.get(key_acc)
            tin = d.get(key_in)
            tout = d.get(key_out)
            if (
                acc is None
                or sc_acc is None
                or sc_acc == 0
                or tin is None
                or tout is None
                or sc_tokens is None
                or sc_tokens == 0
            ):
                return None, None, None
            # RelAcc = Acc_method / Acc_SC * 100
            rel_acc = (acc / sc_acc) * 100.0
            # RelTok = Tokens_method / Tokens_SC * 100
            tokens = tin + tout
            token_ratio = (tokens / sc_tokens) * 100.0
            # Eff = RelAcc / RelTok
            eff = rel_acc / token_ratio if token_ratio > 0 else None
            return rel_acc, token_ratio, eff

        _, _, cot_eff = _efficiency_eff("CoT_acc", "CoT_in", "CoT_out")
        _, _, ac_eff = _efficiency_eff("AC_acc", "AC_in", "AC_out")
        _, _, esc_eff = _efficiency_eff("ESC_acc", "ESC_in", "ESC_out")
        _, _, dsc_eff = _efficiency_eff("DSC_acc", "DSC_in", "DSC_out")
        _, _, ours_eff = _efficiency_eff("ours_acc", "ours_in", "ours_out")

        row = (
            f"{ds:<{DATASET_W}}"
            f" | {_format_float(cot_eff, width=EFF_W, prec=2)}"
            f" | {_format_float(ac_eff, width=EFF_W, prec=2)}"
            f" | {_format_float(esc_eff, width=EFF_W, prec=2)}"
            f" | {_format_float(dsc_eff, width=EFF_W, prec=2)}"
            f" | {_format_float(ours_eff, width=EFF_W, prec=2)}"
        )
        emit(row)
    emit(_sep_line(len(header_eff), ch="="))
    emit()


def find_eval_files(stages_config: str, result_filename: str) -> List[Tuple[str, Path]]:
    """
    results 폴더 안에서 지정된 stages_config 를 가진 모든 결과 txt 경로를 찾는다.

    리턴: [(model_name, path), ...]
          model_name 은 `<provider>/<model>` 형태 (예: `Qwen/Qwen2.5-7B-Instruct`)
    """
    pattern = str(RESULTS_DIR / "*" / "*" / stages_config / result_filename)
    files = glob.glob(pattern)
    result: List[Tuple[str, Path]] = []
    for f in sorted(files):
        p = Path(f)
        # .../results/provider/model/stages_config/<result_filename>
        try:
            provider = p.parents[2].name
            model = p.parents[1].name
        except IndexError:
            provider, model = "unknown", p.parent.name
        model_name = f"{provider}/{model}"
        result.append((model_name, p))
    return result


def sort_eval_files(
    eval_files: List[Tuple[str, Path]]
) -> List[Tuple[str, Path]]:
    """
    모델 출력 순서를 제어한다.
    요청: Qwen2.5 3B, 7B, 14B, gemma-3-4b 순서 우선.

    나머지 모델은 알파벳 순으로 뒤에 붙인다.
    """

    priority_order = [
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct",
        "google/gemma-3-4b-it",
    ]
    idx = {name: i for i, name in enumerate(priority_order)}

    def _key(item: Tuple[str, Path]) -> Tuple[int, str]:
        name, _ = item
        return (idx.get(name, len(priority_order)), name)

    return sorted(eval_files, key=_key)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="results 폴더의 결과 txt들을 읽어서 요약 테이블을 출력합니다."
    )
    parser.add_argument(
        "--stages_config",
        type=str,
        default="stages-100",
        help="사용할 stages 설정 디렉토리 이름 (예: stages-40, stages-100)",
    )
    parser.add_argument(
        "--result_filename",
        type=str,
        default="eval_result.txt",
        help="각 모델의 stages 폴더 안에서 읽어올 결과 파일명 (기본값: eval_result.txt). "
             "예: eval_single_137.txt",
    )
    parser.add_argument(
        "--ours_type",
        type=str,
        default="probe",
        help="Ours 계열 타입 (eval_result.txt 라벨과 일치). 예: probe, math, math2500",
    )
    parser.add_argument(
        "--p",
        type=str,
        default="1.0",
        help="Ours 에 사용할 p 값 (eval_result.txt 에 적힌 그대로, 예: 0.7, 0.75, 1.0)",
    )
    parser.add_argument(
        "--save_txt",
        action="store_true",
        help="출력 결과를 visualization 폴더에 txt로 저장합니다 (터미널 출력은 유지).",
    )
    parser.add_argument(
        "--out_name",
        type=str,
        default=None,
        help="저장할 txt 파일명 (기본값: 자동 생성). 예: summary_stages-100_probe_p1.0.txt",
    )

    args = parser.parse_args()

    # 사용 편의를 위해 숫자만 넣어도 동작하도록 정규화
    # 예: 100 -> stages-100, 40 -> stages-40
    stages_dir = (
        args.stages_config
        if args.stages_config.startswith("stages-")
        else f"stages-{args.stages_config}"
    )

    eval_files = find_eval_files(stages_dir, args.result_filename)
    if not eval_files:
        print(
            f"No {args.result_filename} found under {RESULTS_DIR} with stages_config={stages_dir}"
        )
        return

    # 사용자 지정 우선순위(Qwen 2.5 3B,7B,14B, gemma-3-4b)를 반영해 정렬
    eval_files = sort_eval_files(eval_files)

    out_lines: Optional[List[str]] = [] if args.save_txt else None
    emit = _make_emitter(lines=out_lines, also_print=True)

    # 저장 파일명 결정
    if args.save_txt:
        safe_p = str(args.p).replace("/", "_")
        default_name = f"summary_{stages_dir}_{args.ours_type}_p{safe_p}.txt"
        out_name = args.out_name or default_name
        out_path = (VIS_DIR / out_name).resolve()

    first = True
    for model_name, path in eval_files:
        # 모델 사이를 시각적으로 구분하기 위해 빈 줄 5줄 추가
        if not first:
            for _ in range(5):
                emit()
        first = False

        data = parse_eval_result(path, args.ours_type, args.p)
        if not data:
            emit(f"Skip {model_name} ({path}) - no datasets parsed")
            continue
        print_summary_tables_to(emit, model_name, data)

    if args.save_txt and out_lines is not None:
        os.makedirs(out_path.parent, exist_ok=True)
        out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
        print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()

