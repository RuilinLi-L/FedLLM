#!/usr/bin/env python3
"""
Parse local experiment logs (shell tee under log/runs/, train.py logs, feasibility runs)
and write a summary CSV (optional Markdown).

Log kinds
---------
- **attack_dager** (attack.py / attack_new.py): prefer the final ``RESULT SUMMARY`` block when
  present; otherwise fall back to ROUGE aggregate, per-batch Rec Token rates, and the last
  ``[Aggregate metrics]:`` block before ``Done with all.`` if present.
- **train** (train.py): ``metric eval:`` / ``metric train:`` lines; argv from
  ``terminal log started`` banner when present.
- **feasibility** (attack_len_increment.py): ``Sample accuracy`` / ``Sample set accuracy``.

Shell auto-log (scripts/_dager_auto_log.sh) only records the **shell** argv (e.g.
``gpt2 cola 1 --rank_tol 1e-8``), not the full default ``attack.py`` flags; columns
``dataset_guess`` / ``batch_size_guess`` infer positions 1–2 for standard dataset scripts.

Dependencies: stdlib only.
"""
from __future__ import annotations

import argparse
import ast
import csv
import re
import statistics
from pathlib import Path


RUN_START_RE = re.compile(
    r"^===== run start (?P<ts>.+?) tag=(?P<tag>\S+) argv: (?P<argv>.+) =====\s*$"
)
TERMINAL_LOG_RE = re.compile(
    r"===== terminal log started .+? =====\s*\nargv: (?P<argv>.+)\s*$", re.MULTILINE
)
REC_LINE_RE = re.compile(
    r"Rec MaxB Token:\s*([0-9.]+),\s*Rec Token:\s*([0-9.]+)"
)
RANK_LINE_RE = re.compile(r"^(\d+)/(\d+)/(\d+)\s*$")
AGG_HEADER = "[Aggregate metrics]:"
ROUGE_LINE_RE = re.compile(
    r"^(rouge1|rouge2|rougeL|rougeLsum)\s+\|\s+fm:\s+([0-9.]+)\s+\|\s+p:\s+([0-9.]+)\s+\|\s+r:\s+([0-9.]+)\s*$"
)
R1R2_RE = re.compile(r"r1fm\+r2fm = ([0-9.]+)\s*$")
INPUT_TIME_RE = re.compile(
    r"input #(\d+) time:\s*([^|]+)\|\s*total time:\s*(.+?)\s*$"
)
SAMPLE_ACC_RE = re.compile(r"^Sample accuracy:\s*([0-9.]+)\s*$", re.MULTILINE)
SAMPLE_SET_ACC_RE = re.compile(
    r"^Sample set accuracy\s+([0-9.eE+-]+)\s*$", re.MULTILINE
)
METRIC_EVAL_RE = re.compile(r"^metric eval:\s*(\{.*\})\s*$", re.MULTILINE)
METRIC_TRAIN_RE = re.compile(r"^metric train:\s*(\{.*\})\s*$", re.MULTILINE)
RESULT_SUMMARY_BLOCK_RE = re.compile(
    r"^===== RESULT SUMMARY START =====\s*\n(?P<body>.*?)^===== RESULT SUMMARY END =====\s*$",
    re.MULTILINE | re.DOTALL,
)

# Shell tags where argv[1], argv[2] are DATASET and BATCH_SIZE (see README scripts).
TAG_DATASET_BATCH = frozenset(
    {
        "gpt2",
        "gpt2-large",
        "gpt2-ft",
        "bert",
        "lora",
        "llama",
        "llama_3.1",
        "gpt2_dp",
    }
)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _parse_shell_header(line: str) -> dict | None:
    m = RUN_START_RE.match(line.strip())
    if not m:
        return None
    return {"ts": m.group("ts").strip(), "tag": m.group("tag"), "argv": m.group("argv")}


def _parse_terminal_argv(text: str) -> list[str] | None:
    m = TERMINAL_LOG_RE.search(text)
    if not m:
        return None
    try:
        return ast.literal_eval(m.group("argv").strip())
    except (SyntaxError, ValueError):
        return None


def _shell_argv_tokens(argv_str: str) -> list[str]:
    # Good enough for logs: split; users rarely quote spaces in these runs.
    return argv_str.strip().split()


def _infer_dataset_batch(tag: str, tokens: list[str]) -> tuple[str, str, str]:
    """Return (dataset_guess, batch_size_guess, extra_cli)."""
    if not tokens:
        return "", "", ""
    if tag == "dager_dp" and len(tokens) >= 3:
        ds, bs = tokens[1], tokens[2]
        extra = " ".join(tokens[3:])
        return ds, bs, extra
    if tag not in TAG_DATASET_BATCH or len(tokens) < 3:
        return "", "", " ".join(tokens[1:])
    ds = tokens[1]
    bs = tokens[2]
    extra = " ".join(tokens[3:])
    return ds, bs, extra


def _parse_cli_flags(joined: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for m in re.finditer(r"--([\w_]+)\s+(\S+)", joined):
        out[m.group(1)] = m.group(2)
    if (m := re.search(r"--([\w_]+)=(\S+)", joined)):
        out[m.group(1)] = m.group(2)
    return out


def _parse_train_argv(argv_list: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    i = 0
    while i < len(argv_list):
        a = argv_list[i]
        if a.startswith("--") and i + 1 < len(argv_list) and not argv_list[i + 1].startswith(
            "-"
        ):
            key = a[2:].replace("-", "_")
            out[key] = argv_list[i + 1]
            i += 2
        else:
            i += 1
    return out


def _parse_result_summary(text: str) -> dict[str, str]:
    matches = list(RESULT_SUMMARY_BLOCK_RE.finditer(text))
    if not matches:
        return {}
    body = matches[-1].group("body")
    summary: dict[str, str] = {}
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        summary[key] = value
    return summary


def _parse_aggregate_section(section: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in section.splitlines():
        m = ROUGE_LINE_RE.match(line.strip())
        if m:
            name, fm, p, r = m.group(1), m.group(2), m.group(3), m.group(4)
            metrics[f"agg_{name}_fm"] = float(fm)
            metrics[f"agg_{name}_p"] = float(p)
            metrics[f"agg_{name}_r"] = float(r)
    m = R1R2_RE.search(section)
    if m:
        metrics["agg_r1fm_r2fm"] = float(m.group(1))
    return metrics


def _final_aggregate_text(text: str) -> str:
    if "Done with all." in text:
        prefix = text.split("Done with all.")[0]
    else:
        prefix = text
    idx = prefix.rfind(AGG_HEADER)
    if idx < 0:
        return ""
    chunk = prefix[idx:]
    m = re.search(r"\ninput #\d+ time:", chunk)
    if m:
        return chunk[: m.start()]
    m2 = re.search(r"\n\nRunning input #", chunk)
    if m2:
        return chunk[: m2.start()]
    return chunk


def _last_input_time(text: str) -> tuple[str, str, str]:
    last = None
    for m in INPUT_TIME_RE.finditer(text):
        last = m
    if not last:
        return "", "", ""
    return last.group(1), last.group(2).strip(), last.group(3).strip()


def parse_attack_dager(text: str, meta: dict) -> dict:
    row = {**meta}
    row["log_kind"] = "attack_dager"
    summary = _parse_result_summary(text)
    if summary:
        row.update(summary)

    rec_pairs = REC_LINE_RE.findall(text)
    if rec_pairs and "rec_token_mean" not in row:
        maxb = [float(a) for a, _ in rec_pairs]
        tok = [float(b) for _, b in rec_pairs]
        row["n_rec_lines"] = str(len(rec_pairs))
        row["rec_maxb_token_mean"] = f"{statistics.mean(maxb):.6f}"
        row["rec_token_mean"] = f"{statistics.mean(tok):.6f}"
        if len(tok) > 1:
            row["rec_token_std"] = f"{statistics.stdev(tok):.6f}"
        else:
            row["rec_token_std"] = ""
    else:
        if "n_rec_lines" not in row:
            row["n_rec_lines"] = row.get("n_inputs_completed", "0")
        row.setdefault("rec_maxb_token_mean", "")
        row.setdefault("rec_token_mean", "")
        row.setdefault("rec_token_std", "")

    ranks = []
    for line in text.splitlines():
        m = RANK_LINE_RE.match(line.strip())
        if m:
            ranks.append((int(m.group(1)), int(m.group(2)), int(m.group(3))))
    if ranks and "rank_b_mean" not in row:
        row["n_rank_lines"] = str(len(ranks))
        row["rank_b_mean"] = f"{statistics.mean(r[0] for r in ranks):.4f}"
    else:
        row.setdefault("n_rank_lines", "0")
        row.setdefault("rank_b_mean", "")

    agg_sec = _final_aggregate_text(text)
    agg = _parse_aggregate_section(agg_sec)
    for k, v in agg.items():
        row.setdefault(k, f"{v:.6f}" if isinstance(v, float) else str(v))

    inp, tin, ttot = _last_input_time(text)
    if inp:
        row.setdefault("last_input_idx", inp)
    if tin:
        row.setdefault("last_input_time", tin)
    if ttot:
        row.setdefault("last_total_time", ttot)
    return row


def parse_feasibility(text: str, meta: dict) -> dict:
    row = {**meta}
    row["log_kind"] = "feasibility"
    s_acc = [float(x) for x in SAMPLE_ACC_RE.findall(text)]
    set_acc = [float(x) for x in SAMPLE_SET_ACC_RE.findall(text)]
    row["n_sample_accuracy_lines"] = str(len(s_acc))
    row["sample_accuracy_mean"] = f"{statistics.mean(s_acc):.4f}" if s_acc else ""
    row["sample_accuracy_last"] = f"{s_acc[-1]:.4f}" if s_acc else ""
    row["n_sample_set_accuracy_lines"] = str(len(set_acc))
    row["sample_set_accuracy_last"] = f"{set_acc[-1]:.6f}" if set_acc else ""
    return row


def parse_train(text: str, meta: dict) -> dict:
    row = {**meta}
    row["log_kind"] = "train"
    argv_list = _parse_terminal_argv(text)
    if argv_list:
        flags = _parse_train_argv(argv_list)
        for k in (
            "dataset",
            "batch_size",
            "noise",
            "num_epochs",
            "model_path",
            "train_method",
            "lora_r",
        ):
            if k in flags:
                row[f"train_{k}"] = flags[k]

    evals = []
    for m in METRIC_EVAL_RE.finditer(text):
        try:
            evals.append(ast.literal_eval(m.group(1)))
        except (SyntaxError, ValueError):
            pass
    if evals:
        last = evals[-1]
        row["metric_eval_raw"] = repr(last)
        for k, v in last.items():
            row[f"eval_{k}"] = str(v)
    trains = []
    for m in METRIC_TRAIN_RE.finditer(text):
        try:
            trains.append(ast.literal_eval(m.group(1)))
        except (SyntaxError, ValueError):
            pass
    if trains:
        row["metric_train_last_raw"] = repr(trains[-1])
    return row


def classify_and_parse(path: Path, text: str) -> dict:
    lines = text.splitlines()
    meta: dict[str, str] = {"log_path": str(path.resolve())}

    header = _parse_shell_header(lines[0]) if lines else None
    if header:
        meta["run_ts"] = header["ts"]
        meta["tag"] = header["tag"]
        meta["shell_argv_raw"] = header["argv"]
        tokens = _shell_argv_tokens(header["argv"])
        ds, bs, extra = _infer_dataset_batch(header["tag"], tokens)
        meta["dataset_guess"] = ds
        meta["batch_size_guess"] = bs
        meta["extra_cli"] = extra
        flags = _parse_cli_flags(header["argv"])
        if "defense_noise" in flags:
            meta["defense_noise"] = flags["defense_noise"]
        if "rank_tol" in flags:
            meta["rank_tol"] = flags["rank_tol"]
        if header["tag"] == "gpt2_dp" and len(tokens) >= 4:
            meta["defense_noise"] = meta.get("defense_noise") or tokens[3]
        meta["model_path_guess"] = ""
        if header["tag"] == "dager_dp" and len(tokens) >= 4:
            meta["model_path_guess"] = tokens[3]
    else:
        meta["run_ts"] = ""
        meta["tag"] = ""
        meta["shell_argv_raw"] = ""
        meta["dataset_guess"] = ""
        meta["batch_size_guess"] = ""
        meta["extra_cli"] = ""
        meta["defense_noise"] = ""
        meta["rank_tol"] = ""
        meta["model_path_guess"] = ""

    has_result_summary = bool(_parse_result_summary(text))
    has_agg = AGG_HEADER in text or re.search(r"^rouge1\s+\|", text, re.MULTILINE) is not None
    has_feas_acc = SAMPLE_ACC_RE.search(text) is not None

    if has_result_summary:
        return parse_attack_dager(text, meta)
    if has_feas_acc and not has_agg:
        return parse_feasibility(text, meta)
    if has_agg:
        return parse_attack_dager(text, meta)
    if "===== terminal log started" in text and "Attacking.." not in text:
        return parse_train(text, meta)
    if METRIC_EVAL_RE.search(text) or METRIC_TRAIN_RE.search(text):
        return parse_train(text, meta)

    row = {**meta}
    row["log_kind"] = "unknown"
    return row


def _all_keys(rows: list[dict]) -> list[str]:
    keys: set[str] = set()
    for r in rows:
        keys.update(r.keys())
    preferred = [
        "log_path",
        "log_kind",
        "tag",
        "run_ts",
        "shell_argv_raw",
        "dataset_guess",
        "batch_size_guess",
        "extra_cli",
        "defense_noise",
        "rank_tol",
        "model_path_guess",
        "summary_version",
        "result_status",
        "dataset",
        "split",
        "task",
        "model_path",
        "finetuned_path",
        "batch_size",
        "defense",
        "defense_param_name",
        "defense_param_value",
        "n_inputs_requested",
        "n_inputs_completed",
        "last_rec_status",
        "rec_l1_mean",
        "rec_l1_maxb_mean",
        "rec_l2_mean",
        "n_rec_lines",
        "rec_token_mean",
        "rec_token_std",
        "rec_maxb_token_mean",
        "n_rank_lines",
        "rank_b_mean",
        "agg_rouge1_fm",
        "agg_rouge1_p",
        "agg_rouge1_r",
        "agg_rouge2_fm",
        "agg_rouge2_p",
        "agg_rouge2_r",
        "agg_rougeL_fm",
        "agg_rougeL_p",
        "agg_rougeL_r",
        "agg_rougeLsum_fm",
        "agg_rougeLsum_p",
        "agg_rougeLsum_r",
        "agg_r1fm_r2fm",
        "last_input_idx",
        "last_input_time",
        "last_total_time",
        "error_type",
        "error_message",
        "n_sample_accuracy_lines",
        "sample_accuracy_mean",
        "sample_accuracy_last",
        "n_sample_set_accuracy_lines",
        "sample_set_accuracy_last",
        "train_dataset",
        "train_batch_size",
        "train_noise",
        "train_num_epochs",
        "train_model_path",
        "train_train_method",
        "train_lora_r",
        "metric_eval_raw",
        "metric_train_last_raw",
    ]
    rest = sorted(k for k in keys if k not in preferred)
    return [k for k in preferred if k in keys] + rest


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = _all_keys(rows)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def write_markdown(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = _all_keys(rows)

    def esc(cell: str) -> str:
        return cell.replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join("---" for _ in keys) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(esc(str(r.get(k, ""))) for k in keys) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_paths(inputs: list[str]) -> list[Path]:
    out: list[Path] = []
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            out.extend(sorted(p.glob("*.txt")))
        else:
            out.append(p)
    # unique preserve order
    seen = set()
    uniq: list[Path] = []
    for p in out:
        rp = p.resolve()
        if rp not in seen and p.exists():
            seen.add(rp)
            uniq.append(p)
    return uniq


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Summarize DAGER / train / feasibility logs into CSV (optional Markdown)."
    )
    ap.add_argument(
        "paths",
        nargs="*",
        default=[],
        help="Log files or directories (default: log/runs/*.txt)",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("experiments_summary.csv"),
        help="Output CSV path",
    )
    ap.add_argument(
        "--markdown",
        type=Path,
        default=None,
        help="Also write a Markdown pipe table to this path",
    )
    args = ap.parse_args()

    if args.paths:
        paths = collect_paths([str(p) for p in args.paths])
    else:
        default_dir = Path("log/runs")
        paths = sorted(default_dir.glob("*.txt")) if default_dir.is_dir() else []

    rows = []
    for p in paths:
        try:
            text = _read_text(p)
        except OSError as e:
            rows.append({"log_path": str(p.resolve()), "log_kind": "error", "parse_error": str(e)})
            continue
        rows.append(classify_and_parse(p, text))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output, rows)
    print(f"Wrote {len(rows)} row(s) to {args.output.resolve()}")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(args.markdown, rows)
        print(f"Wrote Markdown to {args.markdown.resolve()}")


if __name__ == "__main__":
    main()
