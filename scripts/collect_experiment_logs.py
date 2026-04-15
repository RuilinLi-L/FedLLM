#!/usr/bin/env python3
"""
Parse local experiment logs and write:
- raw per-log CSV / Markdown summaries
- aggregated utility results from training logs
- privacy-utility tradeoff tables by joining train utility with attack logs
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
LOSS_TRAIN_RE = re.compile(r"^loss train:\s*([0-9.eE+-]+)\s*$", re.MULTILINE)
RESULT_SUMMARY_BLOCK_RE = re.compile(
    r"^===== RESULT SUMMARY START =====\s*\n(?P<body>.*?)^===== RESULT SUMMARY END =====\s*$",
    re.MULTILINE | re.DOTALL,
)
TRAIN_SUMMARY_BLOCK_RE = re.compile(
    r"^===== TRAIN RESULT SUMMARY START =====\s*\n(?P<body>.*?)^===== TRAIN RESULT SUMMARY END =====\s*$",
    re.MULTILINE | re.DOTALL,
)
PROXY_SUMMARY_BLOCK_RE = re.compile(
    r"^===== PROXY UTILITY SUMMARY START =====\s*\n(?P<body>.*?)^===== PROXY UTILITY SUMMARY END =====\s*$",
    re.MULTILINE | re.DOTALL,
)

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
        "train",
        "defense_baselines",
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
    return argv_str.strip().split()


def _infer_dataset_batch(tag: str, tokens: list[str]) -> tuple[str, str, str]:
    if not tokens:
        return "", "", ""
    if tag == "dager_dp" and len(tokens) >= 3:
        return tokens[1], tokens[2], " ".join(tokens[3:])
    if tag not in TAG_DATASET_BATCH or len(tokens) < 3:
        return "", "", " ".join(tokens[1:])
    return tokens[1], tokens[2], " ".join(tokens[3:])


def _parse_cli_flags(joined: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for m in re.finditer(r"--([\w_]+)\s+(\S+)", joined):
        out[m.group(1)] = m.group(2)
    for m in re.finditer(r"--([\w_]+)=(\S+)", joined):
        out[m.group(1)] = m.group(2)
    return out


def _parse_train_argv(argv_list: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    i = 0
    while i < len(argv_list):
        arg = argv_list[i]
        if arg.startswith("--") and i + 1 < len(argv_list) and not argv_list[i + 1].startswith("-"):
            out[arg[2:].replace("-", "_")] = argv_list[i + 1]
            i += 2
        else:
            i += 1
    return out


def _parse_summary_blocks(text: str, block_re: re.Pattern[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for match in block_re.finditer(text):
        body = match.group("body")
        summary: dict[str, str] = {}
        for raw_line in body.splitlines():
            line = raw_line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            summary[key] = value
        if summary:
            rows.append(summary)
    return rows


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
    for match in INPUT_TIME_RE.finditer(text):
        last = match
    if not last:
        return "", "", ""
    return last.group(1), last.group(2).strip(), last.group(3).strip()


def _meta_from_text(path: Path, text: str) -> dict[str, str]:
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
        meta["model_path_guess"] = flags.get("model_path", "")
    else:
        meta.update(
            {
                "run_ts": "",
                "tag": "",
                "shell_argv_raw": "",
                "dataset_guess": "",
                "batch_size_guess": "",
                "extra_cli": "",
                "defense_noise": "",
                "rank_tol": "",
                "model_path_guess": "",
            }
        )
    return meta


def parse_attack_dager(text: str, meta: dict) -> list[dict]:
    rows: list[dict] = []
    summaries = _parse_summary_blocks(text, RESULT_SUMMARY_BLOCK_RE)
    if summaries:
        for summary in summaries:
            row = {**meta, **summary}
            row["log_kind"] = "attack_dager"
            rows.append(row)
        return rows

    row = {**meta, "log_kind": "attack_dager"}
    rec_pairs = REC_LINE_RE.findall(text)
    if rec_pairs:
        maxb = [float(a) for a, _ in rec_pairs]
        tok = [float(b) for _, b in rec_pairs]
        row["n_rec_lines"] = str(len(rec_pairs))
        row["rec_maxb_token_mean"] = f"{statistics.mean(maxb):.6f}"
        row["rec_token_mean"] = f"{statistics.mean(tok):.6f}"
        row["rec_token_std"] = f"{statistics.stdev(tok):.6f}" if len(tok) > 1 else ""
    ranks = []
    for line in text.splitlines():
        m = RANK_LINE_RE.match(line.strip())
        if m:
            ranks.append((int(m.group(1)), int(m.group(2)), int(m.group(3))))
    if ranks:
        row["n_rank_lines"] = str(len(ranks))
        row["rank_b_mean"] = f"{statistics.mean(r[0] for r in ranks):.4f}"
    agg = _parse_aggregate_section(_final_aggregate_text(text))
    for key, value in agg.items():
        row[key] = f"{value:.6f}"
    inp, tin, ttot = _last_input_time(text)
    if inp:
        row["last_input_idx"] = inp
    if tin:
        row["last_input_time"] = tin
    if ttot:
        row["last_total_time"] = ttot
    return [row]


def parse_train(text: str, meta: dict) -> list[dict]:
    row = {**meta, "log_kind": "train"}
    summaries = _parse_summary_blocks(text, TRAIN_SUMMARY_BLOCK_RE)
    if summaries:
        row.update(summaries[-1])

    argv_list = _parse_terminal_argv(text)
    if argv_list:
        flags = _parse_train_argv(argv_list)
        for key in (
            "dataset",
            "batch_size",
            "noise",
            "num_epochs",
            "model_path",
            "train_method",
            "lora_r",
            "defense",
            "defense_noise",
        ):
            if key in flags:
                row[f"train_{key}"] = flags[key]

    evals = []
    for match in METRIC_EVAL_RE.finditer(text):
        try:
            evals.append(ast.literal_eval(match.group(1)))
        except (SyntaxError, ValueError):
            pass
    if evals:
        last = evals[-1]
        row["metric_eval_raw"] = repr(last)
        for key, value in last.items():
            row.setdefault(f"eval_{key}", str(value))

    trains = []
    for match in METRIC_TRAIN_RE.finditer(text):
        try:
            trains.append(ast.literal_eval(match.group(1)))
        except (SyntaxError, ValueError):
            pass
    if trains:
        row["metric_train_last_raw"] = repr(trains[-1])

    losses = [float(value) for value in LOSS_TRAIN_RE.findall(text)]
    if losses:
        row["loss_train_last"] = f"{losses[-1]:.6f}"
        row.setdefault("final_train_loss", f"{losses[-1]:.6f}")

    return [row]


def parse_proxy_utility(text: str, meta: dict) -> list[dict]:
    row = {**meta, "log_kind": "proxy_utility"}
    summaries = _parse_summary_blocks(text, PROXY_SUMMARY_BLOCK_RE)
    if summaries:
        row.update(summaries[-1])
    return [row]


def parse_feasibility(text: str, meta: dict) -> list[dict]:
    row = {**meta, "log_kind": "feasibility"}
    s_acc = [float(x) for x in SAMPLE_ACC_RE.findall(text)]
    set_acc = [float(x) for x in SAMPLE_SET_ACC_RE.findall(text)]
    row["n_sample_accuracy_lines"] = str(len(s_acc))
    row["sample_accuracy_mean"] = f"{statistics.mean(s_acc):.4f}" if s_acc else ""
    row["sample_accuracy_last"] = f"{s_acc[-1]:.4f}" if s_acc else ""
    row["n_sample_set_accuracy_lines"] = str(len(set_acc))
    row["sample_set_accuracy_last"] = f"{set_acc[-1]:.6f}" if set_acc else ""
    return [row]


def classify_and_parse(path: Path, text: str) -> list[dict]:
    meta = _meta_from_text(path, text)

    if _parse_summary_blocks(text, PROXY_SUMMARY_BLOCK_RE):
        return parse_proxy_utility(text, meta)
    if _parse_summary_blocks(text, TRAIN_SUMMARY_BLOCK_RE):
        return parse_train(text, meta)
    if _parse_summary_blocks(text, RESULT_SUMMARY_BLOCK_RE):
        return parse_attack_dager(text, meta)
    if SAMPLE_ACC_RE.search(text) is not None and AGG_HEADER not in text:
        return parse_feasibility(text, meta)
    if AGG_HEADER in text or re.search(r"^rouge1\s+\|", text, re.MULTILINE):
        return parse_attack_dager(text, meta)
    if "===== terminal log started" in text or METRIC_EVAL_RE.search(text) or METRIC_TRAIN_RE.search(text):
        return parse_train(text, meta)
    return [{**meta, "log_kind": "unknown"}]


def _all_keys(rows: list[dict]) -> list[str]:
    keys: set[str] = set()
    for row in rows:
        keys.update(row.keys())
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
        "batch_size",
        "train_method",
        "num_epochs",
        "seed",
        "defense",
        "defense_param_name",
        "defense_param_value",
        "output_dir",
        "steps_completed",
        "n_inputs_requested",
        "n_inputs_completed",
        "last_rec_status",
        "rec_l1_mean",
        "rec_l1_maxb_mean",
        "rec_l2_mean",
        "rec_token_mean",
        "rec_maxb_token_mean",
        "agg_rouge1_fm",
        "agg_rouge2_fm",
        "agg_r1fm_r2fm",
        "last_total_time",
        "final_train_loss",
        "eval_accuracy",
        "eval_macro_f1",
        "eval_loss",
        "total_train_time",
        "base_val_accuracy",
        "base_val_macro_f1",
        "base_val_loss",
        "grad_cosine_mean",
        "norm_retention_mean",
        "delta_train_loss_mean",
        "delta_val_loss_mean",
        "delta_val_accuracy_mean",
        "delta_val_macro_f1_mean",
        "step_runtime_mean",
        "error_type",
        "error_message",
    ]
    rest = sorted(key for key in keys if key not in preferred)
    return [key for key in preferred if key in keys] + rest


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = _all_keys(rows)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_markdown(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = _all_keys(rows)

    def esc(cell: str) -> str:
        return cell.replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(keys) + " |",
        "| " + " | ".join("---" for _ in keys) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(esc(str(row.get(key, ""))) for key in keys) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_paths(inputs: list[str]) -> list[Path]:
    out: list[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            out.extend(sorted(path.glob("*.txt")))
        else:
            out.append(path)
    seen = set()
    uniq: list[Path] = []
    for path in out:
        resolved = path.resolve()
        if resolved not in seen and path.exists():
            seen.add(resolved)
            uniq.append(path)
    return uniq


def _to_float(value) -> float | None:
    if value in (None, "", "n/a"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _time_to_seconds(value: str | None) -> float | None:
    if not value or value == "n/a":
        return None
    parts = str(value).split(":")
    try:
        nums = [float(part) for part in parts]
    except ValueError:
        return None
    if len(nums) == 3:
        hours, minutes, seconds = nums
    elif len(nums) == 2:
        hours, minutes, seconds = 0.0, nums[0], nums[1]
    else:
        return None
    return hours * 3600.0 + minutes * 60.0 + seconds


def _seconds_to_hms(seconds: float | None) -> str:
    if seconds is None:
        return ""
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _stats(values: list[float]) -> tuple[str, str]:
    if not values:
        return "", ""
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean:.6f}", f"{std:.6f}"


def build_utility_results(rows: list[dict]) -> list[dict]:
    train_rows = [row for row in rows if row.get("log_kind") == "train"]
    grouped: dict[tuple[str, ...], list[dict]] = {}
    for row in train_rows:
        key = (
            row.get("dataset", ""),
            row.get("batch_size", row.get("train_batch_size", "")),
            row.get("defense", row.get("train_defense", "none")),
            row.get("defense_param_name", ""),
            row.get("defense_param_value", ""),
        )
        grouped.setdefault(key, []).append(row)

    out: list[dict] = []
    for key, items in sorted(grouped.items()):
        dataset, batch_size, defense, param_name, param_value = key
        row = {
            "log_kind": "utility_summary",
            "dataset": dataset,
            "batch_size": batch_size,
            "defense": defense,
            "defense_param_name": param_name,
            "defense_param_value": param_value,
            "n_runs": str(len(items)),
            "seeds": " ".join(sorted({item.get("seed", "") for item in items if item.get("seed", "")})),
        }

        statuses = [item.get("result_status", "ok") for item in items]
        if statuses and all(status == "ok" for status in statuses):
            row["result_status"] = "ok"
        elif statuses and all(status != "ok" for status in statuses):
            row["result_status"] = "failed"
        else:
            row["result_status"] = "mixed"

        for field in ("eval_accuracy", "eval_macro_f1", "eval_loss", "final_train_loss"):
            values = [_to_float(item.get(field)) for item in items]
            clean_values = [value for value in values if value is not None]
            mean, std = _stats(clean_values)
            row[field] = mean
            row[f"{field}_std"] = std

        time_values = [_time_to_seconds(item.get("total_train_time")) for item in items]
        clean_time_values = [value for value in time_values if value is not None]
        row["total_train_time_seconds"], row["total_train_time_seconds_std"] = _stats(clean_time_values)
        row["total_train_time"] = _seconds_to_hms(
            statistics.mean(clean_time_values) if clean_time_values else None
        )

        failed = [item for item in items if item.get("result_status") != "ok"]
        row["failed_runs"] = str(len(failed))
        if failed:
            row["error_types"] = "; ".join(
                sorted({item.get("error_type", "") for item in failed if item.get("error_type", "")})
            )
        out.append(row)
    return out


def build_attack_anchor_results(rows: list[dict]) -> list[dict]:
    attack_rows = [row for row in rows if row.get("log_kind") == "attack_dager"]
    grouped: dict[tuple[str, ...], list[dict]] = {}
    for row in attack_rows:
        key = (
            row.get("dataset", row.get("dataset_guess", "")),
            row.get("batch_size", row.get("batch_size_guess", "")),
            row.get("defense", "none"),
            row.get("defense_param_name", ""),
            row.get("defense_param_value", ""),
        )
        grouped.setdefault(key, []).append(row)

    out: list[dict] = []
    for key, items in sorted(grouped.items()):
        dataset, batch_size, defense, param_name, param_value = key
        row = {
            "dataset": dataset,
            "batch_size": batch_size,
            "defense": defense,
            "defense_param_name": param_name,
            "defense_param_value": param_value,
            "n_privacy_runs": str(len(items)),
        }
        for field in ("rec_token_mean", "agg_rouge1_fm", "agg_rouge2_fm", "agg_r1fm_r2fm"):
            values = [_to_float(item.get(field)) for item in items]
            clean_values = [value for value in values if value is not None]
            mean, std = _stats(clean_values)
            row[field] = mean
            row[f"{field}_std"] = std
        out.append(row)
    return out


def _mark_pareto(rows: list[dict]) -> None:
    numeric_rows = []
    for row in rows:
        privacy_score = _to_float(row.get("privacy_score"))
        utility_drop = _to_float(row.get("utility_drop"))
        if privacy_score is None or utility_drop is None:
            row["pareto_optimal"] = ""
            continue
        numeric_rows.append((row, privacy_score, utility_drop))

    for row, privacy_score, utility_drop in numeric_rows:
        dominated = False
        for other, other_privacy, other_drop in numeric_rows:
            if other is row:
                continue
            if other_privacy >= privacy_score and other_drop <= utility_drop:
                if other_privacy > privacy_score or other_drop < utility_drop:
                    dominated = True
                    break
        row["pareto_optimal"] = "true" if not dominated else "false"


def build_privacy_utility_tradeoff(rows: list[dict]) -> list[dict]:
    utility_rows = build_utility_results(rows)
    attack_rows = build_attack_anchor_results(rows)

    attack_index = {
        (
            row.get("dataset", ""),
            row.get("batch_size", ""),
            row.get("defense", ""),
            row.get("defense_param_value", ""),
        ): row
        for row in attack_rows
    }
    none_index = {
        (row.get("dataset", ""), row.get("batch_size", "")): row
        for row in utility_rows
        if row.get("defense") == "none"
    }

    out: list[dict] = []
    for utility in utility_rows:
        dataset = utility.get("dataset", "")
        batch_size = utility.get("batch_size", "")
        defense = utility.get("defense", "")
        param_value = utility.get("defense_param_value", "")

        row = dict(utility)
        attack = attack_index.get((dataset, batch_size, defense, param_value))
        if attack is not None:
            row["rec_token_mean"] = attack.get("rec_token_mean", "")
            row["agg_rouge1_fm"] = attack.get("agg_rouge1_fm", "")
            row["agg_rouge2_fm"] = attack.get("agg_rouge2_fm", "")
            row["agg_r1fm_r2fm"] = attack.get("agg_r1fm_r2fm", "")

        acc = _to_float(row.get("eval_accuracy"))
        none_acc = _to_float(none_index.get((dataset, batch_size), {}).get("eval_accuracy"))
        rec_token = _to_float(row.get("rec_token_mean"))
        row["utility_drop"] = f"{(none_acc - acc):.6f}" if acc is not None and none_acc is not None else ""
        row["privacy_score"] = f"{(1.0 - rec_token):.6f}" if rec_token is not None else ""
        out.append(row)

    _mark_pareto(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize attack/train/proxy logs into CSV tables."
    )
    parser.add_argument("paths", nargs="*", default=[], help="Log files or directories.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("experiments_summary.csv"),
        help="Output CSV path for raw rows.",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=None,
        help="Also write a Markdown pipe table for raw rows.",
    )
    parser.add_argument("--utility-output", type=Path, default=None)
    parser.add_argument("--utility-markdown", type=Path, default=None)
    parser.add_argument("--tradeoff-output", type=Path, default=None)
    parser.add_argument("--tradeoff-markdown", type=Path, default=None)
    args = parser.parse_args()

    if args.paths:
        paths = collect_paths([str(path) for path in args.paths])
    else:
        default_dir = Path("log/runs")
        paths = sorted(default_dir.glob("*.txt")) if default_dir.is_dir() else []

    rows: list[dict] = []
    for path in paths:
        try:
            text = _read_text(path)
        except OSError as exc:
            rows.append({"log_path": str(path.resolve()), "log_kind": "error", "parse_error": str(exc)})
            continue
        rows.extend(classify_and_parse(path, text))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output, rows)
    print(f"Wrote {len(rows)} row(s) to {args.output.resolve()}")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(args.markdown, rows)
        print(f"Wrote Markdown to {args.markdown.resolve()}")

    if args.utility_output or args.utility_markdown:
        utility_rows = build_utility_results(rows)
        if args.utility_output:
            args.utility_output.parent.mkdir(parents=True, exist_ok=True)
            write_csv(args.utility_output, utility_rows)
            print(f"Wrote utility CSV to {args.utility_output.resolve()}")
        if args.utility_markdown:
            args.utility_markdown.parent.mkdir(parents=True, exist_ok=True)
            write_markdown(args.utility_markdown, utility_rows)
            print(f"Wrote utility Markdown to {args.utility_markdown.resolve()}")

    if args.tradeoff_output or args.tradeoff_markdown:
        tradeoff_rows = build_privacy_utility_tradeoff(rows)
        if args.tradeoff_output:
            args.tradeoff_output.parent.mkdir(parents=True, exist_ok=True)
            write_csv(args.tradeoff_output, tradeoff_rows)
            print(f"Wrote tradeoff CSV to {args.tradeoff_output.resolve()}")
        if args.tradeoff_markdown:
            args.tradeoff_markdown.parent.mkdir(parents=True, exist_ok=True)
            write_markdown(args.tradeoff_markdown, tradeoff_rows)
            print(f"Wrote tradeoff Markdown to {args.tradeoff_markdown.resolve()}")


if __name__ == "__main__":
    main()
