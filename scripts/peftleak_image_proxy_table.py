#!/usr/bin/env python3
"""Collect PEFTLeak image privacy logs into a compact CSV table."""

from __future__ import annotations

import argparse
import codecs
import csv
import re
from pathlib import Path


SUMMARY_START = "===== RESULT SUMMARY START ====="
SUMMARY_END = "===== RESULT SUMMARY END ====="

DEFAULT_COLUMNS = [
    "log_path",
    "result_status",
    "dataset",
    "synthetic_fallback",
    "attack_variant",
    "reproduction_level",
    "defense",
    "defense_param_name",
    "defense_param_value",
    "rng_seed",
    "sample_strategy",
    "split_seed",
    "peftleak_protocol",
    "attack_indices_hash",
    "public_indices_hash",
    "attack_index_count",
    "public_index_count",
    "mse",
    "psnr",
    "ssim",
    "lpips",
    "lpips_status",
    "patch_recovery_rate",
    "primary_metric_source",
    "vit_adapter_loss",
    "batch_top1_acc",
    "candidate_patch_count",
    "recovered_patch_count",
    "collision_patch_count",
    "unresolved_patch_count",
    "runtime",
]


def read_log_text(path: Path) -> str:
    data = path.read_bytes()
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return data.decode("utf-16", errors="ignore")
    sample = data[:200]
    if sample.count(b"\x00") > max(1, len(sample) // 10):
        return data.decode("utf-16", errors="ignore")
    return data.decode("utf-8", errors="ignore")


def parse_summary(path: Path) -> dict[str, str] | None:
    fields: dict[str, str] = {"log_path": str(path)}
    inside = False
    for line in read_log_text(path).splitlines():
        stripped = line.strip()
        if stripped == SUMMARY_START:
            inside = True
            continue
        if stripped == SUMMARY_END:
            inside = False
            continue
        if inside and "=" in stripped:
            key, value = stripped.split("=", 1)
            fields[key] = value
    if fields.get("attack") != "peftleak_image_repro":
        return None
    if fields.get("rng_seed") in {None, "", "n/a"}:
        match = re.search(r"_seed(\d+)", path.stem)
        if match:
            fields["rng_seed"] = match.group(1)
    return fields


def write_markdown(rows: list[dict[str, str]], columns: list[str], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("| " + " | ".join(columns) + " |\n")
        handle.write("| " + " | ".join("---" for _ in columns) + " |\n")
        for row in rows:
            handle.write("| " + " | ".join(str(row.get(col, "n/a")) for col in columns) + " |\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", default="./outputs/peftleak_image_cifar10/privacy")
    parser.add_argument("--output", default="./outputs/peftleak_image_cifar10/tables/image_privacy_proxy_utility.csv")
    parser.add_argument("--markdown-output", default=None)
    args = parser.parse_args(argv)

    log_dir = Path(args.log_dir)
    output = Path(args.output)
    rows = [
        parsed
        for path in sorted(log_dir.glob("*.log"))
        for parsed in [parse_summary(path)]
        if parsed is not None
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DEFAULT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "n/a") for col in DEFAULT_COLUMNS})

    if args.markdown_output:
        markdown_output = Path(args.markdown_output)
        markdown_output.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(rows, DEFAULT_COLUMNS, markdown_output)

    print(f"wrote {len(rows)} rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
