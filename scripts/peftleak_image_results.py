#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics


PRIVACY_START = "===== RESULT SUMMARY START ====="
PRIVACY_END = "===== RESULT SUMMARY END ====="
UTILITY_START = "===== TRAIN RESULT SUMMARY START ====="
UTILITY_END = "===== TRAIN RESULT SUMMARY END ====="

PRIVACY_PROTOCOL_FIELDS = (
    "dataset",
    "model",
    "batch_size",
    "attack_index_count",
    "img_list_path",
    "public_split",
)
UTILITY_PROTOCOL_FIELDS = (
    "dataset",
    "profile",
    "model_path",
    "pretrained_weights",
    "shared_scope",
    "local_scope",
    "utility_control",
    "adapter_bottleneck_dim",
    "batch_size",
    "eval_batch_size",
    "validation_size",
    "num_epochs",
    "split_seed",
    "lr_adapter",
    "lr_head",
    "weight_decay",
    "warmup_epochs",
    "amp",
)
UTILITY_ANCHOR_FIELDS = tuple(field for field in UTILITY_PROTOCOL_FIELDS if field != "utility_control")
PRIVACY_METRIC_FIELDS = ("patch_recovery_rate", "mse", "ssim", "lpips")
UTILITY_METRIC_FIELDS = (
    "eval_accuracy",
    "eval_macro_f1",
    "eval_loss",
    "validation_accuracy",
    "final_train_loss",
    "total_train_time_seconds",
    "mean_step_time_seconds",
)


def parse_last_summary(path: Path, start_marker: str, end_marker: str) -> dict[str, str] | None:
    text = path.read_text(encoding="utf-8", errors="ignore")
    current = None
    summaries = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line == start_marker:
            current = {}
            continue
        if line == end_marker:
            if current is not None:
                summaries.append(current)
            current = None
            continue
        if current is not None and "=" in line:
            key, value = line.split("=", 1)
            current[key] = value
    if not summaries:
        return None
    result = summaries[-1]
    result["log_path"] = str(path)
    return result


def load_rows(root: Path | None, start: str, end: str) -> list[dict[str, str]]:
    if root is None or not root.exists():
        return []
    rows = []
    for path in sorted(root.rglob("*.log")):
        parsed = parse_last_summary(path, start, end)
        if parsed is not None:
            rows.append(parsed)
    return rows


def as_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalized_value(value) -> str:
    text = "n/a" if value in {None, ""} else str(value)
    numeric = as_float(text)
    return f"{numeric:.12g}" if numeric is not None else text


def method_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        row.get("defense", ""),
        row.get("defense_param_name", "n/a") or "n/a",
        normalized_value(row.get("defense_param_value", "n/a")),
    )


def protocol_key(row: dict[str, str], fields: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(normalized_value(row.get(field, "n/a")) for field in fields)


def protocol_columns(row: dict[str, str], fields: tuple[str, ...], prefix: str) -> dict[str, str]:
    return {f"{prefix}_{field}": normalized_value(row.get(field, "n/a")) for field in fields}


def _metric_fingerprint(row: dict[str, str], fields: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(normalized_value(row.get(field, "n/a")) for field in fields)


def deduplicate_rows(
    rows: list[dict[str, str]],
    *,
    protocol_fields: tuple[str, ...],
    metric_fields: tuple[str, ...],
    label: str,
) -> list[dict[str, str]]:
    deduplicated: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        key = (*protocol_key(row, protocol_fields), *method_key(row), row.get("seed", ""))
        previous = deduplicated.get(key)
        if previous is not None:
            if _metric_fingerprint(previous, metric_fields) != _metric_fingerprint(row, metric_fields):
                raise ValueError(
                    f"Conflicting duplicate {label} runs for key={key}: "
                    f"{previous.get('log_path')} vs {row.get('log_path')}"
                )
            continue
        deduplicated[key] = row
    return list(deduplicated.values())


def stats(rows: list[dict], field: str) -> tuple[float | None, float | None]:
    values = [as_float(row.get(field)) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return None, None
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def aggregate_privacy(rows: list[dict[str, str]]) -> list[dict]:
    valid_rows = [
        row
        for row in rows
        if row.get("result_status") == "ok" and row.get("attack") == "peftleak_official_image"
    ]
    valid = deduplicate_rows(
        valid_rows,
        protocol_fields=PRIVACY_PROTOCOL_FIELDS,
        metric_fields=PRIVACY_METRIC_FIELDS,
        label="privacy",
    )
    grouped: dict[tuple[str, ...], list[dict]] = {}
    for row in valid:
        grouped.setdefault((*protocol_key(row, PRIVACY_PROTOCOL_FIELDS), *method_key(row)), []).append(row)

    output = []
    for key, items in sorted(grouped.items()):
        defense, param_name, param_value = key[-3:]
        result = {
            "defense": defense,
            "defense_param_name": param_name,
            "defense_param_value": param_value,
            "privacy_n_runs": len(items),
            "privacy_seeds": " ".join(sorted({item.get("seed", "") for item in items})),
            **protocol_columns(items[0], PRIVACY_PROTOCOL_FIELDS, "privacy"),
        }
        for field in PRIVACY_METRIC_FIELDS:
            result[field], result[f"{field}_std"] = stats(items, field)
        output.append(result)
    return output


def aggregate_utility(rows: list[dict[str, str]]) -> list[dict]:
    valid_rows = [
        row
        for row in rows
        if row.get("result_status") == "ok" and row.get("reportable") == "true"
    ]
    valid = deduplicate_rows(
        valid_rows,
        protocol_fields=UTILITY_PROTOCOL_FIELDS,
        metric_fields=UTILITY_METRIC_FIELDS,
        label="utility",
    )
    none_by_anchor_seed = {
        (*protocol_key(row, UTILITY_ANCHOR_FIELDS), row.get("seed", "")): as_float(row.get("eval_accuracy"))
        for row in valid
        if row.get("defense") == "none" and row.get("utility_control", "standard") == "standard"
    }
    grouped: dict[tuple[str, ...], list[dict]] = {}
    for original in valid:
        row = dict(original)
        none_accuracy = none_by_anchor_seed.get(
            (*protocol_key(row, UTILITY_ANCHOR_FIELDS), row.get("seed", ""))
        )
        accuracy = as_float(row.get("eval_accuracy"))
        row["utility_drop"] = (
            None if none_accuracy is None or accuracy is None else none_accuracy - accuracy
        )
        grouped.setdefault((*protocol_key(row, UTILITY_PROTOCOL_FIELDS), *method_key(row)), []).append(row)

    output = []
    for key, items in sorted(grouped.items()):
        defense, param_name, param_value = key[-3:]
        result = {
            "defense": defense,
            "defense_param_name": param_name,
            "defense_param_value": param_value,
            "utility_n_runs": len(items),
            "utility_seeds": " ".join(sorted({item.get("seed", "") for item in items})),
            **protocol_columns(items[0], UTILITY_PROTOCOL_FIELDS, "utility"),
        }
        for field in (*UTILITY_METRIC_FIELDS, "utility_drop"):
            result[field], result[f"{field}_std"] = stats(items, field)
        output.append(result)
    return output


def seed_set(raw: str) -> set[str]:
    return {value for value in str(raw).split() if value}


def validate_expected_seeds(rows: list[dict], *, seed_field: str, expected: set[str], label: str) -> None:
    if not rows:
        raise ValueError(f"No reportable {label} rows were found.")
    for row in rows:
        actual = seed_set(row.get(seed_field, ""))
        if actual != expected:
            raise ValueError(
                f"{label} seed mismatch for {method_key(row)}: expected {sorted(expected)}, got {sorted(actual)}"
            )


def _aggregate_protocol_signature(row: dict, fields: tuple[str, ...], prefix: str) -> tuple[str, ...]:
    return tuple(str(row.get(f"{prefix}_{field}", "n/a")) for field in fields)


def build_cross_protocol_comparison(privacy: list[dict], utility: list[dict]) -> list[dict]:
    standard_utility = [row for row in utility if row.get("utility_utility_control") == "standard"]
    if not privacy or not standard_utility:
        return []

    privacy_protocols = {
        _aggregate_protocol_signature(row, PRIVACY_PROTOCOL_FIELDS, "privacy") for row in privacy
    }
    utility_protocols = {
        _aggregate_protocol_signature(row, UTILITY_PROTOCOL_FIELDS, "utility") for row in standard_utility
    }
    if len(privacy_protocols) != 1:
        raise ValueError("Cross-protocol comparison requires exactly one privacy protocol signature.")
    if len(utility_protocols) != 1:
        raise ValueError("Cross-protocol comparison requires exactly one standard utility protocol signature.")

    privacy_index = {method_key(row): row for row in privacy}
    utility_index = {method_key(row): row for row in standard_utility}
    output = []
    for key in sorted(set(privacy_index) & set(utility_index)):
        privacy_row = privacy_index[key]
        utility_row = utility_index[key]
        privacy_seeds = seed_set(privacy_row.get("privacy_seeds", ""))
        utility_seeds = seed_set(utility_row.get("utility_seeds", ""))
        if privacy_seeds != utility_seeds:
            raise ValueError(
                f"Cross-protocol seed mismatch for {key}: "
                f"privacy={sorted(privacy_seeds)}, utility={sorted(utility_seeds)}"
            )
        merged = {
            "comparison_scope": "cross_protocol_supplementary",
            "defense": key[0],
            "defense_param_name": key[1],
            "defense_param_value": key[2],
        }
        merged.update(privacy_row)
        merged.update(utility_row)
        output.append(merged)
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not fieldnames:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict]) -> None:
    columns = [
        "comparison_scope",
        "defense",
        "defense_param_value",
        "privacy_n_runs",
        "privacy_seeds",
        "patch_recovery_rate",
        "ssim",
        "lpips",
        "utility_n_runs",
        "utility_seeds",
        "eval_accuracy",
        "eval_accuracy_std",
        "utility_drop",
        "utility_drop_std",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("| " + " | ".join(columns) + " |\n")
        handle.write("| " + " | ".join("---" for _ in columns) + " |\n")
        for row in rows:
            handle.write("| " + " | ".join(str(row.get(column, "n/a")) for column in columns) + " |\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--privacy-log-dir", default=None)
    parser.add_argument("--utility-log-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-seeds", nargs="*", default=None)
    args = parser.parse_args(argv)

    privacy_rows = load_rows(
        Path(args.privacy_log_dir) if args.privacy_log_dir else None,
        PRIVACY_START,
        PRIVACY_END,
    )
    utility_rows = load_rows(Path(args.utility_log_dir), UTILITY_START, UTILITY_END)
    privacy = aggregate_privacy(privacy_rows)
    utility = aggregate_utility(utility_rows)
    expected = set(args.expected_seeds or [])
    if expected:
        validate_expected_seeds(utility, seed_field="utility_seeds", expected=expected, label="utility")
        if args.privacy_log_dir:
            validate_expected_seeds(privacy, seed_field="privacy_seeds", expected=expected, label="privacy")
    comparison = build_cross_protocol_comparison(privacy, utility)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "image_privacy.csv", privacy)
    write_csv(output_dir / "image_utility.csv", utility)
    write_csv(output_dir / "image_cross_protocol_comparison.csv", comparison)
    write_markdown(output_dir / "image_cross_protocol_comparison.md", comparison)
    print(
        f"wrote privacy={len(privacy)} utility={len(utility)} "
        f"cross_protocol={len(comparison)} rows to {output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
