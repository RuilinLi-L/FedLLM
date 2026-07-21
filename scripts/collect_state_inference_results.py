"""Collect only ``state_inference_v1`` summaries into an independent CSV."""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


START = "===== STATE INFERENCE SUMMARY START ====="
END = "===== STATE INFERENCE SUMMARY END ====="
BLOCK = re.compile(re.escape(START) + r"\s*\n(?P<body>.*?)" + re.escape(END), re.DOTALL)


def parse(path: Path) -> list[dict[str, str]]:
    rows = []
    text = path.read_text(encoding="utf-8", errors="replace")
    for match in BLOCK.finditer(text):
        row = {"log_path": str(path.resolve())}
        for line in match.group("body").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                row[key.strip()] = value.strip()
        if row.get("protocol") == "state_inference_v1":
            rows.append(row)
    return rows


def parse_manifest(path: Path) -> dict[str, str]:
    fields = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key.strip()] = value.strip()
    return fields


def _split_csv(values: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in values.split(",") if part.strip())


def _read_exit_codes(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {str(row.get("seed", "")): str(row.get("exit_code", "")) for row in csv.DictReader(handle)}


def validate(root: Path, rows: list[dict[str, str]]) -> list[str]:
    errors = []
    manifest_path = root / "run_manifest.txt"
    if not manifest_path.exists():
        return ["missing run_manifest.txt"]
    manifest = parse_manifest(manifest_path)
    required = {"protocol", "eval_count", "m_values", "budgets", "seeds"}
    missing = sorted(required.difference(manifest))
    if missing:
        errors.append(f"run manifest is missing fields: {','.join(missing)}")
    if manifest.get("protocol") != "state_inference_v1":
        errors.append("run manifest protocol is not state_inference_v1")

    before, after = root / "legacy_inputs_before.sha256", root / "legacy_inputs_after.sha256"
    if not before.exists() or not after.exists():
        errors.append("missing pre/post legacy SHA-256 manifests")
    elif before.read_bytes() != after.read_bytes():
        errors.append("legacy SHA-256 manifests differ")

    expected_seeds = tuple(manifest.get("seeds", "").split())
    expected_m = _split_csv(manifest.get("m_values", ""))
    expected_budgets = _split_csv(manifest.get("budgets", ""))
    expected_estimator = {(m_value, budget) for m_value in expected_m for budget in expected_budgets}
    if not expected_seeds or not expected_estimator:
        errors.append("run manifest has no expected seeds or estimator conditions")

    exit_codes = _read_exit_codes(root / "exit_codes.csv")
    by_seed: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_seed.setdefault(row.get("rng_seed", ""), []).append(row)
        if row.get("result_status") != "ok":
            errors.append(f"seed {row.get('rng_seed', 'unknown')} has non-ok summary")
        if row.get("n_inputs_requested") != row.get("n_inputs_completed"):
            errors.append(f"seed {row.get('rng_seed', 'unknown')} has incomplete held-out updates")
        if row.get("n_inputs_requested") != manifest.get("eval_count"):
            errors.append(f"seed {row.get('rng_seed', 'unknown')} does not use manifest held-out count")

    for seed in expected_seeds:
        if exit_codes.get(seed) != "0":
            errors.append(f"seed {seed} lacks a successful exit code")
        seed_rows = by_seed.get(seed, [])
        variants = {row.get("state_attack_variant") for row in seed_rows}
        if {"method_only", "oracle", "state_estimator"}.difference(variants):
            errors.append(f"seed {seed} is missing a required attack variant")
        estimator_conditions = {
            (row.get("state_fit_updates"), row.get("state_budget"))
            for row in seed_rows
            if row.get("state_attack_variant") == "state_estimator"
        }
        if estimator_conditions != expected_estimator:
            errors.append(f"seed {seed} has incomplete estimator M/budget coverage")
        for variant in ("method_only", "oracle"):
            if sum(row.get("state_attack_variant") == variant for row in seed_rows) != 1:
                errors.append(f"seed {seed} must have exactly one {variant} diagnostic")
    return list(dict.fromkeys(errors))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    root = Path(args.run_root)
    logs = sorted((root / "logs").glob("*.txt"))
    rows = [row for path in logs for row in parse(path)]
    errors = validate(root, rows)
    for row in rows:
        row["acceptance_status"] = "accepted" if not errors else "incomplete_or_invalid"
    output = root / "state_inference_results.csv"
    fields = sorted({key for row in rows for key in row})
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    validation_path = root / "state_inference_validation.txt"
    validation_path.write_text("\n".join(errors) + ("\n" if errors else "accepted\n"), encoding="utf-8")
    print(f"[state-inference-collect] rows={len(rows)} accepted={not errors} output={output}")
    for error in errors:
        print(f"[state-inference-collect] invalid: {error}")
    return 0 if rows and not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
