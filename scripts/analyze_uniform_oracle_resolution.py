#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
import re
import statistics
from dataclasses import dataclass
from pathlib import Path


DEFAULT_EXISTING = Path(
    "log/同预算 white-box baselines/new/"
    "adaptive_lrb_matrix_sst2_official_validation_20260718_114719"
)
DEFAULT_SWEEP = Path("log/runs/uniform_oracle_resolution_sst2_official_validation")
DEFAULT_OUTPUT = Path("analysis/uniform_oracle_resolution")
SEEDS = (101, 202, 303)
METRICS = {
    "candidate_recall": "rec_token_mean",
    "topb_recall": "rec_maxb_token_mean",
    "l1": "rec_l1_mean",
    "l2": "rec_l2_mean",
    "r1_r2": "agg_r1fm_r2fm",
}
CURVE = (
    "uniform_r0.5",
    "uniform_r0.65",
    "uniform_r0.75",
    "uniform_r0.9",
    "none_r1.0",
)
CONTROL = "uniform_pool_r0.5"
CONDITION_RATIO = {
    "uniform_r0.5": 0.5,
    "uniform_r0.65": 0.65,
    "uniform_r0.75": 0.75,
    "uniform_r0.9": 0.9,
    "none_r1.0": 1.0,
    CONTROL: 0.5,
}


@dataclass(frozen=True)
class Run:
    condition: str
    seed: int
    metrics: dict[str, float]
    log_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit and summarize the uniform-ratio oracle DAGER sweep."
    )
    parser.add_argument("--existing-r05", type=Path, default=DEFAULT_EXISTING)
    parser.add_argument("--sweep-root", type=Path, default=DEFAULT_SWEEP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260719)
    return parser.parse_args()


def condition_for(row: dict[str, str]) -> str | None:
    if row.get("adaptive_lrb_knowledge") != "oracle":
        return None
    value = row.get("defense_param_value", "")
    if value == "proj_uniform_pool@k=0.5":
        return CONTROL
    match = re.fullmatch(r"proj_uniform@k=(0\.5|0\.65|0\.75|0\.9)", value)
    if match:
        return f"uniform_r{match.group(1)}"
    if row.get("defense") == "none":
        return "none_r1.0"
    return None


def resolve_log(results_csv: Path, raw_path: str) -> Path:
    basename = Path(raw_path.strip()).name
    candidates = (
        results_csv.parent / "logs" / basename,
        results_csv.parent / basename,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ValueError(f"missing local log for {raw_path!r} beside {results_csv}")


def load_results(existing: Path, sweep: Path) -> dict[tuple[str, int], Run]:
    csv_files = [existing / "results.csv"]
    csv_files.extend(sorted((sweep / "formal").glob("**/results.csv")))
    if not csv_files[0].is_file():
        raise ValueError(f"missing audited r=0.5 results: {csv_files[0]}")

    runs: dict[tuple[str, int], Run] = {}
    for results_csv in csv_files:
        if not results_csv.is_file():
            continue
        with results_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                condition = condition_for(row)
                if condition is None:
                    continue
                seed = int(row["seed"])
                if seed not in SEEDS:
                    continue
                if row.get("result_status") != "ok":
                    raise ValueError(f"non-ok result for {condition}/seed{seed}: {results_csv}")
                if int(row.get("n_inputs_requested", 0)) != 100:
                    raise ValueError(f"requested inputs != 100 for {condition}/seed{seed}")
                if int(row.get("n_inputs_completed", 0)) != 100:
                    raise ValueError(f"completed inputs != 100 for {condition}/seed{seed}")
                key = (condition, seed)
                if key in runs:
                    raise ValueError(f"duplicate admitted result: {condition}/seed{seed}")
                metrics = {name: float(row[column]) for name, column in METRICS.items()}
                runs[key] = Run(
                    condition=condition,
                    seed=seed,
                    metrics=metrics,
                    log_path=resolve_log(results_csv, row["log_path"]),
                )

    required = set(CURVE + (CONTROL,))
    missing = [
        f"{condition}/seed{seed}"
        for condition in sorted(required)
        for seed in SEEDS
        if (condition, seed) not in runs
    ]
    if missing:
        raise ValueError("missing admitted runs: " + ", ".join(missing))
    return runs


def parse_input_r1_r2(log_path: Path) -> list[float]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(
        r"Running input #(\d+) of 100\..*?"
        r"\[Curr input metrics\]:.*?r1fm\+r2fm = ([0-9]+(?:\.[0-9]+)?)",
        text,
        flags=re.DOTALL,
    )
    values = {int(index): float(value) for index, value in matches}
    if sorted(values) != list(range(100)):
        raise ValueError(f"expected 100 unique current-input metrics in {log_path}")
    return [values[index] for index in range(100)]


def percentile(sorted_values: list[float], probability: float) -> float:
    index = probability * (len(sorted_values) - 1)
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = index - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def clustered_paired_bootstrap(
    low: dict[int, list[float]],
    high: dict[int, list[float]],
    *,
    reps: int,
    rng: random.Random,
) -> tuple[float, float, float]:
    differences = {
        seed: [h - l for l, h in zip(low[seed], high[seed])]
        for seed in SEEDS
    }
    observed = statistics.mean(value for seed in SEEDS for value in differences[seed])
    boot = []
    for _ in range(reps):
        sampled_seeds = [rng.choice(SEEDS) for _ in SEEDS]
        total = 0.0
        count = 0
        for seed in sampled_seeds:
            values = differences[seed]
            for _ in range(len(values)):
                total += values[rng.randrange(len(values))]
                count += 1
        boot.append(total / count)
    boot.sort()
    return observed, percentile(boot, 0.025), percentile(boot, 0.975)


def summarize(runs: dict[tuple[str, int], Run]) -> dict[str, dict[str, tuple[float, float]]]:
    output = {}
    for condition in CURVE + (CONTROL,):
        condition_metrics = {}
        for metric in METRICS:
            values = [runs[(condition, seed)].metrics[metric] for seed in SEEDS]
            condition_metrics[metric] = (statistics.mean(values), statistics.stdev(values))
        output[condition] = condition_metrics
    return output


def write_outputs(
    output_dir: Path,
    runs: dict[tuple[str, int], Run],
    summary: dict[str, dict[str, tuple[float, float]]],
    bootstrap: list[dict[str, float | str]],
    branch: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / "uniform_oracle_resolution_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["condition", "ratio", "n_seeds"]
        for metric in METRICS:
            fieldnames.extend((f"{metric}_mean", f"{metric}_sample_sd"))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for condition in CURVE + (CONTROL,):
            row: dict[str, float | str | int] = {
                "condition": condition,
                "ratio": CONDITION_RATIO[condition],
                "n_seeds": len(SEEDS),
            }
            for metric, (mean, sd) in summary[condition].items():
                row[f"{metric}_mean"] = f"{mean:.6f}"
                row[f"{metric}_sample_sd"] = f"{sd:.6f}"
            writer.writerow(row)

    bootstrap_csv = output_dir / "uniform_oracle_resolution_paired_bootstrap.csv"
    with bootstrap_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("comparison", "mean_delta_high_minus_low", "ci95_low", "ci95_high"),
        )
        writer.writeheader()
        writer.writerows(bootstrap)

    markdown = output_dir / "uniform_oracle_resolution_results.md"
    lines = [
        "# Uniform Projection-LRB Oracle Resolution Sweep",
        "",
        "All rows passed `result_status=ok`, 100/100 completion, unique-log, and three-seed admission checks.",
        "Sample SD is computed across seed-level summaries (`ddof=1`).",
        "",
        "| Condition | Candidate recall | Top-B | L1 | L2 | R1+R2 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for condition in CURVE + (CONTROL,):
        cells = []
        for metric in METRICS:
            mean, sd = summary[condition][metric]
            cells.append(f"`{mean:.6f}±{sd:.6f}`")
        lines.append(f"| {condition} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Paired Cluster Bootstrap",
            "",
            "Differences are `higher ratio - lower ratio` in current-input R1+R2. "
            "Bootstrap resamples seeds as clusters and inputs within each sampled seed.",
            "",
            "| Comparison | Mean delta | 95% CI |",
            "| --- | ---: | ---: |",
        ]
    )
    for row in bootstrap:
        lines.append(
            f"| {row['comparison']} | {row['mean_delta_high_minus_low']} | "
            f"[{row['ci95_low']}, {row['ci95_high']}] |"
        )
    lines.extend(
        [
            "",
            "## Pre-Specified Interpretation",
            "",
            f"`{branch}`",
            "",
        ]
    )
    if branch == "resolution_supported":
        lines.append(
            "Oracle sequence recovery increases monotonically with reconstruction ratio, and the "
            "overall r=0.5 to undefended paired CI is above zero. The manuscript may attribute a "
            "defense-aware component to reduced reconstruction resolution while retaining the stated limitations."
        )
    else:
        lines.append(
            "The pre-specified resolution criterion was not met. The manuscript must retain only the "
            "transformed-span mismatch explanation and must not infer irreversible information loss from standard DAGER failure."
        )
    lines.extend(["", "## Audited Logs", ""])
    for condition in CURVE + (CONTROL,):
        for seed in SEEDS:
            lines.append(f"- `{condition}` seed `{seed}`: `{runs[(condition, seed)].log_path}`")
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.bootstrap_reps < 1000:
        raise ValueError("--bootstrap-reps must be at least 1000")
    runs = load_results(args.existing_r05, args.sweep_root)
    summary = summarize(runs)
    input_metrics = {
        (condition, seed): parse_input_r1_r2(runs[(condition, seed)].log_path)
        for condition in CURVE
        for seed in SEEDS
    }

    rng = random.Random(args.bootstrap_seed)
    comparisons = list(zip(CURVE[:-1], CURVE[1:]))
    bootstrap = []
    intervals = {}
    for low_condition, high_condition in comparisons:
        observed, ci_low, ci_high = clustered_paired_bootstrap(
            {seed: input_metrics[(low_condition, seed)] for seed in SEEDS},
            {seed: input_metrics[(high_condition, seed)] for seed in SEEDS},
            reps=args.bootstrap_reps,
            rng=rng,
        )
        label = f"{low_condition} -> {high_condition}"
        intervals[label] = (observed, ci_low, ci_high)
        bootstrap.append(
            {
                "comparison": label,
                "mean_delta_high_minus_low": f"{observed:.6f}",
                "ci95_low": f"{ci_low:.6f}",
                "ci95_high": f"{ci_high:.6f}",
            }
        )

    curve_means = [summary[condition]["r1_r2"][0] for condition in CURVE]
    ordered_means = all(high >= low for low, high in zip(curve_means[:-1], curve_means[1:]))
    overall_low = {seed: input_metrics[(CURVE[0], seed)] for seed in SEEDS}
    overall_high = {seed: input_metrics[(CURVE[-1], seed)] for seed in SEEDS}
    observed, ci_low, ci_high = clustered_paired_bootstrap(
        overall_low,
        overall_high,
        reps=args.bootstrap_reps,
        rng=rng,
    )
    bootstrap.append(
        {
            "comparison": f"{CURVE[0]} -> {CURVE[-1]}",
            "mean_delta_high_minus_low": f"{observed:.6f}",
            "ci95_low": f"{ci_low:.6f}",
            "ci95_high": f"{ci_high:.6f}",
        }
    )
    branch = "resolution_supported" if ordered_means and ci_low > 0.0 else "span_mismatch_only"
    write_outputs(args.output_dir, runs, summary, bootstrap, branch)
    print(f"analysis_status=ok")
    print(f"interpretation_branch={branch}")
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()

