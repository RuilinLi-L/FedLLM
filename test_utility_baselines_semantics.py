#!/usr/bin/env python3
from __future__ import annotations

import os
import shlex
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def _candidate_bash_paths() -> list[str]:
    candidates = [
        os.environ.get("BASH"),
        shutil.which("bash"),
        shutil.which("bash.exe"),
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files\Git\usr\bin\bash.exe",
    ]
    out = []
    seen = set()
    for candidate in candidates:
        if not candidate:
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        if Path(candidate).exists():
            out.append(candidate)
    return out


def _to_bash_path(path: Path, bash_path: str) -> str:
    resolved = path.resolve()
    if os.name != "nt":
        return resolved.as_posix()

    bash_lower = bash_path.lower()
    if "system32\\bash.exe" in bash_lower:
        drive = resolved.drive.rstrip(":").lower()
        tail = resolved.as_posix().split(":/", 1)[1]
        return f"/mnt/{drive}/{tail}"
    return resolved.as_posix()


def _resolve_working_bash() -> str | None:
    for candidate in _candidate_bash_paths():
        try:
            proc = subprocess.run(
                [candidate, "-lc", "echo codex-bash-ok"],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if proc.returncode == 0 and "codex-bash-ok" in proc.stdout:
            return candidate
    return None


WORKING_BASH = _resolve_working_bash()


def _make_executable(path: Path) -> None:
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _write_fake_python3(repo_root: Path) -> Path:
    fakebin = repo_root / "fakebin"
    fakebin.mkdir(parents=True, exist_ok=True)
    shim_path = fakebin / "python3"
    shim_path.write_text(
        "#!/usr/bin/env sh\n"
        "set -eu\n"
        "record_file=\"$(CDPATH= cd -- \"$(dirname \"$0\")/..\" && pwd)/python3_calls.tsv\"\n"
        "script=\"${1:-}\"\n"
        "if [ \"$#\" -gt 0 ]; then\n"
        "  shift\n"
        "fi\n"
        "{\n"
        "  printf '%s' \"$script\"\n"
        "  for arg in \"$@\"; do\n"
        "    printf '\\t%s' \"$arg\"\n"
        "  done\n"
        "  printf '\\n'\n"
        "} >> \"$record_file\"\n"
        "prev=\"\"\n"
        "for arg in \"$@\"; do\n"
        "  case \"$prev\" in\n"
        "    --log_file|-o|--markdown|--utility-output|--utility-markdown|--tradeoff-output|--tradeoff-markdown)\n"
        "      mkdir -p \"$(dirname \"$arg\")\"\n"
        "      : > \"$arg\"\n"
        "      ;;\n"
        "    --output_dir)\n"
        "      mkdir -p \"$arg\"\n"
        "      ;;\n"
        "  esac\n"
        "  prev=\"$arg\"\n"
        "done\n"
        "exit 0\n",
        encoding="utf-8",
        newline="\n",
    )
    _make_executable(shim_path)
    return shim_path


def _prepare_fake_repo(repo_root: Path) -> None:
    (repo_root / "scripts").mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "scripts" / "utility_baselines.sh", repo_root / "scripts" / "utility_baselines.sh")
    _make_executable(repo_root / "scripts" / "utility_baselines.sh")

    anchor_dir = repo_root / "models" / "gpt2-ft-rt"
    anchor_dir.mkdir(parents=True, exist_ok=True)
    (anchor_dir / "config.json").write_text("{}", encoding="utf-8")
    (anchor_dir / "model.safetensors").write_text("", encoding="utf-8")

    _write_fake_python3(repo_root)


def _parse_calls(record_file: Path) -> list[list[str]]:
    if not record_file.exists():
        return []
    calls = []
    for line in record_file.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        calls.append(line.split("\t"))
    return calls


def _arg_value(call: list[str], flag: str) -> str | None:
    try:
        idx = call.index(flag)
    except ValueError:
        return None
    if idx + 1 >= len(call):
        return None
    return call[idx + 1]


def _call_label(call: list[str]) -> str:
    log_path = _arg_value(call, "--log_file")
    assert_true(log_path is not None, f"missing --log_file in call: {call}")
    return Path(log_path).stem


def _run_utility_script(extra_args: list[str]) -> dict:
    assert_true(WORKING_BASH is not None, "functional bash is required for this test")

    tmpdir = tempfile.TemporaryDirectory()
    repo_root = Path(tmpdir.name)
    _prepare_fake_repo(repo_root)

    repo_root_bash = _to_bash_path(repo_root, WORKING_BASH)
    base_args = [
        "./scripts/utility_baselines.sh",
        "sst2",
        "2",
        "gpt2",
        "1",
        "--anchor_dir",
        "./models/gpt2-ft-rt",
    ]
    cmd = (
        f"cd {shlex.quote(repo_root_bash)} && "
        f"PATH=./fakebin:$PATH "
        f"{' '.join(shlex.quote(arg) for arg in (base_args + extra_args))}"
    )
    proc = subprocess.run(
        [WORKING_BASH, "-lc", cmd],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return {
        "proc": proc,
        "repo_root": repo_root,
        "calls": _parse_calls(repo_root / "python3_calls.tsv"),
        "tmpdir": tmpdir,
    }


def _split_calls(calls: list[list[str]]) -> tuple[list[list[str]], list[list[str]], list[list[str]]]:
    proxy_calls = [call for call in calls if call and call[0] == "scripts/proxy_utility.py"]
    train_calls = [call for call in calls if call and call[0] == "train.py"]
    collect_calls = [call for call in calls if call and call[0] == "scripts/collect_experiment_logs.py"]
    return proxy_calls, train_calls, collect_calls


def _run_dir_names(repo_root: Path) -> list[str]:
    runs_root = repo_root / "log" / "runs"
    if not runs_root.exists():
        return []
    return sorted(path.name for path in runs_root.iterdir() if path.is_dir())


def test_default_invocation_schedules_full_operating_point_table():
    result = _run_utility_script([])
    proc = result["proc"]
    assert_true(proc.returncode == 0, proc.stderr or proc.stdout)

    proxy_calls, train_calls, collect_calls = _split_calls(result["calls"])
    assert_true(len(proxy_calls) == 8, "default run should schedule 8 proxy utility calls")
    assert_true(len(train_calls) == 24, "default run should schedule 8*3 training utility calls")
    assert_true(len(collect_calls) == 1, "default run should collect results once")

    proxy_labels = {_call_label(call) for call in proxy_calls}
    expected_proxy = {
        "proxy_none",
        "proxy_lrb_0.2",
        "proxy_topk_0.1",
        "proxy_compression_8",
        "proxy_noise_5e-4",
        "proxy_dpsgd_5e-4",
        "proxy_mixup_0.3",
        "proxy_soteria_30",
    }
    assert_true(proxy_labels == expected_proxy, f"unexpected proxy labels: {proxy_labels}")

    defenses = {_arg_value(call, "--defense") for call in proxy_calls + train_calls}
    assert_true(defenses == {"none", "lrb", "topk", "compression", "noise", "dpsgd", "mixup", "soteria"}, "default run should cover the full utility baseline table")

    run_dirs = _run_dir_names(result["repo_root"])
    assert_true(len(run_dirs) == 1, "default run should create one run directory")
    assert_true("_focus_" not in run_dirs[0], "default run directory should not include focus suffix")
    result["tmpdir"].cleanup()


def test_focus_none_runs_only_none_pipeline():
    result = _run_utility_script(["--baseline_defense", "none"])
    proc = result["proc"]
    assert_true(proc.returncode == 0, proc.stderr or proc.stdout)

    proxy_calls, train_calls, collect_calls = _split_calls(result["calls"])
    assert_true(len(proxy_calls) == 1, "focused none run should schedule one proxy call")
    assert_true(len(train_calls) == 3, "focused none run should schedule three train calls")
    assert_true(len(collect_calls) == 1, "focused none run should still collect results")

    proxy_labels = {_call_label(call) for call in proxy_calls}
    train_labels = {_call_label(call) for call in train_calls}
    assert_true(proxy_labels == {"proxy_none"}, f"unexpected proxy labels: {proxy_labels}")
    assert_true(
        train_labels == {"train_none_seed101", "train_none_seed202", "train_none_seed303"},
        f"unexpected train labels: {train_labels}",
    )

    run_dirs = _run_dir_names(result["repo_root"])
    assert_true(len(run_dirs) == 1 and "_focus_none_" in run_dirs[0], "focused none run dir should advertise the focus defense")
    result["tmpdir"].cleanup()


def test_focus_dpsgd_runs_none_plus_default_point():
    result = _run_utility_script(["--baseline_defense", "dpsgd"])
    proc = result["proc"]
    assert_true(proc.returncode == 0, proc.stderr or proc.stdout)

    proxy_calls, train_calls, _ = _split_calls(result["calls"])
    proxy_labels = {_call_label(call) for call in proxy_calls}
    train_labels = {_call_label(call) for call in train_calls}
    expected_train = {
        "train_none_seed101",
        "train_none_seed202",
        "train_none_seed303",
        "train_dpsgd_5e-4_seed101",
        "train_dpsgd_5e-4_seed202",
        "train_dpsgd_5e-4_seed303",
    }
    assert_true(proxy_labels == {"proxy_none", "proxy_dpsgd_5e-4"}, f"unexpected proxy labels: {proxy_labels}")
    assert_true(train_labels == expected_train, f"unexpected train labels: {train_labels}")

    defenses = {_arg_value(call, "--defense") for call in proxy_calls + train_calls}
    assert_true(defenses == {"none", "dpsgd"}, f"unexpected focused defenses: {defenses}")

    dpsgd_calls = [call for call in proxy_calls + train_calls if _arg_value(call, "--defense") == "dpsgd"]
    assert_true(all(_arg_value(call, "--defense_noise") == "5e-4" for call in dpsgd_calls), "focused dpsgd run should use the default utility operating point")

    run_dirs = _run_dir_names(result["repo_root"])
    assert_true(len(run_dirs) == 1 and "_focus_dpsgd_" in run_dirs[0], "focused dpsgd run dir should advertise the focus defense")
    result["tmpdir"].cleanup()


def test_focus_param_override_reaches_proxy_and_train():
    result = _run_utility_script(["--baseline_defense", "dpsgd", "--baseline_param", "1e-4"])
    proc = result["proc"]
    assert_true(proc.returncode == 0, proc.stderr or proc.stdout)

    proxy_calls, train_calls, _ = _split_calls(result["calls"])
    dpsgd_calls = [call for call in proxy_calls + train_calls if _arg_value(call, "--defense") == "dpsgd"]
    assert_true(dpsgd_calls, "focused override should schedule dpsgd calls")
    assert_true(all(_arg_value(call, "--defense_noise") == "1e-4" for call in dpsgd_calls), "focused override should replace the default operating point in both proxy and train calls")

    run_dirs = _run_dir_names(result["repo_root"])
    assert_true(len(run_dirs) == 1 and "_focus_dpsgd_1e-4_" in run_dirs[0], "focused override run dir should include the parameter slug")
    result["tmpdir"].cleanup()


def test_malformed_cli_combinations_exit_with_code_2():
    bad_cases = [
        ["--baseline_param", "1e-4"],
        ["--baseline_defense", "none", "--baseline_param", "1e-4"],
        ["--baseline_defense", "bogus"],
    ]
    for args in bad_cases:
        result = _run_utility_script(args)
        proc = result["proc"]
        assert_true(proc.returncode == 2, f"expected rc=2 for {args}, got {proc.returncode} ({proc.stderr or proc.stdout})")
        assert_true(not result["calls"], f"malformed args should fail before scheduling python calls: {args}")
        result["tmpdir"].cleanup()


def test_focused_lrb_sensitivity_adds_extra_point_without_sweep():
    result = _run_utility_script(["--baseline_defense", "lrb", "--include_sensitivity"])
    proc = result["proc"]
    assert_true(proc.returncode == 0, proc.stderr or proc.stdout)

    proxy_calls, train_calls, _ = _split_calls(result["calls"])
    proxy_labels = {_call_label(call) for call in proxy_calls}
    train_labels = {_call_label(call) for call in train_calls}
    assert_true(
        proxy_labels == {"proxy_none", "proxy_lrb_0.2", "proxy_lrb_0.35"},
        f"unexpected focused lrb proxy labels: {proxy_labels}",
    )
    assert_true(
        train_labels == {
            "train_none_seed101",
            "train_none_seed202",
            "train_none_seed303",
            "train_lrb_0.2_seed101",
            "train_lrb_0.2_seed202",
            "train_lrb_0.2_seed303",
            "train_lrb_0.35_seed101",
            "train_lrb_0.35_seed202",
            "train_lrb_0.35_seed303",
        },
        f"unexpected focused lrb train labels: {train_labels}",
    )
    result["tmpdir"].cleanup()


def main():
    if WORKING_BASH is None:
        print("Skipping utility baseline semantics tests: no functional bash executable available.")
        return 0

    tests = [
        test_default_invocation_schedules_full_operating_point_table,
        test_focus_none_runs_only_none_pipeline,
        test_focus_dpsgd_runs_none_plus_default_point,
        test_focus_param_override_reaches_proxy_and_train,
        test_malformed_cli_combinations_exit_with_code_2,
        test_focused_lrb_sensitivity_adds_extra_point_without_sweep,
    ]
    for test in tests:
        print(f"Running {test.__name__}...")
        test()
    print("All utility baseline semantic tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
