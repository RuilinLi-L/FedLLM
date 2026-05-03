## [ERR-20260402-001] rg_exe_access_denied

**Logged**: 2026-04-02T00:00:00+08:00
**Priority**: low
**Status**: pending
**Area**: docs

### Summary
`rg.exe` in this environment returns "Access is denied", so text search should fall back to PowerShell-native commands.

### Error
```
Program 'rg.exe' failed to run: Access is denied
```

### Context
- Command attempted: `rg -n ...`
- Workspace: `D:\code\Projects\FedLLM`
- Impact: switched to `Select-String` and `Get-Content` for search

### Suggested Fix
Check execution permission or local installation path for `rg.exe` before relying on it in this workspace.

### Metadata
- Reproducible: yes
- Related Files: N/A

---
## [ERR-20260408-002] python_windowsapp_alias_unusable

**Logged**: 2026-04-08T11:05:00+08:00
**Priority**: medium
**Status**: pending
**Area**: config

### Summary
`python` resolves to the Windows Store app alias in this workspace, but the alias is not runnable, so local test scripts cannot be executed directly.

### Error
```
python --version
Exit code: 1

where.exe python
C:\Users\RuilinLi\AppData\Local\Microsoft\WindowsApps\python.exe

py -3 --version
The term 'py' is not recognized as the name of a cmdlet, function, script file, or operable program.
```

### Context
- Commands attempted: `python --version`, `python .\test_dager_defense.py`, `py -3 --version`
- Workspace: `D:\code\Projects\FedLLM`
- Impact: could inspect code and logs, but could not execute local validation scripts

### Suggested Fix
Install or expose a real Python interpreter on PATH, or activate the intended Conda environment before running repo scripts.

### Metadata
- Reproducible: yes
- Related Files: environment.yml, test_dager_defense.py, test_dager_defense_simple.py

---
## [ERR-20260410-003] local_validation_shells_unavailable

**Logged**: 2026-04-10T00:00:00+08:00
**Priority**: medium
**Status**: pending
**Area**: config

### Summary
This workspace session cannot run the usual local validation commands because `bash` returns an access-denied startup error and `python` is still mapped to an unusable Windows app alias.

### Error
```
bash -n scripts/defense_baselines.sh
CreateInstance / E_ACCESSDENIED

python --version
python.exe cannot run: the system cannot access this file
```

### Context
- Commands attempted: `bash -n scripts/defense_baselines.sh`, `python --version`, `py -3 --version`
- Workspace: `D:\code\Projects\FedLLM`
- Impact: implementation could proceed, but runtime syntax checks and parser smoke tests had to be replaced with static review

### Suggested Fix
Expose a usable Bash executable for shell-script validation and activate a real Python interpreter or Conda environment before running repo verification commands.

### Metadata
- Reproducible: yes
- Related Files: scripts/defense_baselines.sh, scripts/collect_experiment_logs.py, environment.yml
- See Also: ERR-20260408-002

---
## [ERR-20260416-001] git-bash-syntax-check

**Logged**: 2026-04-16T19:59:38+08:00
**Priority**: low
**Status**: pending
**Area**: config

### Summary
Git Bash syntax check failed in the Codex desktop Windows environment before the shell script could be validated.

### Error
`
C:\Program Files\Git\bin\..\usr\bin\bash.exe: *** fatal error - couldn't create signal pipe, Win32 error 5
`

### Context
- Command attempted: ash -n scripts/proxy_baselines.sh
- Environment: Codex desktop on Windows PowerShell
- The failure happened before the script body ran.

### Suggested Fix
Treat local bash syntax validation as unavailable in this desktop environment and validate on the Linux server when needed.

### Metadata
- Reproducible: yes
- Related Files: scripts/proxy_baselines.sh

---
## [ERR-20260416-002] codex-skill-install-home-write-denied

**Logged**: 2026-04-16T20:20:00+08:00
**Priority**: high
**Status**: pending
**Area**: config

### Summary
Installing a Codex skill to `C:\Users\RuilinLi\.codex\skills` is blocked in this sandboxed desktop environment, even when the skill source has already been cloned locally.

### Error
```
Copy-Item : Access to the path "C:\Users\RuilinLi\.codex\skills\taste-skill" was denied.

uv python list
error: Failed to initialize cache at `C:\Users\RuilinLi\AppData\Local\uv\cache`
Caused by: failed to create directory `C:\Users\RuilinLi\AppData\Local\uv\cache`: Access is denied. (os error 5)
```

### Context
- Operation attempted: install `skills/taste-skill` from `https://github.com/Leonxlnx/taste-skill` into Codex's user skill directory
- Workspace: `D:\code\Projects\FedLLM`
- Environment: Codex desktop on Windows PowerShell with `workspace-write` sandboxing
- Impact: repository vetting and local cloning succeeded, but the final install step into the user's Codex home could not complete

### Suggested Fix
When a task requires writing to `~/.codex/skills` or other user-home paths, expect the sandbox to block it. Prefer either a non-sandboxed install path, a user-run command outside Codex, or a workspace-local fallback package.

### Metadata
- Reproducible: yes
- Related Files: C:\Users\RuilinLi\.codex\skills\.system\skill-installer\scripts\install-skill-from-github.py
- See Also: ERR-20260408-002

---
## [ERR-20260426-004] train_py_annotation_runtime_compat

**Logged**: 2026-04-26T00:00:00+08:00
**Priority**: medium
**Status**: pending
**Area**: config

### Summary
`scripts/utility_baselines.sh` can fail only on the training stage because `train.py` evaluates `dict[str, float]` annotations at import time, while `scripts/proxy_utility.py` already defers annotation evaluation with `from __future__ import annotations`.

### Error
```
Traceback (most recent call last):
  File "train.py", line 142, in <module>
    def evaluate_model(model, eval_loader, device, dataset_name: str) -> dict[str, float]:
TypeError: 'type' object is not subscriptable
```

### Context
- Command attempted: `bash scripts/utility_baselines.sh ... --baseline_defense compression --baseline_param 16`
- Observed behavior: `proxy_none` and `proxy_compression_16` ran, but all `train_*` jobs failed before execution
- Relevant files: `train.py`, `scripts/proxy_utility.py`, `environment.yml`

### Suggested Fix
Add `from __future__ import annotations` to `train.py` so its type hints behave like the newer scripts, and verify the server is using the repo's intended Python environment.

### Metadata
- Reproducible: unknown
- Related Files: train.py, scripts/proxy_utility.py, environment.yml

---
