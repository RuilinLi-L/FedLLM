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
