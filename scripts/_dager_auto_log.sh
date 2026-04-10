# shellcheck shell=bash
# Source from scripts under scripts/*.sh — not meant to be executed directly.
#
# Env:
#   DAGER_NO_AUTO_LOG=1  — disable file logging (stdout/stderr behave as usual)
#   DAGER_LOG_DIR=path   — default: log/runs

dager_auto_log_enable() {
	local tag="$1"
	if [ -n "${DAGER_NO_AUTO_LOG:-}" ]; then
		return 0
	fi
	local stamp log_dir logfile
	stamp=$(date +%Y%m%d_%H%M%S)
	log_dir="${DAGER_LOG_DIR:-log/runs}"
	mkdir -p "$log_dir" || return 0
	tag=$(printf '%s' "$tag" | tr -c 'a-zA-Z0-9._-' '_')
	logfile="${log_dir}/${tag}_${stamp}.txt"
	echo "[dager] Terminal log: $logfile" >&2
	exec > >(tee -a "$logfile") 2>&1
	echo "===== run start $(date '+%Y-%m-%d %H:%M:%S') tag=${tag} argv: $* ====="
}
