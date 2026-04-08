#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_dager_auto_log.sh
source "${SCRIPT_DIR}/_dager_auto_log.sh"
dager_auto_log_enable "dec_feasibility" "$@"

python attack_len_increment.py --model_path gpt2 --n_inputs 100 -b 4 --dataset glnmario/ECHR --rank_tol 1e-9 --label feasibility_decoders_$1 --parallel 1000 --l1_span_thresh 1e-4 --start_input 167 --end_input 241 --rank_cutoff $1 --task seq_class --split val --l1_filter maxB --l2_filter  non-overlap --cache_dir ./models_cache
