#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_dager_auto_log.sh
source "${SCRIPT_DIR}/_dager_auto_log.sh"
dager_auto_log_enable "fed_avg" "$@"

array=( $@ )
len=${#array[@]}
last_args=${array[@]:3:$len}

python attack.py --dataset rotten_tomatoes --split val --n_inputs 100 --batch_size 16 --l1_filter all --l2_filter non-overlap --model_path gpt2 --device cuda --task seq_class --algo fedavg --b_mini $2 --avg_epochs $1 --avg_lr $3 --cache_dir ./models_cache --rank_tol 5e-6 --l1_span_thresh 5e-3 --l2_span_thresh 5e-3  --attn_implementation eager --precision double $last_args
