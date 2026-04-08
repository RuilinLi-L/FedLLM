#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_dager_auto_log.sh
source "${SCRIPT_DIR}/_dager_auto_log.sh"
dager_auto_log_enable "dager_dp" "$@"

array=( $@ )
len=${#array[@]}
last_args=${array[@]:3:$len}

python attack_new.py --dataset $1 --split val --n_inputs 100 --batch_size $2 --l1_filter all --l2_filter non-overlap --model_path $3 --device cuda --task seq_class --cache_dir ./models_cache --pad left $last_args
