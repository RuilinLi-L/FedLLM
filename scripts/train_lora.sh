#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_dager_auto_log.sh
source "${SCRIPT_DIR}/_dager_auto_log.sh"
dager_auto_log_enable "train_lora" "$@"

array=( $@ )
len=${#array[@]}
last_args=${array[@]:2:$len}

python ./train.py --dataset rotten_tomatoes --batch_size 1 --num_epoch 1 --model_path meta-llama/Meta-Llama-3.1-8B --train_method lora --lora_r 256 --save_every $1
rsync -ar ./finetune/ "$(rsync-path /home/ivo_petrov)"/dager/finetune
