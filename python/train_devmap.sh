#!/bin/bash
#flux: -N 1
#flux: -n 1
#flux: -g 1
#flux: -t 6h
#flux: --job-name=devmap_train
#flux: --output=logs/flux-{{id.dec}}-out.log
#flux: --output=logs/flux-{{id.dec}}-err.log
set -eo pipefail

echo "START TIME: $(date)"

WORK_DIR="/usr/WS2/LExperts/mltraining/Protorch/python"
VENV="/usr/WS2/LExperts/mltraining/venvs/toss_4_x86_64_ib_cray"
CONFIG_NAME="$1"

PWD=`pwd`
if [[ "$PWD" == "$WORK_DIR" ]]; then
    echo "In the correct working directory, fixing ..."
    cd "$WORK_DIR"
fi

if [[ "$VIRTUAL_ENV" != "$VENV" ]]; then
    echo "Not in the correct virtual environment, fixing ..."
    source "$VENV/bin/activate"
fi

if [[ -z "$CONFIG_NAME" ]]; then
    echo "No config file name provided"
    exit 1
fi

echo "Config File Name: $CONFIG_NAME"

python train_devmap.py --config "$CONFIG_NAME"

echo "END TIME: $(date)"

