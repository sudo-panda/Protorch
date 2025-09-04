#!/bin/bash
#flux: -N 1
#flux: -n 1
#flux: -g 1
#flux: -t 12h
#flux: --job-name=GraMI_train
#flux: --output=/p/vast1/kundu1/protorch/logs/flux-{{id.dec}}-out.log
#flux: --error=/p/vast1/kundu1/protorch/logs/flux-{{id.dec}}-{{id}}.log
set -eo pipefail

echo "START TIME: $(date)"

WORK_DIR="/usr/WS2/LExperts/mltraining/Protorch/python"
VENV="/usr/WS2/LExperts/mltraining/venvs/toss_4_x86_64_ib_cray"

PWD=`pwd`
if [[ "$PWD" == "$WORK_DIR" ]]; then
    >&2 echo "Not in the correct working directory, fixing ..."
    cd "$WORK_DIR"
fi

if [[ "$VIRTUAL_ENV" != "$VENV" ]]; then
    >&2 echo "Not in the correct virtual environment, fixing ..."
    source "$VENV/bin/activate"
fi


if [ -n "$1" ]; then
    CONFIG_NAME="$1"
else
    >&2 echo "No config file name provided"
    exit 1
fi

echo "Config File Name: $CONFIG_NAME"
python train.py --config $@


echo "END TIME: $(date)"

