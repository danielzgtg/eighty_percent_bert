#!/bin/bash
set -e
REQS=requirements.txt
if [ $# -ne 0 ]; then
    REQS="$1"
    shift
fi
. venv/bin/activate
pip3 install -r "$REQS" --force "$@"
