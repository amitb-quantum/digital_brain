#!/usr/bin/env bash
# Runs on the rented vast.ai host (or locally, to validate it before paying).
# Installs pinned Miniforge, creates the digital-brain env from environment.yml,
# then checks it with pip check and the test suite.
#
#   PREFIX=/opt/db/miniforge bash scripts/vast_remote_setup.sh
set -euo pipefail

PREFIX=${PREFIX:-/opt/db/miniforge}
MF_VER=26.5.3-0
MF_SHA256=14db468222ad564658656f769506056209b6dc375f5e7dfd31eb5ebbf08fa529
cd "$(dirname "$0")/.."

if [ ! -x "$PREFIX/bin/conda" ]; then
  installer=$(mktemp --suffix=.sh)
  curl -fsSL -o "$installer" \
    "https://github.com/conda-forge/miniforge/releases/download/$MF_VER/Miniforge3-$MF_VER-Linux-x86_64.sh"
  echo "$MF_SHA256  $installer" | sha256sum -c -
  bash "$installer" -b -p "$PREFIX"
  rm -f "$installer"
fi

"$PREFIX/bin/conda" env create -q -y -f environment.yml
run() { "$PREFIX/bin/conda" run -n digital-brain --no-capture-output "$@"; }
run pip check
run nvcc --version | tail -1
run python -m pytest -q -p no:logging
