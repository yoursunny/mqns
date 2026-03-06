#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export NATS_URL=${NATS_URL:-nats://localhost:4222}
export MQNS_NATS_PREFIX=${MQNS_NATS_PREFIX:-mqns.classicbridge}
STREAM=${MQNS_NATS_STREAM:-MQNS_CLASSIC_BRIDGE}

info() {
  echo -ne "\e[0;32m"
  echo -n "$*"
  echo -e "\e[0m"
}

die() {
  echo -ne "\e[0;31m"
  echo -n "$*"
  echo -e "\e[0m"
  exit 1
}

if ! command -v nats &>/dev/null; then
  die NATS client not installed, please install from https://github.com/nats-io/natscli
fi
if ! nats account info &>/dev/null; then
  die NATS server not running, please start NATS server
fi

info Deleting existing NATS stream if exists
nats stream rm $STREAM -f || true

info Defining NATS stream
nats stream add $STREAM \
  --subjects "$MQNS_NATS_PREFIX.*.*" \
  --storage memory \
  --retention limits \
  --discard old \
  --max-msgs=-1 \
  --max-bytes=-1 \
  --max-age=1h \
  --defaults

MQNS_LOGLVL=DEBUG python extctrl_dp.py
