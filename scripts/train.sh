#!/usr/bin/env bash
set -euo pipefail
# One process per GPU. WORLD_SIZE must match the complete distributed job.
: "${NNODES:=1}"
: "${NPROC_PER_NODE:=8}"
: "${NODE_RANK:=0}"
: "${MASTER_ADDR:=127.0.0.1}"
: "${MASTER_PORT:=29500}"
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
exec torchrun --nnodes="$NNODES" --nproc_per_node="$NPROC_PER_NODE" \
  --node_rank="$NODE_RANK" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
  -m abot_er.cli train --world-size="$((NNODES * NPROC_PER_NODE))" "$@"
