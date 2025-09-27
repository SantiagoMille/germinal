#!/usr/bin/env bash
set -euo pipefail

# Raise file descriptor limits if possible
if command -v prlimit >/dev/null 2>&1; then
  prlimit --pid $$ --nofile=65536:65536 || true
fi

# If no arguments provided, open a shell in the environment
if [[ $# -eq 0 ]]; then
  exec micromamba run -n germinal bash
fi

# Execute provided command via bash -lc inside the environment to avoid exec parsing issues
cmd="$*"
exec micromamba run -n germinal bash -lc "$cmd"


