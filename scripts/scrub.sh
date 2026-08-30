#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if grep -rinE 'whip|vikings|skol|bandit|strike|\bmike\b|scheme|C·|E·|·s|·w|port.wash|royal' \
    ml/ viewer/ examples/ README.md 2>/dev/null; then
  echo "SCRUB FAILED: proprietary term found"; exit 1
fi
echo "SCRUB OK"
