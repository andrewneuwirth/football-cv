#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if grep -rInE --exclude=scrub.sh 'whip|vikings|skol|bandit|strike|\bmike\b|scheme|C·|E·|·s|·w|port.wash|royal' \
    footballcv/ viewer/ examples/ tests/ scripts/ docs/ README.md LICENSE pyproject.toml; then
  echo "SCRUB FAILED: proprietary term found"; exit 1
fi
echo "SCRUB OK"
