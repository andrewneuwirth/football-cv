#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if grep -rinE --exclude=scrub.sh 'whip|vikings|skol|bandit|strike|\bmike\b|scheme|C·|E·|·s|·w|port.wash|royal' \
    ml/ viewer/ examples/ tests/ scripts/ README.md LICENSE pyproject.toml 2>/dev/null; then
  echo "SCRUB FAILED: proprietary term found"; exit 1
fi
echo "SCRUB OK"
