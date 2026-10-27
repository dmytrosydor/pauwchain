#!/usr/bin/env bash
# Packs the Python part (worker/, sim/) to run the E1 calibration on a Kaggle T4. See docs/kaggle.md.
set -euo pipefail
cd "$(dirname "$0")/.."
out=build/kaggle
rm -rf "$out" && mkdir -p "$out"
zip -qr "$out/pauwchain-py.zip" worker sim -x '*/__pycache__/*' '*/tests/*'
echo "$out/pauwchain-py.zip ($(du -h "$out/pauwchain-py.zip" | cut -f1))"
