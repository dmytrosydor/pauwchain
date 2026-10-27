#!/usr/bin/env bash
# Пакує Python-частину (worker/, sim/) для запуску калібрування E1 на Kaggle T4. Див. docs/kaggle.md.
set -euo pipefail
cd "$(dirname "$0")/.."
out=build/kaggle
rm -rf "$out" && mkdir -p "$out"
zip -qr "$out/pauwchain-py.zip" worker sim -x '*/__pycache__/*' '*/tests/*'
echo "$out/pauwchain-py.zip ($(du -h "$out/pauwchain-py.zip" | cut -f1))"
