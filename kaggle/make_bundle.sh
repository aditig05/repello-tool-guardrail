#!/usr/bin/env bash
# Bundle code + processed data for upload as a Kaggle Dataset (kaggle.com/datasets -> New Dataset).
set -euo pipefail
cd "$(dirname "$0")/.."
rm -f kaggle/guardrail-bundle.zip
zip -qr kaggle/guardrail-bundle.zip scripts data/processed requirements.txt -x '*/__pycache__/*'
ls -lh kaggle/guardrail-bundle.zip
