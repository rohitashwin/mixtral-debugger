#!/usr/bin/env bash
set -euo pipefail

# 1. Install or upgrade the Hugging Face CLI
python3 -m pip install --no-input --break-system-packages --no-cache-dir --upgrade huggingface_hub

# 2. Read your token from the file and log in non-interactively
huggingface-cli login --token "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"
