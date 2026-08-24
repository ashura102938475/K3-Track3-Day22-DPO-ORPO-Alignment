#!/usr/bin/env bash
set -euo pipefail

uv run pref-lab validate data/sample_preferences.jsonl
uv run pref-lab evaluate --config configs/local.yaml
cat outputs/metrics.json
