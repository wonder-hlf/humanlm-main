#!/usr/bin/env bash
set -u

echo "=== Workspace ==="
pwd

echo
echo "=== GPU ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
else
  echo "nvidia-smi not found"
fi

echo
echo "=== Candidate model directories ==="
find . -maxdepth 5 -type f -name config.json \
  | grep -Ei 'qwen|humanlm' \
  | sed 's#/config.json$##' \
  | sort -u

echo
echo "=== Training repo ==="
if [ -d "./verl-recipe-humanlm/.git" ]; then
  echo "./verl-recipe-humanlm exists"
  git -C ./verl-recipe-humanlm rev-parse HEAD
else
  echo "./verl-recipe-humanlm is missing"
fi

echo
echo "=== HumanLM repo ==="
if [ -d "./humanlm-main/.git" ]; then
  echo "./humanlm-main exists"
  git -C ./humanlm-main status -sb
else
  echo "./humanlm-main is missing"
fi

echo
echo "=== Python packages ==="
python - <<'PY'
import importlib.util
import sys

print("python:", sys.executable)
for name in ["torch", "datasets", "pandas", "pyarrow", "transformers", "verl"]:
    print(f"{name}: {'installed' if importlib.util.find_spec(name) else 'missing'}")
PY

echo
echo "=== CPS raw data ==="
if [ -d "./humanlm-main/local_data/team_bundles_atc21s_full" ]; then
  find ./humanlm-main/local_data/team_bundles_atc21s_full -maxdepth 1 -name '*.json' | wc -l
else
  echo "./humanlm-main/local_data/team_bundles_atc21s_full is missing"
fi
