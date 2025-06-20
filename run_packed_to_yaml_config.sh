#!/usr/bin/env bash
# Generate InfScale YAML from placement.json using the new self-contained generator
#
# Usage:
#   ./run_packed_to_yaml.sh <placement.json> <model_type> [output.yaml]
#
# Example:
#   ./run_packed_to_yaml.sh placement.json llama

set -euo pipefail

PLACEMENT=${1:-placement.json}
MODEL=${2:-llama}
DEFAULT_OUT="infscale_config/$(basename "${PLACEMENT%.*}")_${MODEL}_packed.yaml"
OUT=${3:-$DEFAULT_OUT}

echo "Generating InfScale config from $PLACEMENT..."

python generate_config_from_packing_solution.py \
  --placement "$PLACEMENT" \
  --model "$MODEL" \
  --out "$OUT" \
  --dispatcher_device cuda \
  --max_inflight 8