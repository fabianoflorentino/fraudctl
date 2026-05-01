#!/bin/bash
# Run k6 load test against the fraudctl API
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

docker run --rm \
  -v "${PROJECT_DIR}/test-local:/test" \
  -w /test \
  --network fraudctl-network \
  ghcr.io/grafana/k6:latest \
  run test.js