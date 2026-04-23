#!/bin/bash
# Run k6 load test against the fraudctl API
cd /run/media/fabiano/Data/Projects/fraudctl && \
docker run --rm \
  -v $(pwd)/test-local:/test \
  -w /test \
  --network fraudctl-network \
  ghcr.io/grafana/k6:latest \
  run test.js