# syntax=docker/dockerfile:1
# =============================================================================
# fraudctl - Fraud Detection API
# =============================================================================
# Pure Go, no CGO.
# IVF index is pre-built at image build time so startup is fast and lean.
# =============================================================================

FROM golang:1.26-bookworm AS builder

WORKDIR /build

COPY go.mod go.sum ./
RUN go mod download

COPY . .

# Compile the API and the index-builder.
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -ldflags="-s -w" -o fraudctl ./cmd/api && \
    CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -ldflags="-s -w" -o build-index ./cmd/build-index

# Pre-build the IVF index (runs k-means on the 3M reference vectors).
# nlist=600: ~5k vectors per cluster; nprobe=1 scans ~5k vectors (~26us/query, 31% faster than nlist=300).
# With 16 CPUs at build time this takes ~50s.
RUN ./build-index -resources ./resources -nlist 600 -iterations 15

# ── Production stage ──────────────────────────────────────────────────────────
FROM gcr.io/distroless/static:nonroot AS production

COPY --from=builder /build/fraudctl /fraudctl
COPY --from=builder /build/resources /resources

# Resources are read via the -resources flag (default: /resources).
# To override at runtime: docker run ... /fraudctl -resources /other/path
EXPOSE 9999

ENTRYPOINT ["/fraudctl"]
