# syntax=docker/dockerfile:1
# =============================================================================
# fraudctl - Fraud Detection API
# =============================================================================
# CGo + AVX2 (x86-64-v3) for SIMD-accelerated KNN search.
# IVF index with SoA block layout for cache-friendly AVX2 vectorized distance computation.
# =============================================================================

FROM golang:1.26-bookworm AS builder

# gcc required for CGo (AVX2 intrinsics)
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY go.mod go.sum ./
RUN go mod download

COPY . .

# Compile the API with CGo + AVX2 optimizations.
RUN CGO_ENABLED=1 GOOS=linux GOARCH=amd64 \
    CGO_CFLAGS="-march=x86-64-v3 -O3 -flto" \
    CGO_LDFLAGS="-march=x86-64-v3 -flto" \
    go build -ldflags="-s -w -extldflags=-Wl,--gc-sections" -o fraudctl ./cmd/api

# Compile the index-builder (pure Go, no CGo needed).
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -ldflags="-s -w" -o build-index ./cmd/build-index

# Pre-build the IVF index (runs k-means on the 3M reference vectors).
# nlist=1200: ~2.5k vectors per cluster.
# Format v3: SoA blocks of 8 vectors for AVX2 processing.
RUN ./build-index -resources ./resources -nlist 1200 -iterations 15

# ── Production stage ──────────────────────────────────────────────────────────
FROM gcr.io/distroless/base-debian12 AS production

COPY --from=builder /build/fraudctl /fraudctl
COPY --from=builder /build/resources /resources

EXPOSE 9999

ENTRYPOINT ["/fraudctl"]
