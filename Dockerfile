# syntax=docker/dockerfile:1
# =============================================================================
# fraudctl - Fraud Detection API
# =============================================================================
# Stage 1 (build): pure-Go, no CGo.
# Stage 2 (production): distroless — minimal runtime image.
# =============================================================================

# ── Stage 1: build Go binary ───────────────────────────────────────────────────
FROM golang:1.26-bookworm AS builder

WORKDIR /build

COPY go.mod go.sum ./
RUN go mod download

COPY . .

# Pure-Go build — no CGo required. GOAMD64=v3 enables AVX2/BMI2 in the compiler.
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 GOAMD64=v3 \
    go build -ldflags="-s -w" -o fraudctl ./cmd/api

# ── Stage 1.5: build IVF index (cached separately) ────────────────────
FROM builder AS index-builder
# Build the index builder binary.
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 GOAMD64=v3 \
    go build -ldflags="-s -w" -o build-index ./cmd/build-index

# Build the IVF index (nlist=4096, 32 iterations for quality).
# Docker caches this layer; re-run only when resources change.
RUN ./build-index -resources ./resources -nlist 4096 -iterations 32

# ── Stage 2: production ────────────────────────────────────────────────────────
FROM gcr.io/distroless/base-debian12 AS production

COPY --from=builder /build/fraudctl  /fraudctl
COPY --from=builder /build/resources /resources

EXPOSE 9999

ENTRYPOINT ["/fraudctl"]
