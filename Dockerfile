# syntax=docker/dockerfile:1
# =============================================================================
# fraudctl - Fraud Detection API
# =============================================================================
# Stage 1 (build): Go + CGo/AVX2 — compiles the API binary.
# Stage 2 (production): distroless — minimal runtime image.
# =============================================================================

# ── Stage 1: build Go binary ───────────────────────────────────────────────────
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

# Compile the index builder.
RUN CGO_ENABLED=1 GOOS=linux GOARCH=amd64 \
    CGO_CFLAGS="-march=x86-64-v3 -O3 -flto" \
    CGO_LDFLAGS="-march=x86-64-v3 -flto" \
    go build -ldflags="-s -w -extldflags=-Wl,--gc-sections" -o build-index ./cmd/build-index

# Build the IVF index from references.json.gz (nlist=1024, 20 iterations).
# This runs at image build time so startup only needs to mmap the file.
RUN ./build-index -resources ./resources -nlist 1024 -iterations 20

# ── Stage 2: production ────────────────────────────────────────────────────────
FROM gcr.io/distroless/base-debian12 AS production

COPY --from=builder /build/fraudctl  /fraudctl
COPY --from=builder /build/resources /resources

EXPOSE 9999

ENTRYPOINT ["/fraudctl"]
