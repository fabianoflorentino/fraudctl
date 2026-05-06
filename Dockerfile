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

# Pure-Go build — no CGo required.
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -ldflags="-s -w" -o fraudctl ./cmd/api

# Compile the index builder.
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -ldflags="-s -w" -o build-index ./cmd/build-index

# Build the IVF index from references.json.gz (nlist=4096, 60 iterations).
# This runs at image build time so startup only needs to load the file.
RUN ./build-index -resources ./resources -nlist 4096 -iterations 25

# ── Stage 2: production ────────────────────────────────────────────────────────
FROM gcr.io/distroless/base-debian12 AS production

COPY --from=builder /build/fraudctl  /fraudctl
COPY --from=builder /build/resources /resources

EXPOSE 9999

ENTRYPOINT ["/fraudctl"]
