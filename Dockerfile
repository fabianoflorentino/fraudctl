# syntax=docker/dockerfile:1
# =============================================================================
# fraudctl - Fraud Detection API
# =============================================================================
# Multi-stage build with CGO for HNSW (hnswlib C++ library)
# =============================================================================

# Build stage — requires g++ for CGO/hnswlib
FROM golang:1.26-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY go.mod go.sum ./
RUN go mod download

COPY . .

# Build hnswlib static library in the module directory
RUN cd $(go list -m -f '{{.Dir}}' github.com/sunhailin-Leo/hnswlib-to-go) && \
    make portable

# Build the application with CGO enabled
RUN CGO_ENABLED=1 GOOS=linux GOARCH=amd64 \
    go build -ldflags="-s -w" -o fraudctl ./cmd/api

# ── Production stage ─────────────────────────────────────────────────────────────
FROM gcr.io/distroless/static:nonroot AS production

COPY --from=builder /build/fraudctl /fraudctl
COPY --from=builder /build/resources /resources

ENV RESOURCES=/resources
EXPOSE 9999

ENTRYPOINT ["/fraudctl"]
