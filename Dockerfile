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

RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 GOAMD64=v3 \
    go build -ldflags="-s -w" -o fraudctl ./cmd/api

# ── Stage 2: production ────────────────────────────────────────────────────────
FROM gcr.io/distroless/base-debian12 AS production

COPY --from=builder /build/fraudctl  /fraudctl
COPY resources /resources

ENV GOMAXPROCS=1
ENV GOGC=off
ENV GOMEMLIMIT=145MiB

EXPOSE 9999

ENTRYPOINT ["/fraudctl"]
