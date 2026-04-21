# syntax=docker/dockerfile:1
# =============================================================================
# fraudctl - Fraud Detection API
# =============================================================================
# Multi-stage build for optimized production image (distroless)
# =============================================================================

# Build stage
FROM golang:1.26-alpine AS builder

WORKDIR /build

COPY go.mod ./
COPY . .

RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -o fraudctl ./cmd/api

# ── Production stage ─────────────────────────────────────────────────────────────
FROM gcr.io/distroless/static:nonroot AS production

COPY --from=builder /build/fraudctl /fraudctl
COPY --from=builder /build/resources /resources

ENV RESOURCES=/resources
EXPOSE 9999

ENTRYPOINT ["/fraudctl"]