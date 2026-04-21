# syntax=docker/dockerfile:1
# =============================================================================
# fraudctl - Fraud Detection API
# =============================================================================
# Multi-stage build for optimized production image
# =============================================================================

# Build stage
FROM golang:1.26-alpine AS builder

WORKDIR /build

COPY go.mod ./
COPY . .

RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -o fraudctl ./cmd/api

# ── Production stage ─────────────────────────────────────────────────────────────
FROM alpine:3.23 AS production

RUN apk add --no-cache wget

COPY --from=builder /build/fraudctl /fraudctl
COPY --from=builder /build/resources /resources

ENV RESOURCES=/resources
EXPOSE 9999

ENTRYPOINT ["/fraudctl"]