# syntax=docker/dockerfile:1
# =============================================================================
# fraudctl - Fraud Detection API
# =============================================================================
# Multi-stage build with distroless runtime (~2 MB base)
# =============================================================================

# Build stage (CGO_ENABLED=0 means no C dependencies needed)
FROM golang:1.26-alpine AS builder

WORKDIR /build

COPY go.mod ./
COPY . .

RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -o fraudctl ./cmd/api

# Runtime stage (alpine with wget for healthcheck)
FROM alpine:3.23

RUN apk add --no-cache wget

COPY --from=builder /build/fraudctl /fraudctl
COPY --from=builder /build/resources /resources

ENV RESOURCES=/resources
EXPOSE 9999

ENTRYPOINT ["/fraudctl"]
