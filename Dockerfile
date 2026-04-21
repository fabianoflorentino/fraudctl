# =============================================================================
# fraudctl - Fraud Detection API
# =============================================================================
# Multi-stage build for minimal runtime image
# =============================================================================

# -----------------------------------------------------------------------------
# Build stage
# -----------------------------------------------------------------------------
FROM golang:1.26-alpine AS builder

RUN apk add --no-cache gcc musl-dev

WORKDIR /build

# Copy go mod files (only go.mod - no external dependencies)
COPY go.mod ./

# Copy source code
COPY . .

# Build the binary
RUN CGO_ENABLED=0 GOOS=linux go build \
    -ldflags="-s -w" \
    -o fraudctl ./cmd/api

# -----------------------------------------------------------------------------
# Runtime stage
# -----------------------------------------------------------------------------
FROM alpine:3.19

RUN apk add --no-cache ca-certificates wget

WORKDIR /home/appuser

# Copy binary
COPY --from=builder /build/fraudctl .

# Copy resources (required for operation)
COPY --from=builder /build/resources ./resources

# Expose API port
EXPOSE 9999

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:9999/ready || exit 1

# Run the application
CMD ["./fraudctl"]