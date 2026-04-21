# =============================================================================
# fraudctl - Fraud Detection API
# =============================================================================
# Multi-stage build with distroless runtime (~2 MB base)
# =============================================================================

# -----------------------------------------------------------------------------
# Build stage
# -----------------------------------------------------------------------------
FROM golang:1.26-alpine AS builder

RUN apk add --no-cache gcc musl-dev

WORKDIR /build

# Copy go mod files
COPY go.mod ./

# Copy source code
COPY . .

# Build the binary (statically linked, no C dependencies)
RUN CGO_ENABLED=0 GOOS=linux go build \
    -ldflags="-s -w" \
    -o fraudctl ./cmd/api

# -----------------------------------------------------------------------------
# Runtime stage (distroless - ~2 MB)
# -----------------------------------------------------------------------------
FROM gcr.io/distroless/static-debian12:nonroot

# Copy binary and resources
COPY --from=builder /build/fraudctl /fraudctl
COPY --from=builder /build/resources /resources

# Set resources path
ENV RESOURCES=/resources

# Expose API port
EXPOSE 9999

# Run the application
ENTRYPOINT ["/fraudctl"]