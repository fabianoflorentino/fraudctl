# syntax=docker/dockerfile:1
# =============================================================================
# fraudctl - Fraud Detection API
# =============================================================================
# Stage 1 (build): pure-Go, no CGo.
# Stage 2 (production): distroless — minimal runtime image.
# =============================================================================

ARG NLIST=4096
ARG ITERATIONS=36

# ── Stage 1: build Go binary ───────────────────────────────────────────────────
FROM golang:1.26-bookworm AS builder

WORKDIR /build

COPY go.mod go.sum ./
RUN go mod download

COPY . .

RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 GOAMD64=v3 \
    go build -ldflags="-s -w" -o fraudctl ./cmd/api

# ── Stage 1.5: build IVF index ────────────────────────────────────────────
FROM builder AS index-builder

ARG NLIST
ARG ITERATIONS

RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 GOAMD64=v3 \
    go build -ldflags="-s -w" -o build-index ./cmd/build-index

RUN ./build-index -resources ./resources -nlist ${NLIST} -iterations ${ITERATIONS}

# ── Stage 2: production ────────────────────────────────────────────────────────
FROM gcr.io/distroless/static:nonroot AS production

COPY --from=index-builder /build/fraudctl  /fraudctl
COPY --from=index-builder /build/resources /resources

EXPOSE 9999

ENTRYPOINT ["/fraudctl"]
