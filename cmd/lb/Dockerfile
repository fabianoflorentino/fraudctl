# syntax=docker/dockerfile:1
# =============================================================================
# fraudctl-lb - SCM_RIGHTS fd-passing Load Balancer
# =============================================================================

FROM golang:1.26-bookworm AS builder

WORKDIR /build

COPY go.mod* ./
RUN go mod download

COPY . .

RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 GOAMD64=v3 \
    go build -ldflags="-s -w" -o fraudctl-lb .

FROM gcr.io/distroless/base-debian12

COPY --from=builder /build/fraudctl-lb /fraudctl-lb

ENV GOMAXPROCS=1
ENV GOGC=off
ENV GOMEMLIMIT=10MiB

EXPOSE 9999

ENTRYPOINT ["/fraudctl-lb"]
