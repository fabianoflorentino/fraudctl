# ============================================================================
# fraudctl — Makefile
# ============================================================================
# Fraud detection API using KNN vector search
# ============================================================================

.PHONY: help build run test test-short test-race coverage fmt vet lint tidy clean bench bench-knn bench-vectorizer all docker-build docker-up docker-down

# ============================================================================
# Variables
# ============================================================================

BINARY        := fraudctl
MODULE        := github.com/fabianoratm/fraudctl
VERSION       ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")
LDFLAGS       := -s -w -X main.version=$(VERSION)
BUILD_FLAGS   := CGO_ENABLED=0 GOARCH=amd64
GOTEST_FLAGS  := -v -race -count=1

# ============================================================================
# Output colors
# ============================================================================

RED    := \033[0;31m
GREEN  := \033[0;32m
YELLOW := \033[0;33m
BLUE   := \033[0;34m
NC     := \033[0m

# ============================================================================
# Default target
# ============================================================================

.DEFAULT_GOAL := help

##@ Help

help: ## Show this help message
	@echo ""
	@echo -e "$(BLUE)╔══════════════════════════════════════════════════════════════╗$(NC)"
	@echo -e "$(BLUE)║            fraudctl — Available Commands                ║$(NC)"
	@echo -e "$(BLUE)╚══════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf ""} /^[a-zA-Z_-]+:.*?##/ { printf "  $(GREEN)%-18s$(NC) %s\n", $$1, $$2 } /^##@/ { printf "\n$(YELLOW)%s$(NC)\n", substr($$0, 5) } ' $(MAKEFILE_LIST)
	@echo ""

##@ Build & Run

build: ## Compile the binary to ./bin/fraudctl
	@echo -e "$(BLUE)🔨 Building $(BINARY)...$(NC)"
	@mkdir -p bin
	@$(BUILD_FLAGS) go build -trimpath -ldflags="$(LDFLAGS)" -o bin/$(BINARY) .
	@echo -e "$(GREEN)✓ Binary created at bin/$(BINARY)$(NC)"

run: ## Run the API server (requires resources/)
	@echo -e "$(BLUE)▶  Running $(BINARY)...$(NC)"
	@go run ./cmd/api/main.go

##@ Tests & Quality

test: ## Run all unit tests with race detector
	@echo -e "$(BLUE)🧪 Running tests...$(NC)"
	@go test $(GOTEST_FLAGS) ./...
	@echo -e "$(GREEN)✓ All tests passed!$(NC)"

test-short: ## Run tests without race detector (faster)
	@echo -e "$(BLUE)🧪 Running tests (fast mode)...$(NC)"
	@go test -v -count=1 ./...
	@echo -e "$(GREEN)✓ Tests completed!$(NC)"

test-race: ## Run tests with race detector
	@echo -e "$(BLUE)🧪 Running tests with race detector...$(NC)"
	@go test -race -v ./...
	@echo -e "$(GREEN)✓ Tests passed!$(NC)"

coverage: ## Run tests and generate HTML coverage report
	@echo -e "$(BLUE)📊 Generating coverage report...$(NC)"
	@go test -race -coverprofile=coverage.out ./...
	@go tool cover -html=coverage.out -o coverage.html
	@echo -e "$(GREEN)✓ Report generated: coverage.html$(NC)"

fmt: ## Format Go source files
	@echo -e "$(BLUE)✏️  Formatting code...$(NC)"
	@go fmt ./...
	@echo -e "$(GREEN)✓ Formatting done!$(NC)"

vet: ## Run go vet
	@echo -e "$(BLUE)🔍 Running go vet...$(NC)"
	@go vet ./...
	@echo -e "$(GREEN)✓ go vet passed!$(NC)"

lint: ## Run golangci-lint
	@echo -e "$(BLUE)🔍 Running golangci-lint...$(NC)"
	@command -v golangci-lint >/dev/null 2>&1 || { echo -e "$(YELLOW)⚠️  golangci-lint not installed, skipping$(NC)"; exit 0; }
	@golangci-lint run ./...
	@echo -e "$(GREEN)✓ Lint passed!$(NC)"

tidy: ## Tidy and verify Go modules
	@echo -e "$(BLUE)📦 Tidying modules...$(NC)"
	@go mod tidy
	@go mod verify
	@echo -e "$(GREEN)✓ Modules up to date!$(NC)"

all: fmt vet test build ## Format, vet, test and build

##@ Benchmarks

bench: ## Run all benchmarks
	@echo -e "$(BLUE)📈 Running benchmarks...$(NC)"
	@go test -bench=. -benchmem ./...

bench-knn: ## Run KNN benchmarks only
	@echo -e "$(BLUE)📈 Running KNN benchmarks...$(NC)"
	@go test -bench=KNN -benchmem ./internal/knn/...

bench-vectorizer: ## Run vectorizer benchmarks only
	@echo -e "$(BLUE)📈 Running vectorizer benchmarks...$(NC)"
	@go test -bench=Vectorize -benchmem ./internal/vectorizer/...

bench-dataset: ## Run dataset benchmarks only
	@echo -e "$(BLUE)📈 Running dataset benchmarks...$(NC)"
	@go test -bench=Dataset -benchmem ./internal/dataset/...

##@ Docker

docker-build: ## Build production Docker image
	@echo -e "$(BLUE)🔨 Building Docker image...$(NC)"
	@docker build -t fraudctl:latest .
	@echo -e "$(GREEN)✓ Image fraudctl:latest created!$(NC)"

docker-up: ## Start services with docker compose
	@echo -e "$(BLUE)🚀 Starting services...$(NC)"
	@docker compose up -d

docker-down: ## Stop docker compose services
	@echo -e "$(BLUE)🛑 Stopping services...$(NC)"
	@docker compose down

##@ Cleanup

clean: ## Remove build artifacts
	@echo -e "$(YELLOW)🧹 Removing artifacts...$(NC)"
	@rm -rf bin/ coverage.out coverage.html
	@echo -e "$(GREEN)✓ Clean done!$(NC)"

clean-cache: ## Clear Go build cache
	@echo -e "$(YELLOW)🧹 Clearing Go build cache...$(NC)"
	@go clean -cache
	@echo -e "$(GREEN)✓ Go build cache cleared!$(NC)"
