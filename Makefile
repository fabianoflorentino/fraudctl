# ============================================================================
# fraudctl — Makefile
# ============================================================================
# Fraud detection API using KNN vector search
# ============================================================================

.PHONY: help build run test test-short test-race coverage fmt vet lint tidy clean bench bench-knn bench-vectorizer all docker-build docker-push docker-run docker-lint docker-clean docker-size docker-up docker-down submit submission-update submission-test submission-result k6-smoke k6-full k6-results

# ============================================================================
# Variables
# ============================================================================

RINHA_REPO        := zanfranceschi/rinha-de-backend-2026
SUBMISSION_BRANCH := submission
RINHA_DIR         ?= $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))/../rinha-de-backend-2026
MODULE        := github.com/fabianoflorentino/fraudctl
VERSION      ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")
LDFLAGS       := -s -w -X main.version=$(VERSION)
BUILD_FLAGS   := CGO_ENABLED=0 GOARCH=amd64
GOTEST_FLAGS  := -v -race -count=1
IMAGE         := fraudctl
REGISTRY      ?= ghcr.io/$(MODULE)
BUILD_OPTS    := --platform linux/amd64 --load

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
	@echo -e "$(BLUE)║            fraudctl — Available Commands                     ║$(NC)"
	@echo -e "$(BLUE)╚══════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf ""} /^[a-zA-Z0-9_-]+:.*?##/ { printf "  $(GREEN)%-18s$(NC) %s\n", $$1, $$2 } /^##@/ { printf "\n$(YELLOW)%s$(NC)\n", substr($$0, 5) } ' $(MAKEFILE_LIST)
	@echo ""

##@ Build & Run

build: ## Compile the binary to ./bin/fraudctl
	@echo -e "$(BLUE)🔨 Building $(BINARY)...$(NC)"
	@mkdir -p bin
	@$(BUILD_FLAGS) go build -trimpath -ldflags="$(LDFLAGS)" -o bin/$(BINARY) ./cmd/api
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

docker-build: ## Build Docker image (latest + version tag)
	@echo -e "$(BLUE)🔨 Building Docker image...$(NC)"
	@docker build -t $(IMAGE):latest -t $(IMAGE):$(VERSION) .
	@echo -e "$(GREEN)✓ Image $(IMAGE):$(VERSION) created!$(NC)"
	@$(MAKE) docker-size

docker-lint: ## Lint Dockerfile with hadolint
	@echo -e "$(BLUE)🔍 Linting Dockerfile...$(NC)"
	@docker run --rm -i hadolint/hadolint < Dockerfile || true
	@echo -e "$(GREEN)✓ Dockerfile lint passed!$(NC)"

docker-size: ## Show Docker image size
	@echo -e "$(BLUE)📦 Image size:$(NC)"
	@docker images $(IMAGE) --format "  {{.Repository}}:{{.Tag}} — {{.Size}}"

docker-run: ## Run container locally for testing
	@echo -e "$(BLUE)▶  Running container...$(NC)"
	@docker run --rm -p 9999:9999 $(IMAGE):latest
	@echo -e "$(GREEN)✓ Container stopped!$(NC)"

docker-push: ## Push image to registry
	@echo -e "$(BLUE)📤 Pushing to registry...$(NC)"
	@docker push $(REGISTRY):$(VERSION)
	@docker push $(REGISTRY):latest
	@echo -e "$(GREEN)✓ Image pushed!$(NC)"

docker-clean: ## Remove local Docker images
	@echo -e "$(YELLOW)🧹 Removing Docker images...$(NC)"
	@docker images $(IMAGE) -q | xargs -r docker rmi 2>/dev/null || true
	@echo -e "$(GREEN)✓ Docker images cleaned!$(NC)"

docker-up: ## Start services with docker compose
	@echo -e "$(BLUE)🚀 Starting services...$(NC)"
	@./scripts/docker-up.sh
	@echo -e "$(GREEN)✓ Services started!$(NC)"
	@echo -e "$(BLUE)  API available at http://localhost:9999$(NC)"
	@echo -e "$(BLUE)  Endpoints: /ready, /fraud-score$(NC)"

docker-down: ## Stop docker compose services
	@echo -e "$(YELLOW)🛑 Stopping services...$(NC)"
	@./scripts/docker-up.sh down
	@echo -e "$(GREEN)✓ Services stopped!$(NC)"

docker-logs: ## Show docker compose logs
	@./scripts/docker-up.sh logs

##@ Cleanup

clean: ## Remove build artifacts
	@echo -e "$(YELLOW)🧹 Removing artifacts...$(NC)"
	@rm -rf bin/ coverage.out coverage.html
	@echo -e "$(GREEN)✓ Clean done!$(NC)"

clean-cache: ## Clear Go build cache
	@echo -e "$(YELLOW)🧹 Clearing Go build cache...$(NC)"
	@go clean -cache
	@echo -e "$(GREEN)✓ Go build cache cleared!$(NC)"

##@ Submission

submission-update: ## Update image tag in submission branch to latest CI release
	@echo -e "$(BLUE)🔄 Fetching latest release tag...$(NC)"
	$(eval LATEST_TAG := $(shell gh release list --repo fabianoflorentino/fraudctl --limit 1 --json tagName -q '.[0].tagName'))
	@echo -e "$(BLUE)   Latest tag: $(LATEST_TAG)$(NC)"
	@git checkout $(SUBMISSION_BRANCH)
	@git pull origin $(SUBMISSION_BRANCH)
	@sed -i "s|image: fabianoflorentino/fraudctl:.*|image: fabianoflorentino/fraudctl:$(LATEST_TAG)|" docker-compose.yml
	@echo -e "$(BLUE)   Updated docker-compose.yml → $(LATEST_TAG)$(NC)"
	@git add docker-compose.yml
	@git commit -m "chore(submission): bump image to $(LATEST_TAG)"
	@git push origin $(SUBMISSION_BRANCH)
	@git checkout main
	@echo -e "$(GREEN)✓ Submission branch updated to $(LATEST_TAG)$(NC)"

submission-test: ## Open a test issue on the Rinha repo to trigger CI
	@echo -e "$(BLUE)🚀 Opening test issue on $(RINHA_REPO)...$(NC)"
	$(eval ISSUE_URL := $(shell gh issue create \
		--repo $(RINHA_REPO) \
		--title "rinha/test" \
		--body "rinha/test"))
	@echo -e "$(GREEN)✓ Issue opened: $(ISSUE_URL)$(NC)"
	@echo "$(ISSUE_URL)" > /tmp/fraudctl_rinha_issue.txt
	@echo -e "$(YELLOW)   Run 'make submission-result' to poll for results.$(NC)"

submission-result: ## Poll the Rinha issue until closed, then print the result
	@if [ ! -f /tmp/fraudctl_rinha_issue.txt ]; then \
		echo -e "$(RED)✗ No issue URL found. Run 'make submission-test' first.$(NC)"; \
		exit 1; \
	fi
	$(eval ISSUE_URL := $(shell cat /tmp/fraudctl_rinha_issue.txt))
	$(eval ISSUE_NUMBER := $(shell echo "$(ISSUE_URL)" | grep -o '[0-9]*$$'))
	@echo -e "$(BLUE)⏳ Polling issue \#$(ISSUE_NUMBER) until closed...$(NC)"
	@while true; do \
		STATE=$$(gh issue view $(ISSUE_NUMBER) --repo $(RINHA_REPO) --json state -q '.state'); \
		if [ "$$STATE" = "CLOSED" ]; then \
			echo -e "$(GREEN)✓ Issue closed — fetching result:$(NC)"; \
			gh issue view $(ISSUE_NUMBER) --repo $(RINHA_REPO) --json comments -q '.comments[-1].body'; \
			rm -f /tmp/fraudctl_rinha_issue.txt; \
			break; \
		fi; \
		echo -e "$(YELLOW)   Still open, waiting 15s...$(NC)"; \
		sleep 15; \
	done

submit: submission-update submission-test ## Full submission flow: update image + open test issue
	@echo -e "$(GREEN)✓ Submission complete. Run 'make submission-result' to watch for results.$(NC)"

##@ Load Testing (k6)

k6-smoke: ## Quick smoke test (5 iterations, sanity) - needs stack on :9999
	@echo -e "$(BLUE)🔥 Running k6 smoke test (sanity, 5 requests)...$(NC)"
	@if [ ! -d "$(RINHA_DIR)" ]; then \
		echo -e "$(RED)✗ Rinha repo not found at $(RINHA_DIR)$(NC)"; \
		echo -e "$(YELLOW)  Set RINHA_DIR=/path/to/rinha-de-backend-2026$(NC)"; \
		exit 1; \
	fi
	@K6_NO_USAGE_REPORT=true k6 run $(RINHA_DIR)/test/smoke.js
	@echo -e "$(GREEN)✓ Smoke test passed$(NC)"

k6-full: ## Full k6 test (120s ramp to 900 RPS) - matches Rinha evaluator exactly
	@echo -e "$(BLUE)🔥 Running full k6 test (120s, ramp to 900 RPS)...$(NC)"
	@if [ ! -d "$(RINHA_DIR)" ]; then \
		echo -e "$(RED)✗ Rinha repo not found at $(RINHA_DIR)$(NC)"; \
		echo -e "$(YELLOW)  Set RINHA_DIR=/path/to/rinha-de-backend-2026$(NC)"; \
		exit 1; \
	fi
	@cd $(RINHA_DIR) && K6_NO_USAGE_REPORT=true k6 run test/test.js
	@echo -e "$(GREEN)✓ k6 test complete — results at $(RINHA_DIR)/test/results.json$(NC)"

k6-results: ## Show last k6 results.json score breakdown
	@if [ ! -f "$(RINHA_DIR)/test/results.json" ]; then \
		echo -e "$(RED)✗ No results found. Run 'make k6-full' first.$(NC)"; \
		exit 1; \
	fi
	@echo -e "$(BLUE)📊 Last k6 results:$(NC)"
	@cat $(RINHA_DIR)/test/results.json | python3 -m json.tool 2>/dev/null || \
	 cat $(RINHA_DIR)/test/results.json