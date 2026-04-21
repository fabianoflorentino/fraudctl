---
name: git-hooks
description: Configure automated git hooks using lefthook with Go project best practices including linting, testing, and coverage requirements.
origin: fraudctl
---

# Git Hooks Configuration

This skill provides automated git hooks using [lefthook](https://github.com/evilmartians/lefthook) to ensure code quality and prevent bad commits from entering the repository.

## When to Activate

- Initializing a new Go project
- Setting up CI/CD quality gates
- Configuring pre-commit/pre-push hooks
- Adding new linting or testing requirements

## lefthook Configuration

### Standard Structure

```yaml
# lefthook.yml
# https://github.com/evilmartians/lefthook

pre-commit:
  parallel: true
  commands:
    fmt:
      glob: "**/*.go"
      exclude:
        - '^\.agents/.*'
        - '^bin/.*'
      run: bash -c 'files=$(gofmt -l .); [ -z "$files" ] && exit 0; printf "unformatted files:\n%s\n" "$files"; exit 1'

    vet:
      glob: "**/*.go"
      exclude:
        - '^\.agents/.*'
      run: go vet ./...

    lint:
      glob: "**/*.go"
      exclude:
        - '^\.agents/.*'
      run: bash -c 'command -v golangci-lint >/dev/null 2>&1 || exit 0; golangci-lint run'

    test:
      exclude:
        - '^\.agents/.*'
        - '^bin/.*'
      run: |
        go test -race -coverprofile=/tmp/coverage.out ./... &&
        COV=$(go tool cover -func=/tmp/coverage.out | grep "^total:" | awk '{gsub(/%/,"",$3); print $3}') &&
        awk -v cov="$COV" "BEGIN {if (cov+0 < 80) {print \"FAIL: Coverage \" cov \"%\"; exit 1}}"

pre-push:
  parallel: true
  commands:
    build:
      exclude:
        - '^\.agents/.*'
      run: go build ./...

    lint-full:
      exclude:
        - '^\.agents/.*'
      run: bash -c 'command -v golangci-lint >/dev/null 2>&1 || exit 0; golangci-lint run --timeout=5m'

    test-cover:
      exclude:
        - '^\.agents/.*'
        - '^bin/.*'
      run: |
        go test -race -coverprofile=/tmp/coverage.out ./... &&
        COV=$(go tool cover -func=/tmp/coverage.out | grep "^total:" | awk '{gsub(/%/,"",$3); print $3}') &&
        awk -v cov="$COV" "BEGIN {if (cov+0 < 80) {print \"FAIL: Coverage \" cov \"%\"; exit 1}}"

    vuln:
      exclude:
        - '^\.agents/.*'
      run: bash -c 'command -v govulncheck >/dev/null 2>&1 || exit 0; govulncheck ./...'
```

## Important Notes

### exclude Syntax

The `exclude` option in lefthook is for filtering **files** passed to commands, not for glob patterns. Each item must be a regex pattern:

```yaml
# CORRECT - list of regex patterns
exclude:
  - '^\.agents/.*'
  - '^bin/.*'

# WRONG - trying to use anchor as exclude (causes YAML error)
_standard_excludes: &standard_excludes
  - '^\.agents/'
```

### Coverage Check

For reliable coverage extraction, use `awk -v` to pass the coverage value:

```bash
# This is reliable
COV=$(go tool cover -func=coverage.out | grep "^total:" | awk '{gsub(/%/,"",$3); print $3}')

# This can fail with special characters
COV=$(go tool cover -func=coverage.out | grep '^total:' | awk '{print $3}' | tr -d '%')
```

## Installation

```bash
# Install lefthook
npm install -g lefthook

# Or with Go
go install github.com/evilmartians/lefthook/cmd/lefthook@latest

# Initialize in project
lefthook install

# Run hooks manually
lefthook run pre-commit
lefthook run pre-commit --force  # Force run all hooks

# Run specific hook
lefthook run lint
```

## Pre-commit Hooks

| Hook | Purpose | Tool |
|------|---------|------|
| fmt | Check formatting | gofmt |
| goimports | Check import order | goimports |
| vet | Static analysis | go vet |
| lint | Quick lint check | golangci-lint |
| test | Run tests with coverage | go test |

## Pre-push Hooks

| Hook | Purpose | Tool |
|------|---------|------|
| build | Verify build compiles | go build |
| lint-full | Full lint check | golangci-lint |
| test-cover | Verify 80% coverage | go test |
| vuln | Check vulnerabilities | govulncheck |

## Coverage Requirements

Go projects should maintain minimum **80% test coverage**:

```bash
# Check coverage
go test -coverprofile=coverage.out ./...
go tool cover -func=coverage.out | grep '^total:'

# Coverage by package
go test -coverprofile=coverage.out ./... -covermode=atomic
```

## Optional Tools

These tools are optional — hooks skip if not installed:

```bash
# Install golangci-lint
curl -sSfL https://raw.githubusercontent.com/golangci/golangci-lint/master/install.sh | sh -s -- -b $(go env GOPATH)/bin

# Install goimports
go install golang.org/x/tools/cmd/goimports@latest

# Install govulncheck
go install golang.org/x/vuln/cmd/govulncheck@latest

# Install semgrep
npm install -g semgrep
```

## Error Handling

Hooks should fail fast to prevent bad commits:

```bash
# Example: Fail on any formatting issues
gofmt -l . && echo "Unformatted files found" && exit 1

# Example: Fail if coverage drops
COV=$(go tool cover -func=coverage.out | grep "^total:" | awk '{gsub(/%/,"",$3); print $3}')
if [ "$COV" -lt 80 ]; then
  echo "Coverage $COV% is below 80%"
  exit 1
fi
```

## Best Practices

**DO:**
- Use `parallel: true` for speed
- Skip optional tools gracefully with `|| exit 0`
- Set minimum coverage at 80%
- Use absolute paths in coverage outputs
- Keep hooks fast (< 30 seconds)

**DON'T:**
- Block on optional tools
- Run slow operations in pre-commit
- Use YAML anchors for `exclude` (not supported)
- Use special characters in coverage extraction without proper escaping