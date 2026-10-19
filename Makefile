.PHONY: build fmt test check

build:
	go build -mod=readonly -trimpath -o bin/mcp-http ./cmd/mcp-http
	go build -mod=readonly -trimpath -o bin/mcp-stdio ./cmd/mcp-stdio

fmt:
	gofmt -w cmd internal

test:
	go test -mod=readonly ./...
	uv run --frozen pytest -q

check:
	go vet -mod=readonly ./...
	uv run --frozen ruff check src tests scripts
