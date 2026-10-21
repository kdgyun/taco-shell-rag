package httptransport

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	"github.com/kdgyun/taco-shell-rag/internal/catalog"
	mcpserver "github.com/kdgyun/taco-shell-rag/internal/mcp"
	"github.com/kdgyun/taco-shell-rag/internal/retriever"
)

func testHandler(t *testing.T) http.Handler {
	t.Helper()
	svc := &retriever.Service{Ledger: &catalog.Catalog{Path: filepath.Join(t.TempDir(), "missing.sqlite3")}, Sources: []string{"notes"}}
	handler, err := New(mcpserver.New(svc), strings.Repeat("a", 64), nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	return handler
}

func TestAuthenticationHostAndOrigin(t *testing.T) {
	for _, tc := range []struct {
		name, host, origin, token string
		want                      int
	}{
		{"no-token", "localhost:8080", "", "", 401},
		{"foreign-host", "attacker.invalid", "", strings.Repeat("a", 64), 421},
		{"foreign-origin", "localhost:8080", "https://attacker.invalid", strings.Repeat("a", 64), 403},
		{"local-origin", "localhost:8080", "http://localhost:8080", strings.Repeat("a", 64), 400},
	} {
		t.Run(tc.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodGet, "http://"+tc.host+"/mcp", nil)
			req.Host = tc.host
			if tc.origin != "" {
				req.Header.Set("Origin", tc.origin)
			}
			if tc.token != "" {
				req.Header.Set("Authorization", "Bearer "+tc.token)
			}
			w := httptest.NewRecorder()
			testHandler(t).ServeHTTP(w, req)
			if w.Code != tc.want {
				t.Fatalf("status=%d body=%s", w.Code, w.Body.String())
			}
		})
	}
	if _, err := New(nil, "short", nil, nil); err == nil {
		t.Fatal("short token accepted")
	}
}

func TestActualHTTPMCPHandshakeAndTools(t *testing.T) {
	server := httptest.NewServer(testHandler(t))
	defer server.Close()
	call := func(body string) map[string]any {
		t.Helper()
		req, err := http.NewRequest(http.MethodPost, server.URL+"/mcp", bytes.NewBufferString(body))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Authorization", "Bearer "+strings.Repeat("a", 64))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Accept", "application/json, text/event-stream")
		req.Header.Set("MCP-Protocol-Version", "2025-11-25")
		response, err := server.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer response.Body.Close()
		data, _ := io.ReadAll(response.Body)
		if response.StatusCode != 200 {
			t.Fatalf("status=%d body=%s", response.StatusCode, data)
		}
		var result map[string]any
		if json.Unmarshal(data, &result) != nil {
			t.Fatalf("not JSON: %s", data)
		}
		if result["error"] != nil {
			t.Fatalf("MCP error: %s", data)
		}
		return result["result"].(map[string]any)
	}
	init := call(`{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"test","version":"1"}}}`)
	if init["serverInfo"].(map[string]any)["name"] != "taco-shell-rag" {
		t.Fatal(init)
	}
	listing := call(`{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}`)
	tools := listing["tools"].([]any)
	if len(tools) != 5 {
		t.Fatalf("tool count=%d", len(tools))
	}
	status := call(`{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"status","arguments":{}}}`)
	if status["isError"] == true || status["structuredContent"].(map[string]any)["implementation"] != "go" {
		t.Fatalf("status: %+v", status)
	}
}
