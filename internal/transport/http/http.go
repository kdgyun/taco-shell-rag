package httptransport

import (
	"crypto/subtle"
	"errors"
	"net/http"
	"net/url"
	"strings"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

func matches(actual string, patterns []string) bool {
	for _, pattern := range patterns {
		if actual == pattern {
			return true
		}
		if strings.HasSuffix(pattern, ":*") {
			base := strings.TrimSuffix(pattern, ":*")
			if actual == base || strings.HasPrefix(actual, base+":") {
				parsed, err := url.Parse("http://" + actual)
				if err == nil && parsed.Hostname() == strings.Trim(base, "[]") {
					return true
				}
			}
		}
	}
	return false
}

func New(server *mcp.Server, token string, hosts, origins []string) (http.Handler, error) {
	if len(token) < 32 {
		return nil, errors.New("HTTP MCP requires a token of at least 32 characters")
	}
	if hosts == nil {
		hosts = []string{"localhost:*", "127.0.0.1:*", "[::1]:*"}
	}
	if origins == nil {
		origins = []string{"http://localhost:*", "http://127.0.0.1:*", "http://[::1]:*"}
	}
	mcpHandler := mcp.NewStreamableHTTPHandler(func(*http.Request) *mcp.Server { return server }, &mcp.StreamableHTTPOptions{Stateless: true, JSONResponse: true})
	mux := http.NewServeMux()
	mux.Handle("/mcp", mcpHandler)
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"ready":true,"scope":"MCP transport only"}`))
	})
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !matches(strings.ToLower(r.Host), hosts) {
			http.Error(w, "untrusted host", http.StatusMisdirectedRequest)
			return
		}
		origin := r.Header.Get("Origin")
		if origin != "" {
			u, err := url.Parse(origin)
			if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.User != nil || u.Path != "" || u.RawQuery != "" || u.Fragment != "" || !originMatches(origin, origins) {
				http.Error(w, "untrusted origin", http.StatusForbidden)
				return
			}
			w.Header().Set("Access-Control-Allow-Origin", origin)
			w.Header().Set("Vary", "Origin")
			w.Header().Set("Access-Control-Allow-Headers", "Authorization, Content-Type, MCP-Protocol-Version, MCP-Session-Id, Last-Event-ID")
			w.Header().Set("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
		}
		if r.Method == http.MethodOptions && origin != "" {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		if r.URL.Path != "/healthz" && subtle.ConstantTimeCompare([]byte(r.Header.Get("Authorization")), []byte("Bearer "+token)) != 1 {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		r.Body = http.MaxBytesReader(w, r.Body, 1024*1024)
		mux.ServeHTTP(w, r)
	}), nil
}

func originMatches(actual string, patterns []string) bool {
	u, err := url.Parse(actual)
	if err != nil {
		return false
	}
	for _, pattern := range patterns {
		parts := strings.SplitN(pattern, "://", 2)
		if len(parts) == 2 && u.Scheme == parts[0] && matches(u.Host, []string{parts[1]}) {
			return true
		}
	}
	return false
}
