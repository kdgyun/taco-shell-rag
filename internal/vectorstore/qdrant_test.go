package vectorstore

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestEmptyAndMalformedVectorResponses(t *testing.T) {
	for _, tc := range []struct {
		body  string
		valid bool
	}{
		{`{"result":{"points":[]}}`, true}, {`{}`, false}, {`{"result":null}`, false}, {`{"result":{}}`, false},
	} {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Header.Get("api-key") != "test-token" {
				t.Error("missing configured credential")
			}
			_, _ = w.Write([]byte(tc.body))
		}))
		q := New(server.URL, "notes", "test-token")
		hits, err := q.Search(context.Background(), []float64{1, 0}, 5, []string{"notes"})
		server.Close()
		if (err == nil) != tc.valid || (tc.valid && len(hits) != 0) {
			t.Fatalf("response %s: hits=%v err=%v", tc.body, hits, err)
		}
	}
}

func TestVectorRequestNeverFollowsRedirectsOrLeaksUpstreamBody(t *testing.T) {
	calls := 0
	target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { calls++; w.WriteHeader(200) }))
	defer target.Close()
	source := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Location", target.URL)
		w.WriteHeader(302)
		_, _ = w.Write([]byte("private-upstream-body"))
	}))
	defer source.Close()
	_, err := New(source.URL, "notes", "test-token").Search(context.Background(), []float64{1, 0}, 5, []string{"notes"})
	if err == nil || strings.Contains(err.Error(), "private-upstream-body") || calls != 0 {
		t.Fatalf("redirect/body boundary failed: calls=%d err=%v", calls, err)
	}
}
