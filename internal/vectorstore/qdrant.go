package vectorstore

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/kdgyun/taco-shell-rag/internal/core"
)

type Qdrant struct {
	URL, Collection, Token string
	HTTP                   *http.Client
}

func New(url, collection, token string) *Qdrant {
	return &Qdrant{URL: strings.TrimRight(url, "/"), Collection: collection, Token: token,
		HTTP: &http.Client{Timeout: 30 * time.Second, Transport: &http.Transport{Proxy: nil},
			CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}}
}

func (q *Qdrant) Search(ctx context.Context, vector []float64, limit int, sources []string) ([]core.Hit, error) {
	filter := map[string]any{"must": []any{map[string]any{"key": "source_id", "match": map[string]any{"any": sources}}}}
	body, err := json.Marshal(map[string]any{"query": vector, "filter": filter, "limit": limit, "with_payload": true, "with_vector": false})
	if err != nil {
		return nil, errors.New("invalid vector query")
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, q.URL+"/collections/"+url.PathEscape(q.Collection)+"/points/query", bytes.NewReader(body))
	if err != nil {
		return nil, errors.New("invalid vector endpoint")
	}
	req.Header.Set("Content-Type", "application/json")
	if q.Token != "" {
		req.Header.Set("api-key", q.Token)
	}
	response, err := q.HTTP.Do(req)
	if err != nil {
		return nil, errors.New("vector service is unavailable")
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return nil, errors.New("vector query failed; check collection and service health")
	}
	data, err := io.ReadAll(io.LimitReader(response.Body, 8*1024*1024+1))
	var result struct {
		Result *struct {
			Points []struct {
				Payload core.Chunk `json:"payload"`
				Score   float64    `json:"score"`
			} `json:"points"`
		} `json:"result"`
	}
	if err != nil || len(data) > 8*1024*1024 || json.Unmarshal(data, &result) != nil || result.Result == nil || result.Result.Points == nil {
		return nil, errors.New("vector response is malformed or oversized")
	}
	hits := make([]core.Hit, len(result.Result.Points))
	if len(hits) > limit {
		return nil, errors.New("vector service exceeded the candidate bound")
	}
	for i, point := range result.Result.Points {
		hits[i] = core.Hit{Chunk: point.Payload, Score: point.Score, MatchedBy: []string{"dense"}}
	}
	return hits, nil
}
