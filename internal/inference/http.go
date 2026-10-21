package inference

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"

	"github.com/kdgyun/taco-shell-rag/internal/config"
	"github.com/kdgyun/taco-shell-rag/internal/core"
)

type Client struct {
	URL, Token string
	HTTP       *http.Client
}

func New(url, token string) *Client {
	return &Client{URL: strings.TrimRight(url, "/"), Token: token, HTTP: &http.Client{
		Timeout:       config.RequestTimeout,
		Transport:     &http.Transport{Proxy: nil},
		CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
	}}
}

func (c *Client) request(ctx context.Context, method, path string, body any, result any) error {
	var data []byte
	var err error
	if body != nil {
		data, err = json.Marshal(body)
		if err != nil {
			return errors.New("invalid inference request")
		}
	}
	req, err := http.NewRequestWithContext(ctx, method, c.URL+path, bytes.NewReader(data))
	if err != nil {
		return errors.New("invalid inference endpoint")
	}
	if c.Token != "" {
		req.Header.Set("Authorization", "Bearer "+c.Token)
	}
	req.Header.Set("Content-Type", "application/json")
	response, err := c.HTTP.Do(req)
	if err != nil {
		return errors.New("inference service is unavailable")
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return errors.New("inference service rejected the request; check model readiness")
	}
	data, err = io.ReadAll(io.LimitReader(response.Body, 8*1024*1024+1))
	if err != nil || len(data) > 8*1024*1024 || json.Unmarshal(data, result) != nil {
		return errors.New("inference response is malformed or oversized")
	}
	return nil
}

func (c *Client) Space(ctx context.Context) (core.Embedding, error) {
	var data struct {
		Ready       bool                `json:"ready"`
		Space       core.EmbeddingSpace `json:"embedding_space"`
		Fingerprint string              `json:"embedding_fingerprint"`
	}
	err := c.request(ctx, http.MethodGet, "/healthz", nil, &data)
	if err == nil && !data.Ready {
		err = errors.New("inference service is not ready")
	}
	return core.Embedding{Space: data.Space, Fingerprint: data.Fingerprint}, err
}

func (c *Client) Embed(ctx context.Context, query string) (core.Embedding, error) {
	var result core.Embedding
	err := c.request(ctx, http.MethodPost, "/embed", map[string]any{"texts": []string{query}, "purpose": "query"}, &result)
	return result, err
}

func (c *Client) Rerank(ctx context.Context, query string, texts []string) (core.Reranked, error) {
	var result core.Reranked
	err := c.request(ctx, http.MethodPost, "/rerank", map[string]any{"query": query, "texts": texts}, &result)
	return result, err
}
