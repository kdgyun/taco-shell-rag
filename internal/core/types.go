package core

import (
	"context"
	"encoding/json"
	"errors"
)

var ErrNotInitialized = errors.New("catalog is not initialized; run taco sync first")
var ErrLexicalUnavailable = errors.New("keyword index is unavailable; run taco sync once to upgrade the catalog")

type RetrievalOptions struct {
	Mode             string  `toml:"mode"`
	CandidateLimit   int     `toml:"candidate_limit"`
	RRFK             int     `toml:"rrf_k"`
	DenseWeight      float64 `toml:"dense_weight"`
	KeywordWeight    float64 `toml:"keyword_weight"`
	MaxPlanTerms     int     `toml:"max_plan_terms"`
	MaxPlanSources   int     `toml:"max_plan_sources"`
	MaxEvidenceChars int     `toml:"max_evidence_chars"`
}

type Document struct {
	ID        string         `json:"id"`
	SourceID  string         `json:"source_id"`
	Text      string         `json:"text,omitempty"`
	Title     string         `json:"title"`
	URL       string         `json:"url"`
	Path      string         `json:"path"`
	Revision  string         `json:"revision"`
	UpdatedAt string         `json:"updated_at"`
	Metadata  map[string]any `json:"metadata,omitempty"`
}

type Chunk struct {
	DocumentKey string `json:"document_key"`
	SourceID    string `json:"source_id"`
	Version     string `json:"version"`
	Index       int    `json:"index"`
	Text        string `json:"text"`
}

type Record struct {
	Document Document
	Chunks   []Chunk
}

type Hit struct {
	Document  Document `json:"document"`
	Chunk     Chunk    `json:"chunk"`
	Score     float64  `json:"score"`
	MatchedBy []string `json:"matched_by,omitempty"`
}

type EmbeddingSpace struct {
	Model     string         `json:"model"`
	Revision  string         `json:"revision"`
	Dimension int            `json:"dimension"`
	Device    string         `json:"device"`
	Encoding  map[string]any `json:"encoding"`
}

type Embedding struct {
	Space       EmbeddingSpace `json:"embedding_space"`
	Fingerprint string         `json:"embedding_fingerprint"`
	Vectors     [][]float64    `json:"vectors"`
}

type RerankerInfo struct {
	Model     string          `json:"model"`
	Revision  string          `json:"revision"`
	Device    string          `json:"device"`
	Backend   string          `json:"backend"`
	Precision string          `json:"precision"`
	MaxLength int             `json:"max_length"`
	BatchSize int             `json:"batch_size"`
	Prompt    json.RawMessage `json:"prompt"`
	ScoreKind string          `json:"score_kind"`
}

type Reranked struct {
	Scores []float64    `json:"scores"`
	Info   RerankerInfo `json:"reranker"`
}

type Ledger interface {
	CheckSpace(context.Context, Embedding) error
	Get(context.Context, string) (*Record, error)
	Lexical(context.Context, []string, bool, int, []string) ([]Hit, bool, error)
	Status(context.Context) (map[string]any, error)
}

type Embedder interface {
	Space(context.Context) (Embedding, error)
	Embed(context.Context, string) (Embedding, error)
}

type Reranker interface {
	Rerank(context.Context, string, []string) (Reranked, error)
}

type VectorIndex interface {
	Search(context.Context, []float64, int, []string) ([]Hit, error)
}
