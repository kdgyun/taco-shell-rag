package config

import (
	"errors"
	"math"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"time"

	"github.com/kdgyun/taco-shell-rag/internal/core"
	"github.com/pelletier/go-toml/v2"
)

// Go는 소스 ID만 사용한다. 수집기별 옵션은 Python이 소유한다.
// Go consumes source IDs only; collector-specific options belong to Python.
type Source map[string]any

func (s Source) ID() string { id, _ := s["id"].(string); return id }

type Retrieval = core.RetrievalOptions

type Settings struct {
	EmbeddingURL      string         `toml:"embedding_url"`
	InferenceURL      string         `toml:"inference_url"`
	RerankerURL       string         `toml:"reranker_url"`
	EmbeddingTokenEnv string         `toml:"embedding_token_env"`
	RerankerTokenEnv  string         `toml:"reranker_token_env"`
	QdrantURL         string         `toml:"qdrant_url"`
	QdrantTokenEnv    string         `toml:"qdrant_token_env"`
	Collection        string         `toml:"collection"`
	Catalog           string         `toml:"catalog"`
	ChunkSize         int            `toml:"chunk_size"`
	Overlap           int            `toml:"overlap"`
	Rerank            bool           `toml:"rerank"`
	Host              string         `toml:"host"`
	Port              int            `toml:"port"`
	TokenEnv          string         `toml:"token_env"`
	AllowedHosts      []string       `toml:"allowed_hosts"`
	AllowedOrigins    []string       `toml:"allowed_origins"`
	Sources           []Source       `toml:"sources"`
	Generation        map[string]any `toml:"generation"`
	Retrieval         Retrieval      `toml:"retrieval"`
}

const RequestTimeout = 90 * time.Second

func Load(path string) (Settings, error) {
	s := Settings{
		QdrantURL:  "http://qdrant:6333",
		Collection: "taco", Catalog: "data/catalog.sqlite3", Rerank: true,
		Host: "127.0.0.1", Port: 8080, TokenEnv: "MCP_AUTH_TOKEN", ChunkSize: 1000, Overlap: 120,
		Retrieval: Retrieval{Mode: "hybrid", CandidateLimit: 160, RRFK: 60, DenseWeight: 1,
			KeywordWeight: 1, MaxPlanTerms: 6, MaxPlanSources: 4, MaxEvidenceChars: 16000},
	}
	f, err := os.Open(path)
	if err != nil {
		return s, errors.New("cannot read configuration")
	}
	defer f.Close()
	if err := toml.NewDecoder(f).DisallowUnknownFields().Decode(&s); err != nil {
		return s, errors.New("invalid configuration; check taco.toml")
	}
	if s.InferenceURL != "" {
		if s.EmbeddingURL != "" {
			return s, errors.New("use embedding_url without inference_url")
		}
		s.EmbeddingURL = s.InferenceURL
	}
	if s.EmbeddingURL == "" {
		s.EmbeddingURL = "http://inference:8000"
	}
	if s.RerankerURL == "" {
		s.RerankerURL = s.EmbeddingURL
	}
	for _, endpoint := range []string{s.EmbeddingURL, s.RerankerURL, s.QdrantURL} {
		u, err := url.Parse(endpoint)
		if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" || u.User != nil || u.RawQuery != "" || u.Fragment != "" {
			return s, errors.New("service URLs must not embed credentials, queries or fragments")
		}
	}
	if !regexp.MustCompile(`^[a-zA-Z][a-zA-Z0-9_-]{0,63}$`).MatchString(s.Collection) || s.Catalog == "" {
		return s, errors.New("invalid collection or catalog")
	}
	if s.Port < 1 || s.Port > 65535 || s.Host == "" || len(s.Sources) == 0 || s.Overlap < 0 || s.Overlap >= s.ChunkSize {
		return s, errors.New("invalid listener, source or chunk settings")
	}
	seen := map[string]bool{}
	for _, src := range s.Sources {
		if !regexp.MustCompile(`^[a-z][a-z0-9_-]{0,63}$`).MatchString(src.ID()) || seen[src.ID()] {
			return s, errors.New("invalid or duplicate source")
		}
		seen[src.ID()] = true
	}
	r := s.Retrieval
	if (r.Mode != "hybrid" && r.Mode != "dense") || r.CandidateLimit < 20 || r.CandidateLimit > 160 || r.RRFK < 1 || r.RRFK > 1000 || r.DenseWeight <= 0 || r.KeywordWeight <= 0 || math.IsNaN(r.DenseWeight) || math.IsInf(r.DenseWeight, 0) || math.IsNaN(r.KeywordWeight) || math.IsInf(r.KeywordWeight, 0) || r.MaxPlanTerms < 1 || r.MaxPlanTerms > 10 || r.MaxPlanSources < 1 || r.MaxPlanSources > 8 || r.MaxEvidenceChars < 4000 || r.MaxEvidenceChars > 80000 {
		return s, errors.New("invalid retrieval settings; see docs/retrieval.md")
	}
	for _, name := range []string{s.TokenEnv, s.EmbeddingTokenEnv, s.RerankerTokenEnv, s.QdrantTokenEnv} {
		if name != "" && !regexp.MustCompile(`^[A-Z][A-Z0-9_]*$`).MatchString(name) {
			return s, errors.New("invalid credential environment variable name")
		}
	}
	if !filepath.IsAbs(s.Catalog) {
		s.Catalog = filepath.Join(filepath.Dir(path), s.Catalog)
	}
	return s, nil
}

func Credential(name string) (string, error) {
	if name == "" {
		return "", nil
	}
	value := os.Getenv(name)
	if value == "" {
		return "", errors.New("configured credential environment variable is unset")
	}
	return value, nil
}
