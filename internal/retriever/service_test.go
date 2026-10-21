package retriever

import (
	"context"
	"errors"
	"math"
	"strings"
	"testing"

	"github.com/kdgyun/taco-shell-rag/internal/config"
	"github.com/kdgyun/taco-shell-rag/internal/core"
)

type ledgerFake struct {
	records  map[string]*core.Record
	lexical  []core.Hit
	mismatch bool
}

func (l *ledgerFake) CheckSpace(context.Context, core.Embedding) error {
	if l.mismatch {
		return errors.New("different embedding space")
	}
	return nil
}
func (l *ledgerFake) Get(_ context.Context, key string) (*core.Record, error) {
	return l.records[key], nil
}
func (l *ledgerFake) Status(context.Context) (map[string]any, error) {
	return map[string]any{"documents": len(l.records)}, nil
}
func (l *ledgerFake) Lexical(_ context.Context, terms []string, all bool, limit int, sources []string) ([]core.Hit, bool, error) {
	hits := []core.Hit{}
	for _, hit := range l.lexical {
		if !contains(sources, hit.Chunk.SourceID) {
			continue
		}
		n := 0
		for _, term := range terms {
			if core.ContainsTerm(hit.Chunk.Text, term) {
				n++
			}
		}
		if n > 0 && (!all || n == len(terms)) {
			hit.MatchedBy = []string{"keyword"}
			hits = append(hits, hit)
		}
	}
	truncated := len(hits) > limit
	if truncated {
		hits = hits[:limit]
	}
	return hits, truncated, nil
}

type embedFake struct {
	health, execution core.Embedding
	calls             int
	forbidden         bool
}

func (e *embedFake) Space(context.Context) (core.Embedding, error) {
	if e.forbidden {
		panic("keyword search must not call models")
	}
	return e.health, nil
}
func (e *embedFake) Embed(context.Context, string) (core.Embedding, error) {
	e.calls++
	return e.execution, nil
}

type indexFake struct {
	hits        []core.Hit
	calls       int
	ignoreScope bool
}

func (i *indexFake) Search(_ context.Context, _ []float64, limit int, sources []string) ([]core.Hit, error) {
	i.calls++
	out := []core.Hit{}
	for _, hit := range i.hits {
		if i.ignoreScope || contains(sources, hit.Chunk.SourceID) {
			out = append(out, hit)
		}
	}
	if len(out) > limit {
		out = out[:limit]
	}
	return out, nil
}

type rankFake struct {
	before func()
	scores []float64
	device string
}

func (r *rankFake) Rerank(context.Context, string, []string) (core.Reranked, error) {
	if r.before != nil {
		r.before()
	}
	return core.Reranked{Scores: r.scores, Info: core.RerankerInfo{Model: "test-ranker", Revision: "r1", Backend: "test", Device: r.device, ScoreKind: "reranker_raw_uncalibrated"}}, nil
}

func fixture() (*Service, *ledgerFake, *embedFake, *indexFake) {
	dense := core.Chunk{DocumentKey: strings.Repeat("a", 64), SourceID: "notes", Version: strings.Repeat("1", 64), Text: "payment guidance"}
	keyword := core.Chunk{DocumentKey: strings.Repeat("b", 64), SourceID: "specs", Version: strings.Repeat("2", 64), Text: "ABC-123 refund policy"}
	l := &ledgerFake{records: map[string]*core.Record{
		dense.DocumentKey:   {Document: core.Document{ID: "a", SourceID: "notes", Text: dense.Text}, Chunks: []core.Chunk{dense}},
		keyword.DocumentKey: {Document: core.Document{ID: "b", SourceID: "specs", Text: keyword.Text}, Chunks: []core.Chunk{keyword}},
	}, lexical: []core.Hit{{Chunk: keyword, Score: 1}}}
	space := core.Embedding{Space: core.EmbeddingSpace{Model: "test-embedding", Revision: "r1", Dimension: 2, Device: "cuda:0"}, Fingerprint: strings.Repeat("a", 64), Vectors: [][]float64{{1, 0}}}
	e := &embedFake{health: space, execution: space}
	index := &indexFake{hits: []core.Hit{{Chunk: dense, Score: .9}}}
	s := &Service{Ledger: l, Embedder: e, Index: index, Sources: []string{"notes", "specs"}, Options: config.Retrieval{Mode: "hybrid", CandidateLimit: 160, RRFK: 60, DenseWeight: 1, KeywordWeight: 1, MaxPlanTerms: 6, MaxPlanSources: 4, MaxEvidenceChars: 16000}}
	return s, l, e, index
}

func TestHybridIncludesLexicalOnlyEvidence(t *testing.T) {
	s, _, e, _ := fixture()
	result, err := s.Search(context.Background(), SearchInput{Query: "refund"}, false)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Results) != 2 || result.ScoreKind != "rrf" || e.calls != 1 {
		t.Fatalf("unexpected hybrid results: %+v", result)
	}
	for _, hit := range result.Results {
		if hit.Document.Text != "" || hit.Chunk.Version == "" {
			t.Fatal("response must retain bounded chunks and exact versions")
		}
	}
}

func TestDenseModeIsExplicitAndDoesNotAddLexicalHits(t *testing.T) {
	s, _, _, _ := fixture()
	result, err := s.Search(context.Background(), SearchInput{Query: "refund", Mode: "dense"}, false)
	if err != nil || len(result.Results) != 1 || result.ScoreKind != "cosine" {
		t.Fatalf("dense results: %+v %v", result, err)
	}
}

func TestKeywordDoesNotRequireInference(t *testing.T) {
	s, _, e, _ := fixture()
	e.forbidden = true
	s.Reranker = &rankFake{}
	result, err := s.Keyword(context.Background(), SearchInput{Query: "ABC-123"}, "all")
	if err != nil || len(result.Results) != 1 || result.EmbeddingSpace != nil || result.Reranker != nil || result.ScoreKind != "bm25" {
		t.Fatalf("keyword results: %+v %v", result, err)
	}
}

func TestPlannedSearchReusesOneEmbeddingAndReportsUncoveredAnchors(t *testing.T) {
	s, _, e, index := fixture()
	result, err := s.Search(context.Background(), SearchInput{Query: `ABC-123 ABC-999 "refund policy"`}, true)
	if err != nil {
		t.Fatal(err)
	}
	if e.calls != 1 || index.calls != 3 || len(result.Plan.Anchors) != 3 {
		t.Fatalf("unexpected plan: %+v calls=%d/%d", result.Plan, e.calls, index.calls)
	}
	if len(result.Results) != 2 || len(result.CoverageGaps) == 0 || result.Anchors[1].Found {
		t.Fatalf("missing anchor must remain a gap: %+v", result)
	}
}

func TestPlanBoundsAreObservable(t *testing.T) {
	s, _, _, _ := fixture()
	s.Options.MaxPlanTerms = 1
	s.Options.MaxPlanSources = 1
	result, err := s.Search(context.Background(), SearchInput{Query: "ABC-123 ABC-999"}, true)
	if err != nil {
		t.Fatal(err)
	}
	if result.Plan.OmittedAnchors != 1 || result.Plan.OmittedSources != 1 || len(result.CoverageGaps) == 0 {
		t.Fatalf("unreported omission: %+v", result)
	}
}

func TestUnpublishedForeignAndForgedChunksAreDiscarded(t *testing.T) {
	s, _, _, index := fixture()
	index.ignoreScope = true
	original := index.hits[0]
	stale := original
	stale.Chunk.Version = "old"
	foreign := original
	foreign.Chunk.SourceID = "foreign"
	forged := original
	forged.Chunk.Text = "invented"
	index.hits = []core.Hit{stale, foreign, forged, original, original}
	result, err := s.Search(context.Background(), SearchInput{Query: "payment", Mode: "dense"}, false)
	if err != nil || len(result.Results) != 1 || result.DiscardedCandidates != 4 {
		t.Fatalf("stale evidence leaked: %+v %v", result, err)
	}
}

func TestPublicationDuringRerankDoesNotReturnRetiredVersion(t *testing.T) {
	s, l, _, _ := fixture()
	s.Reranker = &rankFake{device: "cuda", scores: []float64{1}, before: func() { l.records = map[string]*core.Record{} }}
	result, err := s.Search(context.Background(), SearchInput{Query: "payment", Mode: "dense"}, false)
	if err != nil || len(result.Results) != 0 || result.DiscardedCandidates != 1 {
		t.Fatalf("retired version leaked: %+v %v", result, err)
	}
}

func TestRejectsChangedOrNonCUDAEmbedding(t *testing.T) {
	for _, kind := range []string{"health-cpu", "execution-space", "dimension", "nan", "zero", "catalog"} {
		t.Run(kind, func(t *testing.T) {
			s, l, e, _ := fixture()
			switch kind {
			case "health-cpu":
				e.health.Space.Device = "cpu"
			case "execution-space":
				e.execution.Space.Model = "other"
			case "dimension":
				e.execution.Vectors = [][]float64{{1}}
			case "nan":
				e.execution.Vectors = [][]float64{{math.NaN(), 0}}
			case "zero":
				e.execution.Vectors = [][]float64{{0, 0}}
			case "catalog":
				l.mismatch = true
			}
			if _, err := s.Search(context.Background(), SearchInput{Query: "refund"}, false); err == nil {
				t.Fatal("unsafe inference accepted")
			}
		})
	}
}

func TestRejectsInvalidReranking(t *testing.T) {
	for _, ranker := range []*rankFake{{device: "cpu", scores: []float64{1, 2}}, {device: "cuda", scores: []float64{1}}, {device: "cuda", scores: []float64{math.NaN(), 2}}} {
		s, _, _, _ := fixture()
		s.Reranker = ranker
		if _, err := s.Search(context.Background(), SearchInput{Query: "refund"}, false); err == nil {
			t.Fatal("invalid reranking accepted")
		}
	}
}

func TestInputAndSourceScope(t *testing.T) {
	for _, in := range []SearchInput{{Query: " "}, {Query: "x", Limit: 21}, {Query: "x", SourceID: "foreign"}, {Query: "x", Mode: "invented"}} {
		s, _, _, _ := fixture()
		if _, err := s.Search(context.Background(), in, false); err == nil {
			t.Fatal("invalid input accepted")
		}
	}
	s, _, _, _ := fixture()
	result, err := s.Search(context.Background(), SearchInput{Query: "refund", SourceID: "specs"}, true)
	if err != nil || len(result.Results) != 1 || result.Results[0].Chunk.SourceID != "specs" {
		t.Fatalf("scope failed: %+v %v", result, err)
	}
}

func TestFusionDeduplicatesAndRewardsBothRetrievers(t *testing.T) {
	a := core.Hit{Chunk: core.Chunk{DocumentKey: "a"}, MatchedBy: []string{"dense"}}
	b := core.Hit{Chunk: core.Chunk{DocumentKey: "b"}, MatchedBy: []string{"keyword"}}
	shared := a
	shared.MatchedBy = []string{"keyword"}
	hits := fuse([][]core.Hit{{a, a}, {b, shared}}, []float64{1, 1}, 60)
	if len(hits) != 2 || hits[0].Chunk.DocumentKey != "a" || len(hits[0].MatchedBy) != 2 {
		t.Fatalf("bad fusion: %+v", hits)
	}
}

func TestBudgetDoesNotTruncateRawEvidence(t *testing.T) {
	s, _, _, _ := fixture()
	s.Options.MaxEvidenceChars = 4
	result, err := s.Search(context.Background(), SearchInput{Query: "refund"}, false)
	if err != nil || len(result.Results) != 0 || result.EvidenceOmitted != 2 {
		t.Fatalf("unreported evidence truncation: %+v %v", result, err)
	}
}

func TestKeywordAllDoesNotSilentlyDropQueryConditions(t *testing.T) {
	s, _, _, _ := fixture()
	query := []string{}
	for i := 0; i < 25; i++ {
		query = append(query, strings.Repeat("a", i+1))
	}
	if _, err := s.Keyword(context.Background(), SearchInput{Query: strings.Join(query, " ")}, "all"); err == nil {
		t.Fatal("all matching dropped query conditions")
	}
}

func TestCandidateBoundKeepsRareExactAnchorBeforeReranking(t *testing.T) {
	hits := []core.Hit{
		{Chunk: core.Chunk{DocumentKey: "first", SourceID: "notes", Text: "general refund policy"}, Score: 1},
		{Chunk: core.Chunk{DocumentKey: "second", SourceID: "notes", Text: "general guidance"}, Score: .9},
		{Chunk: core.Chunk{DocumentKey: "rare", SourceID: "specs", Text: "ABC-123 exception"}, Score: .01},
	}
	plan := &Plan{Anchors: []string{"ABC-123"}, Sources: []string{"notes", "specs"}}
	selected := reserveCandidates(hits, 2, plan)
	if len(selected) != 2 || selected[0].Chunk.DocumentKey != "first" || selected[1].Chunk.DocumentKey != "rare" {
		t.Fatalf("rare anchor was lost before rerank: %+v", selected)
	}
}

func TestCUDADeviceChangeKeepsTheSpaceAndReportsActualExecution(t *testing.T) {
	s, _, e, _ := fixture()
	e.execution.Space.Device = "cuda:1"
	result, err := s.Search(context.Background(), SearchInput{Query: "refund"}, false)
	if err != nil || result.EmbeddingSpace.Device != "cuda:1" {
		t.Fatalf("physical device was treated as model identity: %+v %v", result, err)
	}
}
