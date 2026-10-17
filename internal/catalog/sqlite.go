package catalog

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"net/url"
	"os"
	"path/filepath"
	"reflect"
	"strings"

	"github.com/kdgyun/taco-shell-rag/internal/core"
	_ "modernc.org/sqlite"
)

type Catalog struct{ Path string }

func (c *Catalog) open(ctx context.Context) (*sql.DB, error) {
	if _, err := os.Stat(c.Path); err != nil {
		if os.IsNotExist(err) {
			return nil, core.ErrNotInitialized
		}
		return nil, errors.New("catalog is unreadable")
	}
	path, err := filepath.Abs(c.Path)
	if err != nil {
		return nil, errors.New("invalid catalog path")
	}
	u := url.URL{Scheme: "file", Path: path}
	q := url.Values{"mode": {"ro"}, "_pragma": {"busy_timeout(5000)", "query_only(1)"}}
	u.RawQuery = q.Encode()
	// immutable=1은 WAL의 새 게시를 놓친다. 읽기 전용 연결에서도 WAL을 정상적으로 읽는다.
	// Do not use immutable=1: read-only connections must still observe new WAL publications.
	db, err := sql.Open("sqlite", u.String())
	if err != nil {
		return nil, errors.New("cannot open catalog")
	}
	db.SetMaxOpenConns(1)
	if err = db.PingContext(ctx); err != nil {
		db.Close()
		return nil, errors.New("cannot open catalog")
	}
	return db, nil
}

func (c *Catalog) CheckSpace(ctx context.Context, embedding core.Embedding) error {
	db, err := c.open(ctx)
	if err != nil {
		return err
	}
	defer db.Close()
	var stored string
	if err := db.QueryRowContext(ctx, "SELECT value FROM settings WHERE key='embedding_space'").Scan(&stored); err != nil {
		return core.ErrNotInitialized
	}
	if stored != embedding.Fingerprint {
		return errors.New("embedding space changed; use a matching model or a new collection and catalog")
	}
	var descriptor string
	if db.QueryRowContext(ctx, "SELECT value FROM settings WHERE key='embedding_descriptor'").Scan(&descriptor) != nil {
		return errors.New("embedding identity is not recorded; run taco sync once to upgrade the catalog")
	}
	var expected core.EmbeddingSpace
	if json.Unmarshal([]byte(descriptor), &expected) != nil {
		return errors.New("catalog embedding identity is malformed")
	}
	actual := embedding.Space
	actual.Device = ""
	expected.Device = ""
	// 잘못 재사용된 지문만 믿지 않고 색인 당시 모델과 인코딩 설정도 대조한다.
	// Compare the indexed model and encoding too; do not trust a stale echoed fingerprint alone.
	if !reflect.DeepEqual(actual, expected) {
		return errors.New("embedding model or encoding differs from the indexed identity")
	}
	return nil
}

func readRecord(document, chunks string) (*core.Record, error) {
	var r core.Record
	if json.Unmarshal([]byte(document), &r.Document) != nil || json.Unmarshal([]byte(chunks), &r.Chunks) != nil || len(r.Chunks) == 0 {
		return nil, errors.New("catalog record is malformed")
	}
	return &r, nil
}

func (c *Catalog) Get(ctx context.Context, key string) (*core.Record, error) {
	db, err := c.open(ctx)
	if err != nil {
		return nil, err
	}
	defer db.Close()
	var document, chunks string
	err = db.QueryRowContext(ctx, "SELECT document,chunks FROM documents WHERE key=?", key).Scan(&document, &chunks)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, errors.New("catalog read failed")
	}
	return readRecord(document, chunks)
}

func (c *Catalog) Lexical(ctx context.Context, terms []string, all bool, limit int, sources []string) ([]core.Hit, bool, error) {
	db, err := c.open(ctx)
	if err != nil {
		return nil, false, err
	}
	defer db.Close()
	var schema string
	if db.QueryRowContext(ctx, "SELECT value FROM settings WHERE key='lexical_schema'").Scan(&schema) != nil || schema != "fts5-v1" {
		return nil, false, core.ErrLexicalUnavailable
	}
	if len(terms) == 0 || len(sources) == 0 {
		return []core.Hit{}, false, nil
	}
	quoted := make([]string, len(terms))
	for i, term := range terms {
		quoted[i] = `"` + strings.ReplaceAll(term, `"`, `""`) + `"`
	}
	join := " OR "
	if all {
		join = " AND "
	}
	args := []any{strings.Join(quoted, join)}
	marks := make([]string, len(sources))
	for i, src := range sources {
		marks[i] = "?"
		args = append(args, src)
	}
	// SQL 파라미터와 인용된 FTS 구문을 분리해 사용자의 검색어가 연산자가 되지 않게 한다.
	// Bind SQL values and quote FTS terms so user text never becomes an operator.
	query := `SELECT d.document,d.chunks,f.chunk_index,f.text,bm25(chunk_fts,0,0,0,0,1,2,1)
		FROM chunk_fts AS f JOIN documents AS d ON d.key=f.document_key AND d.version=f.version
		WHERE chunk_fts MATCH ? AND d.source IN (` + strings.Join(marks, ",") + `)
		ORDER BY bm25(chunk_fts,0,0,0,0,1,2,1),f.document_key,CAST(f.chunk_index AS INTEGER) LIMIT ?`
	scanLimit := min(4000, max(200, limit*10))
	args = append(args, scanLimit+1)
	rows, err := db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, false, errors.New("keyword index query failed; check catalog migration")
	}
	defer rows.Close()
	hits := []core.Hit{}
	count := 0
	for rows.Next() {
		count++
		if count > scanLimit {
			break
		}
		var document, chunks, text string
		var index int
		var rank float64
		if err := rows.Scan(&document, &chunks, &index, &text, &rank); err != nil {
			return nil, false, errors.New("keyword result is malformed")
		}
		r, err := readRecord(document, chunks)
		if err != nil {
			return nil, false, err
		}
		if index < 0 || index >= len(r.Chunks) || r.Chunks[index].Text != text {
			return nil, false, errors.New("keyword index differs from published content; rerun sync")
		}
		body := text + "\n" + r.Document.Title + "\n" + r.Document.Path
		matched := 0
		for _, term := range terms {
			if core.ContainsTerm(body, term) {
				matched++
			}
		}
		if (all && matched != len(terms)) || matched == 0 {
			continue
		}
		hits = append(hits, core.Hit{Document: r.Document, Chunk: r.Chunks[index], Score: -rank, MatchedBy: []string{"keyword"}})
		if len(hits) > limit {
			break
		}
	}
	if err := rows.Err(); err != nil {
		return nil, false, errors.New("keyword index read failed")
	}
	truncated := count > scanLimit || len(hits) > limit
	if len(hits) > limit {
		hits = hits[:limit]
	}
	return hits, truncated, nil
}

func (c *Catalog) Status(ctx context.Context) (map[string]any, error) {
	db, err := c.open(ctx)
	if errors.Is(err, core.ErrNotInitialized) {
		return map[string]any{"initialized": false, "documents": 0, "sources": []any{}, "pending_cleanup": 0}, nil
	}
	if err != nil {
		return nil, err
	}
	defer db.Close()
	var documents, retired int
	if db.QueryRowContext(ctx, "SELECT COUNT(*) FROM documents").Scan(&documents) != nil || db.QueryRowContext(ctx, "SELECT COUNT(*) FROM retired").Scan(&retired) != nil {
		return nil, core.ErrNotInitialized
	}
	rows, err := db.QueryContext(ctx, "SELECT id,checkpoint,status,attempted_at,completed_at,documents,error FROM sources ORDER BY id")
	if err != nil {
		return nil, errors.New("cannot read source checkpoints")
	}
	defer rows.Close()
	sources := []map[string]any{}
	for rows.Next() {
		var id, status string
		var checkpoint, attempted, completed, failure sql.NullString
		var n int
		if err := rows.Scan(&id, &checkpoint, &status, &attempted, &completed, &n, &failure); err != nil {
			return nil, errors.New("cannot read source checkpoints")
		}
		sources = append(sources, map[string]any{"id": id, "checkpoint": nullable(checkpoint), "status": status, "attempted_at": nullable(attempted), "completed_at": nullable(completed), "documents": n, "error": nullable(failure)})
	}
	if rows.Err() != nil {
		return nil, errors.New("cannot read source checkpoints")
	}
	return map[string]any{"initialized": true, "documents": documents, "sources": sources, "pending_cleanup": retired}, nil
}

func nullable(value sql.NullString) any {
	if value.Valid {
		return value.String
	}
	return nil
}
