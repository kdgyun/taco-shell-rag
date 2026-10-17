package catalog

import (
	"context"
	"database/sql"
	"path/filepath"
	"testing"
)

func TestReadOnlyConnectionObservesCommittedWALChanges(t *testing.T) {
	path := filepath.Join(t.TempDir(), "catalog.sqlite3")
	writer, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatal(err)
	}
	defer writer.Close()
	for _, statement := range []string{"PRAGMA journal_mode=WAL", "CREATE TABLE sample (value INTEGER)", "INSERT INTO sample VALUES (1)"} {
		if _, err := writer.Exec(statement); err != nil {
			t.Fatal(err)
		}
	}
	reader, err := (&Catalog{Path: path}).open(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	defer reader.Close()
	if _, err := reader.Exec("UPDATE sample SET value=99"); err == nil {
		t.Fatal("read-only catalog accepted a write")
	}
	if _, err := writer.Exec("UPDATE sample SET value=2"); err != nil {
		t.Fatal(err)
	}
	var value int
	if err := reader.QueryRow("SELECT value FROM sample").Scan(&value); err != nil {
		t.Fatal(err)
	}
	if value != 2 {
		t.Fatalf("reader missed committed WAL data: %d", value)
	}
}
