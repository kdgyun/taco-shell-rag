package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestDefaultsAndRelativeCatalog(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "rag.toml")
	if err := os.WriteFile(path, []byte("[[sources]]\nid='notes'\ntype='folder'\npath='docs'\n"), 0600); err != nil {
		t.Fatal(err)
	}
	s, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if s.EmbeddingURL != "http://inference:8000" || s.Retrieval.Mode != "hybrid" || s.Catalog != filepath.Join(dir, "data/catalog.sqlite3") || !s.Rerank {
		t.Fatalf("defaults: %+v", s)
	}
}

func TestRejectsInvalidConfigurationWithoutEchoingSecrets(t *testing.T) {
	for _, body := range []string{
		"embedding_url='https://secret:token@example.invalid'\n",
		"embedding_url='http://inference:8000'\ninference_url='http://other:8000'\n",
		"unknown_option='private-value'\n",
		"[retrieval]\ndense_weight=nan\n",
		"[retrieval]\nkeyword_weight=inf\n",
		"[retrieval]\ncandidate_limit=10000\n",
	} {
		path := filepath.Join(t.TempDir(), "rag.toml")
		body += "\n[[sources]]\nid='notes'\ntype='folder'\npath='docs'\n"
		if err := os.WriteFile(path, []byte(body), 0600); err != nil {
			t.Fatal(err)
		}
		if _, err := Load(path); err == nil {
			t.Fatalf("invalid configuration accepted: %s", body)
		}
	}
}

func TestCollectorExtensionsDoNotRequireGoChanges(t *testing.T) {
	path := filepath.Join(t.TempDir(), "rag.toml")
	body := "[[sources]]\nid='custom'\ntype='custom-tracker'\ncustom_setting='owned-by-python'\n[sources.pagination]\npage_size=10\n"
	if err := os.WriteFile(path, []byte(body), 0600); err != nil {
		t.Fatal(err)
	}
	s, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if s.Sources[0].ID() != "custom" {
		t.Fatal("source identity was lost")
	}
}
