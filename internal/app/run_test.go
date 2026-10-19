package app

import (
	"os"
	"testing"
)

func TestHelpDoesNotRequireConfigurationOrCredentials(t *testing.T) {
	original := os.Args
	defer func() { os.Args = original }()
	os.Args = []string{"mcp-stdio", "--help"}
	if err := Run("stdio"); err != nil {
		t.Fatalf("help should succeed before loading configuration: %v", err)
	}
}
