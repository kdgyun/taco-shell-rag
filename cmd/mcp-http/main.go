package main

import (
	"fmt"
	"github.com/kdgyun/taco-shell-rag/internal/app"
	"os"
)

func main() {
	if err := app.Run("http"); err != nil {
		fmt.Fprintln(os.Stderr, "mcp-http:", err)
		os.Exit(1)
	}
}
