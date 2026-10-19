package main

import (
	"fmt"
	"github.com/kdgyun/taco-shell-rag/internal/app"
	"os"
)

func main() {
	if err := app.Run("stdio"); err != nil {
		fmt.Fprintln(os.Stderr, "mcp-stdio:", err)
		os.Exit(1)
	}
}
