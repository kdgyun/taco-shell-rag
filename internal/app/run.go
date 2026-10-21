package app

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"

	"github.com/kdgyun/taco-shell-rag/internal/catalog"
	"github.com/kdgyun/taco-shell-rag/internal/config"
	"github.com/kdgyun/taco-shell-rag/internal/inference"
	mcpserver "github.com/kdgyun/taco-shell-rag/internal/mcp"
	"github.com/kdgyun/taco-shell-rag/internal/retriever"
	httptransport "github.com/kdgyun/taco-shell-rag/internal/transport/http"
	"github.com/kdgyun/taco-shell-rag/internal/vectorstore"
	"github.com/modelcontextprotocol/go-sdk/mcp"
)

func Run(transport string) error {
	flags := flag.NewFlagSet("mcp-"+transport, flag.ContinueOnError)
	path := flags.String("config", "taco.toml", "configuration file")
	if err := flags.Parse(os.Args[1:]); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil
		}
		return err
	}
	if len(flags.Args()) != 0 {
		return errors.New("unexpected command arguments")
	}
	s, err := config.Load(*path)
	if err != nil {
		return err
	}
	embeddingToken, err := config.Credential(s.EmbeddingTokenEnv)
	if err != nil {
		return err
	}
	qdrantToken, err := config.Credential(s.QdrantTokenEnv)
	if err != nil {
		return err
	}
	svc := &retriever.Service{Ledger: &catalog.Catalog{Path: s.Catalog}, Embedder: inference.New(s.EmbeddingURL, embeddingToken), Index: vectorstore.New(s.QdrantURL, s.Collection, qdrantToken), Options: s.Retrieval}
	for _, src := range s.Sources {
		svc.Sources = append(svc.Sources, src.ID())
	}
	if s.Rerank {
		token, err := config.Credential(s.RerankerTokenEnv)
		if err != nil {
			return err
		}
		svc.Reranker = inference.New(s.RerankerURL, token)
	}
	server := mcpserver.New(svc)
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if transport == "stdio" {
		err := server.Run(ctx, &mcp.StdioTransport{})
		if ctx.Err() != nil {
			return nil
		}
		return err
	}
	token, err := config.Credential(s.TokenEnv)
	if err != nil {
		return err
	}
	handler, err := httptransport.New(server, token, s.AllowedHosts, s.AllowedOrigins)
	if err != nil {
		return err
	}
	httpServer := &http.Server{Addr: net.JoinHostPort(s.Host, strconv.Itoa(s.Port)), Handler: handler, ReadHeaderTimeout: 10 * time.Second, ReadTimeout: 2 * time.Minute, WriteTimeout: 2 * time.Minute, IdleTimeout: time.Minute}
	go func() {
		<-ctx.Done()
		shutdown, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		_ = httpServer.Shutdown(shutdown)
	}()
	fmt.Fprintln(os.Stderr, "MCP HTTP listening; inference and indexing remain separate services")
	err = httpServer.ListenAndServe()
	if errors.Is(err, http.ErrServerClosed) {
		return nil
	}
	return err
}
