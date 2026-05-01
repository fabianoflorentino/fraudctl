package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"

	"github.com/fabianoflorentino/fraudctl/internal/dataset"
	"github.com/fabianoflorentino/fraudctl/internal/handler"
)

var (
	resourcesPath = flag.String("resources", "/resources", "Path to resources directory")
	healthCheck   = flag.Bool("healthcheck", false, "Run healthcheck and exit")
)

func main() {
	flag.Parse()

	if *healthCheck {
		if err := checkHealth(); err != nil {
			log.Fatalf("Healthcheck failed: %v", err)
		}
		log.Println("Healthcheck OK")
		return
	}

	port := 9999
	if p := os.Getenv("PORT"); p != "" {
		if parsed, err := strconv.Atoi(p); err == nil {
			port = parsed
		}
	}
	socketPath := os.Getenv("UNIX_SOCKET")

	log.Printf("Loading dataset from %s", *resourcesPath)
	ds, err := dataset.LoadDefault(*resourcesPath)
	if err != nil {
		log.Fatalf("Failed to load dataset: %v", err)
	}

	log.Printf("Dataset loaded: %d references (%d fraud, %d legit)",
		ds.Count(), ds.FraudCount(), ds.LegitCount())

	knnIndex := ds.Index()
	log.Printf("KNN index ready: %d vectors", knnIndex.Count())

	router := handler.NewRouter()
	router.Handle("/ready", handler.Ready)

	fraudHandler := handler.NewFraudScoreHandler(ds.Vectorizer(), knnIndex)
	router.Handle("/fraud-score", fraudHandler.Handle)

	srv := &http.Server{
		Handler:           router,
		ReadHeaderTimeout: 2 * time.Second,
		ReadTimeout:       5 * time.Second,
		WriteTimeout:      10 * time.Second,
		IdleTimeout:       30 * time.Second,
	}

	if socketPath != "" {
		// Remove stale socket if it exists.
		_ = os.Remove(socketPath)
		ln, err := net.Listen("unix", socketPath)
		if err != nil {
			log.Fatalf("Failed to listen on unix socket %s: %v", socketPath, err)
		}
		// Allow nginx (other users) to connect.
		if err := os.Chmod(socketPath, 0666); err != nil {
			log.Fatalf("Failed to chmod socket: %v", err)
		}
		log.Printf("Server starting on unix socket %s", socketPath)
		go func() {
			if err := srv.Serve(ln); err != nil && err != http.ErrServerClosed {
				log.Fatalf("Server failed: %v", err)
			}
		}()
	} else {
		srv.Addr = fmt.Sprintf(":%d", port)
		go func() {
			log.Printf("Server starting on port %d", port)
			if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
				log.Fatalf("Server failed: %v", err)
			}
		}()
	}

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	log.Println("Shutting down server...")

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		log.Fatalf("Server forced to shutdown: %v", err)
	}

	log.Println("Server exited properly")
}

func checkHealth() error {
	socketPath := os.Getenv("UNIX_SOCKET")

	var client *http.Client
	if socketPath != "" {
		client = &http.Client{
			Transport: &http.Transport{
				DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
					return (&net.Dialer{}).DialContext(ctx, "unix", socketPath)
				},
			},
		}
	} else {
		client = http.DefaultClient
	}

	resp, err := client.Get("http://localhost/ready")
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("unexpected status: %d", resp.StatusCode)
	}
	return nil
}

func init() {
	log.SetOutput(os.Stderr)
}
