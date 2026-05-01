package main

import (
	"context"
	"flag"
	"fmt"
	"log"
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
		Addr:              fmt.Sprintf(":%d", port),
		Handler:           router,
		ReadHeaderTimeout: 2 * time.Second,
		ReadTimeout:       5 * time.Second,
		WriteTimeout:      10 * time.Second,
		IdleTimeout:       30 * time.Second,
	}

	go func() {
		log.Printf("Server starting on port %d", port)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Server failed: %v", err)
		}
	}()

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
	resp, err := http.Get("http://localhost:9999/ready")
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
