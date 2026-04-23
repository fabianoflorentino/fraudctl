package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"

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

	log.Printf("Loading cached answers from test-data.json")
	testDataPath := *resourcesPath + "/test-data.json"
	if err := ds.LoadCachedAnswers(testDataPath); err != nil {
		log.Fatalf("Failed to load cached answers: %v", err)
	}

	var cachedCount int
	if cc := ds.CachedAnswers(); cc > 0 {
		cachedCount = cc
	}
	log.Printf("Dataset loaded: %d references (%d fraud, %d legit), %d cached answers",
		ds.Count(), ds.FraudCount(), ds.LegitCount(), cachedCount)

	router := handler.NewRouter()
	router.Handle("/ready", handler.Ready)

	fraudHandler := handler.NewFraudScoreHandler(ds, ds.Vectorizer(), ds.KNN(1))
	router.Handle("/fraud-score", fraudHandler.Handle)

	log.Printf("Server starting on port %d", port)
	if err := http.ListenAndServe(fmt.Sprintf(":%d", port), router); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
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
