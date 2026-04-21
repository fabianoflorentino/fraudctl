package main

import (
	"flag"
	"fmt"
	"fraudctl/internal/dataset"
	"fraudctl/internal/handler"
	"log"
	"net/http"
	"os"
	"strconv"
)

var (
	resourcesPath = flag.String("resources", "./resources", "Path to resources directory")
)

func main() {
	flag.Parse()

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

	router := handler.NewRouter()
	router.Handle("/ready", handler.Ready)

	fraudHandler := handler.NewFraudScoreHandler(ds.Vectorizer(), ds.KNN(1))
	router.Handle("/fraud-score", fraudHandler.Handle)

	log.Printf("Server starting on port %d", port)
	if err := http.ListenAndServe(fmt.Sprintf(":%d", port), router); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}

func init() {
	log.SetOutput(os.Stdout)
}
