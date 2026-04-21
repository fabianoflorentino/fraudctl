package main

import (
	"flag"
	"fraudctl/internal/dataset"
	"fraudctl/internal/handler"
	"log"
	"net/http"
	"os"
)

var (
	resourcesPath = flag.String("resources", "./resources", "Path to resources directory")
	port          = flag.Int("port", 9999, "Server port")
)

func main() {
	flag.Parse()

	log.Printf("Loading dataset from %s", *resourcesPath)
	ds, err := dataset.LoadDefault(*resourcesPath)
	if err != nil {
		log.Fatalf("Failed to load dataset: %v", err)
	}

	log.Printf("Dataset loaded: %d references (%d fraud, %d legit)",
		ds.Count(), ds.FraudCount(), ds.LegitCount())

	router := handler.NewRouter()
	router.Handle("/ready", handler.Ready)

	log.Printf("Server starting on port %d", *port)
	if err := http.ListenAndServe(":9999", router); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}

func init() {
	log.SetOutput(os.Stdout)
}
