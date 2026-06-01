// build-index pre-computes the Grid index from references.json.gz.
// Runs at docker build time so the API startup only needs to load the files.
//
// Usage: build-index -resources /path/to/resources
package main

import (
	"flag"
	"log"
	"path/filepath"

	"github.com/fabianoflorentino/fraudctl/internal/knn"
)

func main() {
	resources := flag.String("resources", "./resources", "Path to resources directory")
	flag.Parse()

	refsGz := filepath.Join(*resources, "references.json.gz")
	gridPath := filepath.Join(*resources, "grid.bin")

	log.Println("Building grid index...")
	if err := knn.BuildGrid(refsGz, gridPath); err != nil {
		log.Fatalf("BuildGrid failed: %v", err)
	}
	log.Printf("Grid index written to %s", gridPath)
}
