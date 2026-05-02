// build-index pre-computes IVF and brute-force indexes from references.json.gz.
// Runs at docker build time so the API startup only needs to load the files.
//
// Usage: build-index -resources /path/to/resources -nlist 500 -iterations 20
package main

import (
	"flag"
	"log"
	"path/filepath"

	"github.com/fabianoflorentino/fraudctl/internal/knn"
)

func main() {
	resources := flag.String("resources", "/resources", "Path to resources directory")
	nlist := flag.Int("nlist", 500, "Number of IVF clusters")
	iterations := flag.Int("iterations", 20, "K-means iterations")
	flag.Parse()

	refsGz := filepath.Join(*resources, "references.json.gz")

	// Build IVF index
	ivfPath := filepath.Join(*resources, "ivf.bin")
	log.Printf("Building IVF index: nlist=%d iterations=%d", *nlist, *iterations)
	if err := knn.BuildIVF(refsGz, ivfPath, *nlist, *iterations); err != nil {
		log.Fatalf("BuildIVF failed: %v", err)
	}
	log.Printf("IVF index written to %s", ivfPath)

	// Build brute-force index
	brutePath := filepath.Join(*resources, "brute.bin")
	log.Printf("Building brute-force index...")
	if err := knn.BuildBrute(refsGz, brutePath); err != nil {
		log.Fatalf("BuildBrute failed: %v", err)
	}
	log.Printf("Brute index written to %s", brutePath)
}
