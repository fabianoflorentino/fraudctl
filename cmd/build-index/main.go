// build-index pre-computes an IVF index from references.json.gz.
// Runs at docker build time so the API startup only needs to load the file.
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
	outPath := filepath.Join(*resources, "ivf.bin")

	log.Printf("Building IVF index: nlist=%d iterations=%d", *nlist, *iterations)

	if err := knn.BuildIVF(refsGz, outPath, *nlist, *iterations); err != nil {
		log.Fatalf("BuildIVF failed: %v", err)
	}

	log.Printf("IVF index written to %s", outPath)
}
