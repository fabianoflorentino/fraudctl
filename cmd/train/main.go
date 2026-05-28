// Command train trains a decision tree from references.json.gz and exports
// it as a v2 GBDT binary (resources/gbdt.bin). Runs at docker build time.
//
// Usage: train -resources /path/to/resources
package main

import (
	"compress/gzip"
	"encoding/json"
	"flag"
	"log"
	"math/rand"
	"os"
	"path/filepath"

	"github.com/fabianoflorentino/fraudctl/internal/gbdt"
)

func main() {
	resources := flag.String("resources", "./resources", "Path to resources directory")
	maxDepth := flag.Int("depth", 8, "Max tree depth")
	sampleSize := flag.Int("sample", 200000, "Training sample size (0 = all)")
	seed := flag.Int64("seed", 42, "Random seed")
	minGain := flag.Float64("mingain", 0.001, "Minimum Gini gain to split")
	minSamples := flag.Int("minsamples", 500, "Minimum samples per leaf")
	numTrees := flag.Int("trees", 1, "Number of trees (1 = single tree, >1 = Random Forest)")
	bootstrap := flag.Int("bootstrap", 500000, "Bootstrap sample size per tree (0 = full set)")
	flag.Parse()

	refsGz := filepath.Join(*resources, "references.json.gz")
	outPath := filepath.Join(*resources, "gbdt.bin")

	log.Printf("Loading references from %s ...", refsGz)
	file, err := os.Open(refsGz)
	if err != nil {
		log.Fatalf("open: %v", err)
	}
	defer func() { _ = file.Close() }()

	gz, err := gzip.NewReader(file)
	if err != nil {
		log.Fatalf("gzip: %v", err)
	}
	defer func() { _ = gz.Close() }()

	dec := json.NewDecoder(gz)

	// Read opening bracket
	if _, err := dec.Token(); err != nil {
		log.Fatalf("json token: %v", err)
	}

	// Count total first for progress, or stream into memory
	// We need random access for sampling, so load all into memory.
	var allRefs []gbdt.Sample
	for dec.More() {
		var raw struct {
			Vector [14]float64 `json:"vector"`
			Label  string      `json:"label"`
		}
		if err := dec.Decode(&raw); err != nil {
			log.Fatalf("decode: %v", err)
		}
		s := gbdt.Sample{Label: raw.Label == "fraud"}
		for i, v := range raw.Vector {
			s.Vector[i] = float32(v)
		}
		allRefs = append(allRefs, s)
	}
	log.Printf("Loaded %d labeled vectors", len(allRefs))

	// Sample if needed
	trainSet := allRefs
	if *sampleSize > 0 && *sampleSize < len(allRefs) {
		rng := rand.New(rand.NewSource(*seed))
		rng.Shuffle(len(allRefs), func(i, j int) {
			allRefs[i], allRefs[j] = allRefs[j], allRefs[i]
		})
		trainSet = allRefs[:*sampleSize]
		log.Printf("Sampled %d vectors (seed=%d)", *sampleSize, *seed)
	}

	// Count fraud ratio
	var fraudCount int
	for _, s := range trainSet {
		if s.Label {
			fraudCount++
		}
	}
	log.Printf("Training set: %d samples, %d fraud (%.1f%%)",
		len(trainSet), fraudCount, float64(fraudCount)/float64(len(trainSet))*100)

	if *numTrees > 1 {
		log.Printf("Training Random Forest (%d trees, maxDepth=%d, bootstrap=%d, minGain=%g, minSamples=%d)...",
			*numTrees, *maxDepth, *bootstrap, *minGain, *minSamples)
		if err := gbdt.TrainRandomForest(trainSet, outPath, *numTrees, *maxDepth, *minSamples, *minGain, *seed+1, *bootstrap); err != nil {
			log.Fatalf("TrainRandomForest: %v", err)
		}
	} else {
		log.Printf("Training decision tree (maxDepth=%d, minGain=%g, minSamples=%d)...",
			*maxDepth, *minGain, *minSamples)
		if err := gbdt.TrainAndExport(trainSet, outPath, *maxDepth, *minSamples, *minGain); err != nil {
			log.Fatalf("TrainAndExport: %v", err)
		}
	}
	log.Printf("Model written to %s", outPath)

	// Verify: load and test
	model, err := gbdt.Load(outPath)
	if err != nil {
		log.Fatalf("load verification failed: %v", err)
	}
	log.Printf("Verification OK: %d trees, %d features, sigmoid=%v",
		model.NumTrees, model.NumFeatures(), model.Sigmoid())
}
