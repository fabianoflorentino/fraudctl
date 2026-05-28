package main

import (
	"compress/gzip"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/fabianoflorentino/fraudctl/internal/gbdt"
)

func main() {
	refsGz := filepath.Join("resources", "references.json.gz")
	modelPath := filepath.Join("resources", "gbdt.bin")

	model, err := gbdt.Load(modelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Model: %d trees, %d features, sigmoid=%v\n", model.NumTrees, model.NumFeatures(), model.Sigmoid())

	f, err := os.Open(refsGz)
	if err != nil {
		fmt.Fprintf(os.Stderr, "open references: %v\n", err)
		os.Exit(1)
	}
	defer func() { _ = f.Close() }()
	gz, err := gzip.NewReader(f)
	if err != nil {
		fmt.Fprintf(os.Stderr, "gzip: %v\n", err)
		os.Exit(1)
	}
	defer func() { _ = gz.Close() }()

	dec := json.NewDecoder(gz)
	if _, err := dec.Token(); err != nil {
		fmt.Fprintf(os.Stderr, "json token: %v\n", err)
		os.Exit(1)
	}

	var total, bypassed int
	var fp, fn int
	const (
		confLow  = 0.08
		confHigh = 0.92
	)

	// Score distribution
	var hist [11]int
	fraudHist := [11]int{}

	for dec.More() {
		var raw struct {
			Vector [14]float64 `json:"vector"`
			Label  string      `json:"label"`
		}
		if err := dec.Decode(&raw); err != nil {
			break
		}
		total++

		var vec [14]float32
		for i, v := range raw.Vector {
			vec[i] = float32(v)
		}

		score := model.Predict(vec)
		isFraud := raw.Label == "fraud"

		bucket := int(score * 10)
		if bucket > 10 {
			bucket = 10
		}
		hist[bucket]++
		if isFraud {
			fraudHist[bucket]++
		}

		if score <= confLow || score >= confHigh {
			bypassed++
			if score <= confLow && isFraud {
				fn++
			}
			if score >= confHigh && !isFraud {
				fp++
			}
		}

		if total%500000 == 0 {
			fmt.Printf("Progress: %d/3000000\n", total)
		}
	}

	fmt.Printf("\n=== RF Prefilter Validation ===\n")
	fmt.Printf("Total references: %d\n", total)
	fmt.Printf("Bypassed:         %d (%.2f%%)\n", bypassed, float64(bypassed)/float64(total)*100)
	fmt.Printf("False Positives:  %d\n", fp)
	fmt.Printf("False Negatives:  %d\n", fn)
	if fp+fn == 0 {
		fmt.Printf("Accuracy: 100%% - ZERO errors!\n")
	}

	fmt.Printf("\nScore distribution:\n")
	fmt.Printf("Bucket        Total     Fraud   Fraud%%\n")
	for i := 0; i <= 10; i++ {
		lo := float32(i) / 10
		hi := float32(i+1) / 10
		pct := float64(fraudHist[i]) / float64(hist[i]) * 100
		fmt.Printf("[%.1f-%.1f)  %8d  %8d  %6.1f%%\n", lo, hi, hist[i], fraudHist[i], pct)
	}
}
