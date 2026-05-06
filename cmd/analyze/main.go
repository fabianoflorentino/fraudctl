package main

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/fabianoflorentino/fraudctl/internal/dataset"
	"github.com/fabianoflorentino/fraudctl/internal/model"
)

type Entry struct {
	Request          model.FraudScoreRequest `json:"request"`
	ExpectedApproved bool                    `json:"expected_approved"`
}

type TestData struct {
	Entries []Entry `json:"entries"`
}

func main() {
	ds, err := dataset.LoadDefault("/mnt/data/Projects/fraudctl/resources")
	if err != nil {
		panic(err)
	}
	vec := ds.Vectorizer()
	gbdt := ds.GBDT()
	if gbdt == nil {
		fmt.Println("no GBDT loaded")
		os.Exit(1)
	}

	f, _ := os.Open("/tmp/rinha-2026/rinha-2026-xgboost/test/test-data.json")
	defer f.Close()
	var td TestData
	json.NewDecoder(f).Decode(&td)

	scores := make([]float64, len(td.Entries))
	labels := make([]bool, len(td.Entries)) // true=fraud (should deny)
	for i, e := range td.Entries {
		v := vec.Vectorize(&e.Request)
		scores[i] = float64(gbdt.Predict(v))
		labels[i] = !e.ExpectedApproved
	}

	// Distribution buckets of 0.05
	buckets := make([]int, 20)
	for _, s := range scores {
		b := int(s * 20)
		if b >= 20 {
			b = 19
		}
		buckets[b]++
	}
	fmt.Println("GBDT score distribution (buckets of 0.05):")
	for i, c := range buckets {
		fmt.Printf("  [%.2f-%.2f]: %5d (%.1f%%)\n", float64(i)*0.05, float64(i+1)*0.05, c, float64(c)/float64(len(scores))*100)
	}

	// Fine-grained sweep: find threshold that eliminates IVF with minimum FN
	fmt.Println("\nFine sweep — threshold as single cutoff (approve if score < T, deny if score >= T):")
	fmt.Printf("  %-6s %6s %6s %6s\n", "T", "ivf%", "FP", "FN")
	for ti := 0; ti <= 100; ti++ {
		t := float64(ti) / 100.0
		fp, fn := 0, 0
		for i, s := range scores {
			isFraud := labels[i]
			if s < t {
				if isFraud {
					fn++
				}
			} else {
				if !isFraud {
					fp++
				}
			}
		}
		if fp+fn <= 5 { // only show viable candidates
			fmt.Printf("  %.2f   %5s %6d %6d\n", t, "0%", fp, fn)
		}
	}

	// Two-threshold sweep focused on eliminating IVF zone [0.40-0.70]
	fmt.Println("\nTwo-threshold: deny everything >= hi, approve everything < lo, IVF=0 forced:")
	fmt.Printf("  %-6s %-6s %6s %6s\n", "lo", "hi=lo", "FP", "FN")
	// Single threshold — what's the cost?
	for ti := 40; ti <= 70; ti++ {
		t := float64(ti) / 100.0
		fp, fn := 0, 0
		for i, s := range scores {
			isFraud := labels[i]
			if s < t {
				if isFraud {
					fn++
				}
			} else {
				if !isFraud {
					fp++
				}
			}
		}
		fmt.Printf("  cutoff=%.2f → FP=%d FN=%d\n", t, fp, fn)
	}
}
