package knn

import (
	"compress/gzip"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"

	"github.com/fabianoflorentino/fraudctl/internal/model"
)

type BruteIndex struct {
	flat       []float32
	fraudFlags []bool
	count      int
}

func NewBruteIndex() *BruteIndex { return &BruteIndex{} }

func (b *BruteIndex) Build(vectors []model.Vector14, labels []bool) {
	n := len(vectors)
	b.count = n
	b.fraudFlags = labels
	b.flat = make([]float32, n*14)
	for i, v := range vectors {
		copy(b.flat[i*14:], v[:])
	}
}

func (b *BruteIndex) BuildFromGzip(path string, capacity int) error {
	f, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("open gzip: %w", err)
	}
	defer f.Close()
	gz, err := gzip.NewReader(f)
	if err != nil {
		return fmt.Errorf("gzip reader: %w", err)
	}
	defer gz.Close()

	b.flat = make([]float32, 0, capacity*14)
	b.fraudFlags = make([]bool, 0, capacity)

	dec := json.NewDecoder(gz)
	if _, err := dec.Token(); err != nil {
		return fmt.Errorf("json open bracket: %w", err)
	}

	var entry struct {
		Vector []float64 `json:"vector"`
		Label  string    `json:"label"`
	}
	for dec.More() {
		entry.Vector = entry.Vector[:0]
		if err := dec.Decode(&entry); err != nil {
			if err == io.EOF {
				break
			}
			return fmt.Errorf("json decode: %w", err)
		}
		for i := 0; i < 14; i++ {
			if i < len(entry.Vector) {
				b.flat = append(b.flat, float32(entry.Vector[i]))
			} else {
				b.flat = append(b.flat, 0)
			}
		}
		b.fraudFlags = append(b.fraudFlags, entry.Label == "fraud")
	}
	b.count = len(b.fraudFlags)
	return nil
}

func (b *BruteIndex) Predict(query model.Vector14, k int) float64 {
	return brutePredict(b.flat, b.fraudFlags, b.count, query, k)
}

// PredictRaw for BruteIndex ignores nprobe (exact search has no concept of it).
// Returns fraud count out of k=5 neighbors to match IVFIndex behavior.
func (b *BruteIndex) PredictRaw(query model.Vector14, _ int) int {
	const k = 5
	score := brutePredict(b.flat, b.fraudFlags, b.count, query, k)
	return int(math.Round(score * float64(k)))
}

// NProbe for BruteIndex returns 0 — not applicable for exact search.
func (b *BruteIndex) NProbe() int { return 0 }

func (b *BruteIndex) Count() int { return b.count }

func (b *BruteIndex) FraudCount() int {
	n := 0
	for _, f := range b.fraudFlags {
		if f {
			n++
		}
	}
	return n
}
