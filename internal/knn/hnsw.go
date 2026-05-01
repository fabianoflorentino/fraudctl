package knn

import (
	hnswgo "github.com/sunhailin-Leo/hnswlib-to-go"
	"github.com/fabianoflorentino/fraudctl/internal/model"
)

const (
	// K is the number of nearest neighbors to find.
	K = 5
)

// HNSWIndex implements KNN search using hnswlib (C++).
// Thread-safe for concurrent Predict calls.
type HNSWIndex struct {
	index      *hnswgo.HNSW
	fraudFlags []bool
	count      int
}

// NewHNSWIndex creates a new HNSW index for 14-dimensional vectors.
// Parameters optimized for Rinha 2026 (3M vectors, < 30s build):
//   - M=8: connections per node (faster build, slightly less accurate)
//   - efConstruction=100: build quality (balance speed vs accuracy)
//   - efSearch=50: search accuracy (set via SetEf)
func NewHNSWIndex() *HNSWIndex {
	index := hnswgo.New(14, 8, 100, 42, 3000000, hnswgo.SpaceL2)
	return &HNSWIndex{index: index}
}

// Build builds the HNSW index from reference vectors.
// Uses parallel insertion with 4 goroutines for faster build.
func (h *HNSWIndex) Build(vectors []model.Vector14, labels []bool) {
	h.fraudFlags = labels
	h.count = len(vectors)

	f32Vectors := make([][]float32, len(vectors))
	numericLabels := make([]uint32, len(vectors))

	for i := range vectors {
		v := vectors[i]
		f32Vectors[i] = []float32{v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11], v[12], v[13]}
		numericLabels[i] = uint32(i)
	}

	h.index.AddBatchPoints(f32Vectors, numericLabels, 8)
	h.index.SetEf(50)
}

// Predict finds the k nearest neighbors and returns the fraud probability.
// Thread-safe for concurrent calls.
func (h *HNSWIndex) Predict(query model.Vector14, k int) float64 {
	vec := []float32{query[0], query[1], query[2], query[3], query[4], query[5], query[6], query[7], query[8], query[9], query[10], query[11], query[12], query[13]}

	resultLabels, _ := h.index.SearchKNN(vec, k)

	fraudCount := 0
	for _, label := range resultLabels {
		idx := int(label)
		if idx < len(h.fraudFlags) && h.fraudFlags[idx] {
			fraudCount++
		}
	}

	return float64(fraudCount) / float64(len(resultLabels))
}

// Count returns the number of vectors in the index.
func (h *HNSWIndex) Count() int {
	return h.count
}

// Free releases the HNSW index memory.
func (h *HNSWIndex) Free() {
	if h.index != nil {
		h.index.Free()
		h.index = nil
	}
}
