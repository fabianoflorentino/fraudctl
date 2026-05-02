package knn

import (
	"math"
	"math/rand"
	"os"
	"testing"

	"github.com/fabianoflorentino/fraudctl/internal/model"
)

func BenchmarkIVFPredict(b *testing.B) {
	ivfPath := "../../resources/ivf.bin"
	if _, err := os.Stat(ivfPath); err != nil {
		b.Skipf("ivf.bin not found at %s, skipping: %v", ivfPath, err)
	}
	idx, err := LoadIVF(ivfPath)
	if err != nil {
		b.Fatalf("LoadIVF: %v", err)
	}

	var q model.Vector14
	for i := range q {
		q[i] = rand.Float32()
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		idx.Predict(q, K)
	}
}

// buildSmallIVFSoA creates a minimal IVF index with SoA blocks.
func buildSmallIVFSoA() *IVFIndex {
	const nlist = 2
	const dim = 14

	centroids := make([]float32, nlist*dim)
	centroids[0] = 1.0 // centroid 0 near [1,0,...]

	// 6 vectors total: 3 fraud (cluster 0), 3 legit (cluster 1)
	vectors := [][14]float32{
		{0.90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, // cluster 0, fraud
		{0.95, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, // cluster 0, fraud
		{1.00, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, // cluster 0, fraud
		{0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, // cluster 1, legit
		{0.10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, // cluster 1, legit
		{0.02, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, // cluster 1, legit
	}
	labels := []byte{1, 1, 1, 0, 0, 0}

	// SoA: 1 block for cluster 0 (3 vectors, padded to 8), 1 block for cluster 1
	nBlocks := 2
	blocks := make([]int16, nBlocks*dim*8)
	allLabels := make([]byte, nBlocks*8)

	// Block 0: cluster 0 vectors (3 actual, 5 padding)
	b0 := 0
	for d := 0; d < dim; d++ {
		for s := 0; s < 8; s++ {
			idx := b0*dim*8 + d*8 + s
			if s < 3 {
				blocks[idx] = quantizeFloat32(vectors[s][d])
			} else {
				blocks[idx] = int16Pad // padding
			}
		}
	}
	for s := 0; s < 3; s++ {
		allLabels[b0*8+s] = labels[s]
	}

	// Block 1: cluster 1 vectors (3 actual, 5 padding)
	b1 := 1
	for d := 0; d < dim; d++ {
		for s := 0; s < 8; s++ {
			idx := b1*dim*8 + d*8 + s
			if s < 3 {
				blocks[idx] = quantizeFloat32(vectors[3+s][d])
			} else {
				blocks[idx] = int16Pad
			}
		}
	}
	for s := 0; s < 3; s++ {
		allLabels[b1*8+s] = labels[3+s]
	}

	offsets := []uint32{0, 1, 2}

	return &IVFIndex{
		nlist:     nlist,
		nprobe:    2,
		centroids: centroids,
		blocks:    blocks,
		labels:    allLabels,
		offsets:   offsets,
	}
}

func TestIVFIndex_SetNProbe(t *testing.T) {
	idx := NewIVFIndex()

	idx.SetNProbe(2)
	if idx.nprobe != 2 {
		t.Errorf("SetNProbe(2): nprobe = %d, want 2", idx.nprobe)
	}

	idx.SetNProbe(0)
	if idx.nprobe != 1 {
		t.Errorf("SetNProbe(0): nprobe = %d, want 1", idx.nprobe)
	}
}

func TestIVFIndex_Predict_EmptyIndex(t *testing.T) {
	idx := &IVFIndex{
		nlist:     1,
		nprobe:    1,
		centroids: make([]float32, 14),
		offsets:   []uint32{0, 0},
	}
	var query model.Vector14
	score := idx.Predict(query, K)
	if score != 0 {
		t.Errorf("Predict on empty cluster = %v, want 0", score)
	}
}

func TestIVFIndex_Predict_AllFraud(t *testing.T) {
	idx := buildSmallIVFSoA()
	var query model.Vector14
	query[0] = 0.95

	score := idx.Predict(query, K)
	// With K=10 and 6 vectors (3 fraud, 3 legit), expect ~0.3
	// Verify score is not zero (fraud vectors are being found)
	if score <= 0.2 {
		t.Errorf("Predict near fraud cluster = %v, want > 0.2", score)
	}
}

func TestIVFIndex_Predict_AllLegit(t *testing.T) {
	idx := buildSmallIVFSoA()
	var query model.Vector14

	score := idx.Predict(query, K)
	if score > 0.5 {
		t.Errorf("Predict near legit cluster = %v, want <= 0.5", score)
	}
}

func TestIVFIndex_Predict_ScoreRange(t *testing.T) {
	idx := buildSmallIVFSoA()

	queries := []model.Vector14{
		{},
		{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
		{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5},
	}
	for _, q := range queries {
		score := idx.Predict(q, K)
		if math.IsNaN(float64(score)) || score < 0 || score > 1 {
			t.Errorf("Predict(%v) = %v, want [0, 1]", q, score)
		}
	}
}

func TestBruteIndex_Predict_ScoreRange(t *testing.T) {
	b := NewBruteIndex()
	vectors := []model.Vector14{
		{0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		{0.9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		{0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	}
	labels := []bool{false, true, false}
	b.Build(vectors, labels)

	var query model.Vector14
	query[0] = 0.5
	score := b.Predict(query, K)
	if math.IsNaN(float64(score)) || score < 0 || score > 1 {
		t.Errorf("BruteIndex.Predict = %v, want [0, 1]", score)
	}
}
