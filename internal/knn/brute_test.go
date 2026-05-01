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

// buildSmallIVF creates a minimal IVFIndex with two clusters for unit tests.
// cluster 0: 3 fraud vectors near [1,0,...], cluster 1: 3 legit vectors near [0,0,...].
func buildSmallIVF() *IVFIndex {
	const dim = 14
	nlist := 2
	centroids := make([]float32, nlist*dim)
	// centroid 0: dim[0]=1, rest 0
	centroids[0] = 1.0
	// centroid 1: all 0s (default)

	// cluster 0: 3 fraud vectors — each 14 floats, dim[0] near 1
	fraud := make([]float32, 3*dim)
	fraud[0*dim+0] = 0.90
	fraud[1*dim+0] = 0.95
	fraud[2*dim+0] = 1.00

	// cluster 1: 3 legit vectors — each 14 floats, dim[0] near 0
	legit := make([]float32, 3*dim)
	legit[0*dim+0] = 0.05
	legit[1*dim+0] = 0.10
	legit[2*dim+0] = 0.02

	clusters := []ivfCluster{
		{flat: fraud, labels: []bool{true, true, true}},
		{flat: legit, labels: []bool{false, false, false}},
	}

	return &IVFIndex{
		nlist:     nlist,
		nprobe:    1,
		centroids: centroids,
		clusters:  clusters,
	}
}

// TestIVFIndex_SetNProbe verifies SetNProbe clamps to minimum 1 and sets the value.
func TestIVFIndex_SetNProbe(t *testing.T) {
	idx := NewIVFIndex()

	idx.SetNProbe(2)
	if idx.nprobe != 2 {
		t.Errorf("SetNProbe(2): nprobe = %d, want 2", idx.nprobe)
	}

	// clamped to minimum 1
	idx.SetNProbe(0)
	if idx.nprobe != 1 {
		t.Errorf("SetNProbe(0): nprobe = %d, want 1", idx.nprobe)
	}

	idx.SetNProbe(-5)
	if idx.nprobe != 1 {
		t.Errorf("SetNProbe(-5): nprobe = %d, want 1", idx.nprobe)
	}
}

// TestIVFIndex_Predict_EmptyIndex returns 0 when the index has no vectors.
func TestIVFIndex_Predict_EmptyIndex(t *testing.T) {
	idx := &IVFIndex{
		nlist:     1,
		nprobe:    1,
		centroids: make([]float32, 14),
		clusters:  []ivfCluster{{flat: nil, labels: nil}},
	}
	var query model.Vector14
	score := idx.Predict(query, K)
	if score != 0 {
		t.Errorf("Predict on empty cluster = %v, want 0", score)
	}
}

// TestIVFIndex_Predict_AllFraud returns 1.0 when all neighbours are fraud.
func TestIVFIndex_Predict_AllFraud(t *testing.T) {
	idx := buildSmallIVF()
	// query near centroid 0 (fraud cluster)
	var query model.Vector14
	query[0] = 0.95

	score := idx.Predict(query, K)
	if score <= 0.5 {
		t.Errorf("Predict near fraud cluster = %v, want > 0.5", score)
	}
}

// TestIVFIndex_Predict_AllLegit returns low score when all neighbours are legit.
func TestIVFIndex_Predict_AllLegit(t *testing.T) {
	idx := buildSmallIVF()
	// query near centroid 1 (legit cluster)
	var query model.Vector14

	score := idx.Predict(query, K)
	if score > 0.5 {
		t.Errorf("Predict near legit cluster = %v, want <= 0.5", score)
	}
}

// TestIVFIndex_Predict_ScoreRange ensures score is always in [0, 1].
func TestIVFIndex_Predict_ScoreRange(t *testing.T) {
	idx := buildSmallIVF()

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

// TestBruteIndex_Predict_ScoreRange ensures BruteIndex score is always in [0, 1].
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
