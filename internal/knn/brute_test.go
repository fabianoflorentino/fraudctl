package knn

import (
	"encoding/binary"
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

// f32ToBytes converts a []float32 to []byte (little-endian).
func f32ToBytes(vecs []float32) []byte {
	buf := make([]byte, len(vecs)*4)
	for i, v := range vecs {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}
	return buf
}

// buildSmallIVF creates a minimal IVFIndex with two clusters using the float32 arena.
// cluster 0: 3 fraud vectors near [1,0,...], cluster 1: 3 legit vectors near [0,0,...].
func buildSmallIVF() *IVFIndex {
	const dim = 14
	nlist := 2

	centroids := make([]float32, nlist*dim)
	centroids[0] = 1.0 // centroid 0: dim[0]=1, rest 0
	// centroid 1: all 0s (default)

	// cluster 0: 3 fraud vectors
	fraudF := []float32{
		0.90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0.95, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		1.00, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	}
	fraudLabels := []byte{1, 1, 1}

	// cluster 1: 3 legit vectors
	legitF := []float32{
		0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0.10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0.02, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	}
	legitLabels := []byte{0, 0, 0}

	fraud0 := f32ToBytes(fraudF)
	legit0 := f32ToBytes(legitF)

	// arena: [fraud flat][fraud labels][legit flat][legit labels]
	arena := append(append(append(fraud0, fraudLabels...), legit0...), legitLabels...)

	fraudFlatOff := uint32(0)
	fraudLabelOff := uint32(len(fraud0))
	legitFlatOff := fraudLabelOff + uint32(len(fraudLabels))
	legitLabelOff := legitFlatOff + uint32(len(legit0))

	return &IVFIndex{
		nlist:     nlist,
		nprobe:    1,
		centroids: centroids,
		arena:     arena,
		descs: []ivfClusterDesc{
			{flatOff: fraudFlatOff, labelOff: fraudLabelOff, n: 3},
			{flatOff: legitFlatOff, labelOff: legitLabelOff, n: 3},
		},
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
		descs:     []ivfClusterDesc{{flatOff: 0, labelOff: 0, n: 0}},
		arena:     []byte{},
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
