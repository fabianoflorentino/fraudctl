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

// buildSmallIVF creates a minimal IVF index in v5 format (SoA blocks, bit-packed labels).
//
// 6 vectors total:
//   cluster 0 (centroid ≈ [1,0,…]): indices 0-2, all fraud
//   cluster 1 (centroid ≈ [0,0,…]): indices 3-5, all legit
func buildSmallIVF() *IVFIndex {
	const nlist = 2

	centroids := make([]float32, nlist*DIM)
	centroids[0] = 1.0 // centroid 0 near [1,0,...]

	rawVectors := [][DIM]float32{
		{0.90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, // fraud
		{0.95, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, // fraud
		{1.00, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, // fraud
		{0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, // legit
		{0.10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, // legit
		{0.02, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, // legit
	}
	fraudLabels := []bool{true, true, true, false, false, false}

	// cluster 0: vecs 0-2 (3 vecs → 1 block); cluster 1: vecs 3-5 (3 vecs → 1 block)
	clusterLists := [][]int{{0, 1, 2}, {3, 4, 5}}

	offsets := make([]uint32, nlist+1)
	blockOff := make([]uint32, nlist+1)
	totalBlocks := 0
	for ci := 0; ci < nlist; ci++ {
		totalBlocks += (len(clusterLists[ci]) + blockSize - 1) / blockSize
	}
	soaBlocks := make([]int16, totalBlocks*DIM*blockSize)

	n := len(rawVectors)
	bitLabels := make([]byte, (n+7)/8)

	vecPos := 0
	blkPos := 0
	for ci := 0; ci < nlist; ci++ {
		offsets[ci] = uint32(vecPos)
		blockOff[ci] = uint32(blkPos)
		list := clusterLists[ci]
		for lb := 0; lb < len(list); lb += blockSize {
			blockBase := blkPos * DIM * blockSize
			for lane := 0; lane < blockSize; lane++ {
				vi := -1
				if lb+lane < len(list) {
					vi = list[lb+lane]
				}
				for d := 0; d < DIM; d++ {
					idx := blockBase + d*blockSize + lane
					if vi >= 0 {
						soaBlocks[idx] = quantizeFloat32(rawVectors[vi][d])
					} else {
						soaBlocks[idx] = int16Pad
					}
				}
			}
			blkPos++
		}
		for _, vi := range list {
			if fraudLabels[vi] {
				bitLabels[vecPos>>3] |= 1 << uint(vecPos&7)
			}
			vecPos++
		}
	}
	offsets[nlist] = uint32(vecPos)
	blockOff[nlist] = uint32(blkPos)

	return &IVFIndex{
		nlist:     nlist,
		nprobe:    2,
		centroids: centroids,
		blocks:    soaBlocks,
		labels:    bitLabels,
		offsets:   offsets,
		blockOff:  blockOff,
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
		centroids: make([]float32, DIM),
		offsets:   []uint32{0, 0},
		blockOff:  []uint32{0, 0},
	}
	var query model.Vector14
	score := idx.Predict(query, K)
	if score != 0 {
		t.Errorf("Predict on empty cluster = %v, want 0", score)
	}
}

func TestIVFIndex_Predict_AllFraud(t *testing.T) {
	idx := buildSmallIVF()
	var query model.Vector14
	query[0] = 0.95

	score := idx.Predict(query, K)
	if score <= 0.2 {
		t.Errorf("Predict near fraud cluster = %v, want > 0.2", score)
	}
}

func TestIVFIndex_Predict_AllLegit(t *testing.T) {
	idx := buildSmallIVF()
	var query model.Vector14

	score := idx.Predict(query, K)
	if score > 0.5 {
		t.Errorf("Predict near legit cluster = %v, want <= 0.5", score)
	}
}

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
