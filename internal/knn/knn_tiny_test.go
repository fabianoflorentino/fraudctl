package knn

import (
	"math"
	"testing"

	"github.com/fabianoflorentino/fraudctl/internal/model"
)

func TestBBoxMayImprove_LastDimExceeds(t *testing.T) {
	bboxMin := make([]int16, DIM)
	bboxMax := make([]int16, DIM)
	for d := 0; d < DIM; d++ {
		bboxMin[d] = 0
		bboxMax[d] = 10
	}

	var qi [DIM]int16
	// All early dims inside
	for d := 0; d < DIM; d++ {
		qi[d] = 5
	}
	// Only dim 4 exceeds, but by enough to push over worstDist
	qi[4] = 100

	if bboxMayImprove(bboxMin, bboxMax, 0, qi, 1000) {
		t.Error("bboxMayImprove should return false when dim 4 alone exceeds worstDist")
	}
}

func TestBBoxMayImprove_Dim4Alone(t *testing.T) {
	bboxMin := make([]int16, DIM)
	bboxMax := make([]int16, DIM)
	for d := 0; d < DIM; d++ {
		bboxMin[d] = 0
		bboxMax[d] = 10
	}

	var qi [DIM]int16
	// All dims inside except dim 4
	for d := 0; d < DIM; d++ {
		qi[d] = 5
	}
	qi[4] = 100

	// With large worstDist, should still return true since dims 0-3 might be close enough
	result := bboxMayImprove(bboxMin, bboxMax, 0, qi, 1000000)
	if !result {
		t.Error("should return true when early dims are inside even if dim 4 is far")
	}
}

func TestBrutePredict_NoFraudFlags(t *testing.T) {
	flat := []float32{0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	query := model.Vector14{0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}

	score := brutePredict(flat, nil, 1, query, 1)
	if score != 0 {
		t.Errorf("brutePredict with nil flags = %v, want 0", score)
	}
}

func TestBrutePredict_OutOfRangeIdx(t *testing.T) {
	flat := []float32{0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	flags := []bool{false}
	query := model.Vector14{0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}

	score := brutePredict(flat, flags, 1, query, 1)
	if score != 0 {
		t.Errorf("brutePredict = %v, want 0", score)
	}
}

func TestPredictRaw_AllFastProbesEmpty(t *testing.T) {
	centroids := make([]float32, 2*DIM)
	centroids[0] = 0.1
	centroids[DIM] = 0.9

	idx := &IVFIndex{
		nlist:     2,
		nprobe:    6,
		centroids: centroids,
		vectors:   make([]int16, 6*DIM),
		labels:    make([]byte, 1),
		offsets:   []uint32{0, 0, 6},
	}

	raw := idx.PredictRaw(model.Vector14{0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 6)
	if raw < 0 || raw > 5 {
		t.Errorf("PredictRaw = %d, want [0,5]", raw)
	}
}

func TestPredictRaw_FastProbesOnly(t *testing.T) {
	centroids := make([]float32, 2*DIM)
	centroids[0] = 0.1
	centroids[DIM] = 0.9

	vectors := make([]int16, 6*DIM)
	labels := make([]byte, 1)
	labels[0] = 0b00000001
	for i := 0; i < 3; i++ {
		vectors[i*DIM] = 1000
	}
	for i := 3; i < 6; i++ {
		vectors[i*DIM] = 9000
	}

	idx := &IVFIndex{
		nlist:     2,
		nprobe:    6,
		centroids: centroids,
		vectors:   vectors,
		labels:    labels,
		offsets:   []uint32{0, 3, 6},
	}

	// Query near cluster 0 -> fraud=3 or 0 -> no pass2 triggered, or fraud=2 -> pass2 triggered
	raw := idx.PredictRaw(model.Vector14{0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 6)
	if raw < 0 || raw > 5 {
		t.Errorf("PredictRaw = %d, want [0,5]", raw)
	}
}

func TestBBoxMayImprove_Dim1Edge(t *testing.T) {
	bboxMin := make([]int16, DIM)
	bboxMax := make([]int16, DIM)
	for d := 0; d < DIM; d++ {
		bboxMin[d] = 0
		bboxMax[d] = 10
	}

	var qi [DIM]int16
	for d := 0; d < DIM; d++ {
		qi[d] = 5
	}
	qi[1] = 15

	if !bboxMayImprove(bboxMin, bboxMax, 0, qi, math.MaxUint64) {
		t.Error("should return true at dim 1 boundary")
	}
}
