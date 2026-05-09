package knn

import (
	"math"
	"testing"

	"github.com/fabianoflorentino/fraudctl/internal/model"
)

func TestBruteAVX2Index_FraudCount_NilLabels(t *testing.T) {
	idx := &BruteAVX2Index{
		N:    0,
		data: nil,
	}
	if idx.FraudCount() != 0 {
		t.Errorf("FraudCount = %d, want 0", idx.FraudCount())
	}
}

func TestBBoxMayImprove_Dim3Far(t *testing.T) {
	bboxMin := make([]int16, DIM)
	bboxMax := make([]int16, DIM)
	for d := 0; d < DIM; d++ {
		bboxMin[d] = 0
		bboxMax[d] = 10
	}

	var qi [DIM]int16
	for d := 0; d < DIM; d++ {
		qi[d] = 1000
	}

	if bboxMayImprove(bboxMin, bboxMax, 0, qi, 100) {
		t.Error("should return false when query is far from bbox")
	}
}

func TestBBoxMayImprove_SomeDimsInside(t *testing.T) {
	bboxMin := make([]int16, DIM)
	bboxMax := make([]int16, DIM)
	for d := 0; d < DIM; d++ {
		bboxMin[d] = 0
		bboxMax[d] = 10
	}

	var qi [DIM]int16
	qi[0] = 5
	qi[1] = 5
	qi[2] = 5
	qi[3] = 1000
	qi[4] = 1000

	// Some dims inside, some outside - should still return true with large worstDist
	if !bboxMayImprove(bboxMin, bboxMax, 0, qi, 10000000) {
		t.Error("should return true when some dims are inside bbox")
	}
}

func TestBruteAVX2Index_Predict_KEqualsZero(t *testing.T) {
	idx := createSmallBruteAVX2(t)
	score := idx.Predict(model.Vector14{0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 0)
	if score < 0 || score > 1 {
		t.Errorf("Predict with k=0 = %v, want [0,1]", score)
	}
}

func TestBruteAVX2Index_Predict_AllFar(t *testing.T) {
	N := 3
	data := make([]int16, N*DIM)
	for i := 0; i < N; i++ {
		for d := 0; d < DIM; d++ {
			data[i*DIM+d] = 10000
		}
	}
	labels := []byte{0, 1, 0}
	idx := &BruteAVX2Index{data: data, labels: labels, N: N}

	var query model.Vector14
	for d := 0; d < DIM; d++ {
		query[d] = 0
	}

	score := idx.Predict(query, 3)
	if score < 0 || score > 1 {
		t.Errorf("Predict = %v, want [0,1]", score)
	}
}

func TestPredictRaw_BoundaryZone(t *testing.T) {
	idx := buildSmallIVF()
	idx.SetNProbe(2)

	raw := idx.PredictRaw(model.Vector14{0.95, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 2)
	if raw < 0 || raw > 5 {
		t.Errorf("PredictRaw = %d, want [0,5]", raw)
	}
}

func TestBrutePredict_EmptyK(t *testing.T) {
	score := brutePredict(nil, nil, 0, model.Vector14{}, 0)
	if score != 0 {
		t.Errorf("brutePredict with k=0 = %v, want 0", score)
	}
}

func TestBrutePredict_AllSameDistance(t *testing.T) {
	flat := []float32{
		0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	}
	flags := []bool{true, false, true}
	query := model.Vector14{0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}

	score := brutePredict(flat, flags, 3, query, 3)
	expected := 2.0 / 3.0
	if math.Abs(score-expected) > 0.0001 {
		t.Errorf("brutePredict = %v, want %v", score, expected)
	}
}

func TestLoadIVF_InvalidPath(t *testing.T) {
	_, err := LoadIVF("/nonexistent/ivf.bin")
	if err == nil {
		t.Fatal("expected error for nonexistent path")
	}
}

func TestScanCluster_AllWorse(t *testing.T) {
	nVecs := 32
	vectors := make([]int16, nVecs*DIM)
	labels := make([]byte, (nVecs+7)/8)
	for i := 0; i < nVecs; i++ {
		for d := 0; d < DIM; d++ {
			vectors[i*DIM+d] = 10000
		}
	}

	var qi [DIM]int16
	h := newTopK5()
	scanCluster(vectors, labels, 0, nVecs, qi, &h)
}
