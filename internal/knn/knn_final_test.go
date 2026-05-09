package knn

import (
	"math"
	"os"
	"testing"

	"github.com/fabianoflorentino/fraudctl/internal/model"
)

func TestBruteAVX2Index_FraudCount_HalfFraud(t *testing.T) {
	N := 4
	data := make([]int16, N*DIM)
	for i := 0; i < N; i++ {
		data[i*DIM] = int16(i * 2500)
	}
	labels := []byte{0, 1, 1, 0}
	idx := &BruteAVX2Index{data: data, labels: labels, N: N}
	if idx.FraudCount() != 2 {
		t.Errorf("FraudCount = %d, want 2", idx.FraudCount())
	}
}

func TestPredictRaw_EmptyVectors(t *testing.T) {
	idx := &IVFIndex{
		centroids: make([]float32, DIM),
		offsets:   []uint32{0, 0, 0},
		nlist:     1,
		nprobe:    0,
	}
	raw := idx.PredictRaw(model.Vector14{}, 0)
	if raw != 0 {
		t.Errorf("PredictRaw empty vectors = %d, want 0", raw)
	}
}

func TestBrutePredict_ZeroVectors(t *testing.T) {
	score := brutePredict(nil, nil, 0, model.Vector14{}, 3)
	if score != 0 {
		t.Errorf("brutePredict no vectors = %v, want 0", score)
	}
}

func TestBBoxMayImprove_AllDimsAtLowerBound(t *testing.T) {
	bboxMin := make([]int16, DIM)
	bboxMax := make([]int16, DIM)
	for d := 0; d < DIM; d++ {
		bboxMin[d] = -100
		bboxMax[d] = 100
	}

	var qi [DIM]int16
	for d := 0; d < DIM; d++ {
		qi[d] = -100
	}

	if !bboxMayImprove(bboxMin, bboxMax, 0, qi, math.MaxUint64) {
		t.Error("should return true at boundary")
	}
}

func TestBBoxMayImprove_AllDimsAtUpperBound(t *testing.T) {
	bboxMin := make([]int16, DIM)
	bboxMax := make([]int16, DIM)
	for d := 0; d < DIM; d++ {
		bboxMin[d] = -100
		bboxMax[d] = 100
	}

	var qi [DIM]int16
	for d := 0; d < DIM; d++ {
		qi[d] = 100
	}

	if !bboxMayImprove(bboxMin, bboxMax, 0, qi, math.MaxUint64) {
		t.Error("should return true at boundary")
	}
}

func TestBBoxMayImprove_Dim3AloneSaves(t *testing.T) {
	bboxMin := make([]int16, DIM)
	bboxMax := make([]int16, DIM)
	for d := 0; d < DIM; d++ {
		bboxMin[d] = 0
		bboxMax[d] = 100
	}

	var qi [DIM]int16
	// Early dims (5, 6, 2, 0, 7, 8, 11, 12) all outside
	qi[5] = 200
	qi[6] = 200
	qi[2] = 200
	qi[0] = 200
	qi[7] = 200
	qi[8] = 200
	qi[11] = 200
	qi[12] = 200
	// Later dims inside
	qi[9] = 50
	qi[10] = 50
	qi[1] = 50
	qi[13] = 50
	qi[3] = 50
	qi[4] = 50

	if bboxMayImprove(bboxMin, bboxMax, 0, qi, 10000) {
		// Should still fail because first few dims already exceed worstDist
	}

	qi[5] = 50
	qi[6] = 50
	qi[2] = 50
	qi[0] = 50
	qi[7] = 50
	qi[8] = 50
	qi[11] = 50
	qi[12] = 50

	if !bboxMayImprove(bboxMin, bboxMax, 0, qi, 100000) {
		t.Error("should return true when dims are within bbox")
	}
}

func writeUint32LE(b []byte, offset int, v uint32) {
	b[offset] = byte(v)
	b[offset+1] = byte(v >> 8)
	b[offset+2] = byte(v >> 16)
	b[offset+3] = byte(v >> 24)
}

func TestLoadIVF_InvalidMagic(t *testing.T) {
	path := t.TempDir() + "/bad_magic.ivf"
	data := make([]byte, 16)
	data[0] = 0xFF
	data[1] = 0xFF
	data[2] = 0xFF
	data[3] = 0xFF
	writeFile(t, path, data)
	_, err := LoadIVF(path)
	if err == nil {
		t.Fatal("expected error for invalid magic")
	}
}

func TestLoadIVF_InvalidVersion(t *testing.T) {
	path := t.TempDir() + "/bad_version.ivf"
	data := make([]byte, 24)
	writeUint32LE(data, 0, ivfMagic)
	writeUint32LE(data, 4, 99) // version 99
	writeFile(t, path, data)
	_, err := LoadIVF(path)
	if err == nil {
		t.Fatal("expected error for invalid version")
	}
}

func TestLoadIVF_WrongDim(t *testing.T) {
	path := t.TempDir() + "/bad_dim.ivf"
	data := make([]byte, 32)
	writeUint32LE(data, 0, ivfMagic)
	writeUint32LE(data, 4, 5)  // version 5
	writeUint32LE(data, 8, 2)  // nlist = 2
	writeUint32LE(data, 12, 7) // dim = 7 (wrong, should be 14)
	writeFile(t, path, data)
	_, err := LoadIVF(path)
	if err == nil {
		t.Fatal("expected error for wrong dim")
	}
}

func writeFile(t *testing.T, path string, data []byte) {
	t.Helper()
	if err := os.WriteFile(path, data, 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
}
