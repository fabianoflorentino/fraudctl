package knn

import (
	"bytes"
	"compress/gzip"
	"encoding/json"
	"math"
	"os"
	"sync"
	"testing"
	"time"

	"github.com/fabianoflorentino/fraudctl/internal/model"
)

func TestGridIndex_CountFraudCount(t *testing.T) {
	idx := &GridIndex{
		n:      3,
		labels: []byte{0b00000101},
	}
	if idx.Count() != 3 {
		t.Errorf("Count = %d, want 3", idx.Count())
	}
	if idx.FraudCount() != 2 {
		t.Errorf("FraudCount = %d, want 2", idx.FraudCount())
	}
}

func TestGridIndex_NProbe(t *testing.T) {
	idx := &GridIndex{}
	if idx.NProbe() != 0 {
		t.Errorf("NProbe = %d, want 0", idx.NProbe())
	}
}

func TestGridPartitionKey(t *testing.T) {
	tests := []struct {
		name string
		v    [DIM]int16
		want uint32
	}{
		{
			name: "all zeros",
			v:    [DIM]int16{},
			want: 0,
		},
		{
			name: "hour bucket 2",
			v:    [DIM]int16{0, 0, 0, 3000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			want: 2,
		},
		{
			name: "hour bucket 7",
			v:    [DIM]int16{0, 0, 0, 9000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			want: 7,
		},
		{
			name: "installments bucket 2",
			v:    [DIM]int16{0, 6000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			want: 2 << 3,
		},
		{
			name: "mcc high",
			v:    [DIM]int16{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5000, 0},
			want: 1 << 5,
		},
		{
			name: "mcc low",
			v:    [DIM]int16{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1000, 0},
			want: 0,
		},
		{
			name: "mcc threshold exactly 3000",
			v:    [DIM]int16{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3000, 0},
			want: 0,
		},
		{
			name: "online positive",
			v:    [DIM]int16{0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
			want: 1 << 6,
		},
		{
			name: "unknown merchant",
			v:    [DIM]int16{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
			want: 1 << 7,
		},
		{
			name: "all flags",
			v:    [DIM]int16{0, 6000, 0, 9000, 0, 0, 0, 0, 0, 1, 0, 1, 5000, 0},
			want: (7 << 0) | (2 << 3) | (1 << 5) | (1 << 6) | (1 << 7),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := gridPartitionKey(&tt.v)
			if got != tt.want {
				t.Errorf("gridPartitionKey = %d (0b%08b), want %d (0b%08b)", got, got, tt.want, tt.want)
			}
		})
	}
}

func TestGridBBoxLowerBound(t *testing.T) {
	tests := []struct {
		name string
		q    [DIM]int16
		min  [DIM]int16
		max  [DIM]int16
		want uint64
	}{
		{
			name: "query inside bbox",
			q:    [DIM]int16{5000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			min:  [DIM]int16{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			max:  [DIM]int16{10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000},
			want: 0,
		},
		{
			name: "query outside bbox",
			q:    [DIM]int16{15000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			min:  [DIM]int16{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			max:  [DIM]int16{10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000},
			want: 5000 * 5000,
		},
		{
			name: "query below bbox",
			q:    [DIM]int16{-5000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			min:  [DIM]int16{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			max:  [DIM]int16{10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000},
			want: 5000 * 5000,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := gridBBoxLowerBound(&tt.q, &tt.min, &tt.max)
			if got != tt.want {
				t.Errorf("gridBBoxLowerBound = %d, want %d", got, tt.want)
			}
		})
	}
}

func TestGridScanLeaf_Empty(t *testing.T) {
	vectors := make([]int16, 0)
	labels := make([]byte, 0)
	var q [DIM]int16
	h := newTopK5()
	var scratch [8]float32
	gridScanLeaf(vectors, labels, 0, 0, 0, q, &h, &scratch)
	if h.count != 0 {
		t.Errorf("count on empty scan = %d, want 0", h.count)
	}
}

func TestExactI64Dist(t *testing.T) {
	block := make([]int16, DIM*gridBlockLANES)
	for d := 0; d < DIM; d++ {
		block[d*gridBlockLANES+0] = 5000
	}

	var q [DIM]int16
	for d := 0; d < DIM; d++ {
		q[d] = 5000
	}

	dist := exactI64Dist(&q, &block[0], 0)
	if dist != 0 {
		t.Errorf("exactI64Dist = %d, want 0", dist)
	}
}

func TestBuildGrid(t *testing.T) {
	refs := []model.Reference{
		{Vector: model.Vector14{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}, Label: "fraud"},
		{Vector: model.Vector14{0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9}, Label: "legit"},
		{Vector: model.Vector14{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5}, Label: "fraud"},
	}

	var buf bytes.Buffer
	gz := gzip.NewWriter(&buf)
	enc := json.NewEncoder(gz)
	if _, err := gz.Write([]byte(`[`)); err != nil {
		t.Fatal(err)
	}
	for i, r := range refs {
		if i > 0 {
			if _, err := gz.Write([]byte(`,`)); err != nil {
				t.Fatal(err)
			}
		}
		vec := make([]float64, 14)
		for d := 0; d < 14; d++ {
			vec[d] = float64(r.Vector[d])
		}
		entry := map[string]interface{}{
			"vector": vec,
			"label":  r.Label,
		}
		if err := enc.Encode(entry); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := gz.Write([]byte(`]`)); err != nil {
		t.Fatal(err)
	}
	if err := gz.Close(); err != nil {
		t.Fatal(err)
	}

	tmpDir := t.TempDir()
	refsPath := tmpDir + "/refs.json.gz"
	gridPath := tmpDir + "/grid.bin"

	if err := os.WriteFile(refsPath, buf.Bytes(), 0644); err != nil {
		t.Fatal(err)
	}

	err := BuildGrid(refsPath, gridPath)
	if err != nil {
		t.Fatalf("BuildGrid failed: %v", err)
	}

	idx, err := LoadGrid(gridPath)
	if err != nil {
		t.Fatalf("LoadGrid failed: %v", err)
	}

	if idx.Count() != 3 {
		t.Errorf("Count = %d, want 3", idx.Count())
	}
	if idx.FraudCount() != 2 {
		t.Errorf("FraudCount = %d, want 2", idx.FraudCount())
	}
}

func TestBuildGrid_InvalidFile(t *testing.T) {
	err := BuildGrid("/nonexistent/file.gz", "/tmp/test.bin")
	if err == nil {
		t.Fatal("expected error for nonexistent file")
	}
}

func TestLoadGrid_InvalidFile(t *testing.T) {
	_, err := LoadGrid("/nonexistent/grid.bin")
	if err == nil {
		t.Fatal("expected error for nonexistent file")
	}
}

func TestLoadGrid_InvalidMagic(t *testing.T) {
	tmpFile := t.TempDir() + "/bad.bin"
	if err := os.WriteFile(tmpFile, []byte("not a grid"), 0644); err != nil {
		t.Fatal(err)
	}
	_, err := LoadGrid(tmpFile)
	if err == nil {
		t.Fatal("expected error for invalid grid file")
	}
}

func TestGridIndex_Predict(t *testing.T) {
	refs := []model.Reference{
		{Vector: model.Vector14{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}, Label: "fraud"},
		{Vector: model.Vector14{0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9}, Label: "legit"},
	}

	var buf bytes.Buffer
	gz := gzip.NewWriter(&buf)
	enc := json.NewEncoder(gz)
	if _, err := gz.Write([]byte(`[`)); err != nil {
		t.Fatal(err)
	}
	for i, r := range refs {
		if i > 0 {
			if _, err := gz.Write([]byte(`,`)); err != nil {
				t.Fatal(err)
			}
		}
		vec := make([]float64, 14)
		for d := 0; d < 14; d++ {
			vec[d] = float64(r.Vector[d])
		}
		entry := map[string]interface{}{
			"vector": vec,
			"label":  r.Label,
		}
		if err := enc.Encode(entry); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := gz.Write([]byte(`]`)); err != nil {
		t.Fatal(err)
	}
	if err := gz.Close(); err != nil {
		t.Fatal(err)
	}

	tmpDir := t.TempDir()
	refsPath := tmpDir + "/refs.json.gz"
	gridPath := tmpDir + "/grid.bin"

	if err := os.WriteFile(refsPath, buf.Bytes(), 0644); err != nil {
		t.Fatal(err)
	}

	if err := BuildGrid(refsPath, gridPath); err != nil {
		t.Fatalf("BuildGrid failed: %v", err)
	}

	idx, err := LoadGrid(gridPath)
	if err != nil {
		t.Fatalf("LoadGrid failed: %v", err)
	}

	query := model.Vector14{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}
	score := idx.Predict(query, 5)
	if score < 0 || score > 1 {
		t.Errorf("Predict = %v, want [0,1]", score)
	}
}

func TestGridIndex_PredictRaw(t *testing.T) {
	refs := []model.Reference{
		{Vector: model.Vector14{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}, Label: "fraud"},
		{Vector: model.Vector14{0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9}, Label: "legit"},
	}

	var buf bytes.Buffer
	gz := gzip.NewWriter(&buf)
	enc := json.NewEncoder(gz)
	if _, err := gz.Write([]byte(`[`)); err != nil {
		t.Fatal(err)
	}
	for i, r := range refs {
		if i > 0 {
			if _, err := gz.Write([]byte(`,`)); err != nil {
				t.Fatal(err)
			}
		}
		vec := make([]float64, 14)
		for d := 0; d < 14; d++ {
			vec[d] = float64(r.Vector[d])
		}
		entry := map[string]interface{}{
			"vector": vec,
			"label":  r.Label,
		}
		if err := enc.Encode(entry); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := gz.Write([]byte(`]`)); err != nil {
		t.Fatal(err)
	}
	if err := gz.Close(); err != nil {
		t.Fatal(err)
	}

	tmpDir := t.TempDir()
	refsPath := tmpDir + "/refs.json.gz"
	gridPath := tmpDir + "/grid.bin"

	if err := os.WriteFile(refsPath, buf.Bytes(), 0644); err != nil {
		t.Fatal(err)
	}

	if err := BuildGrid(refsPath, gridPath); err != nil {
		t.Fatalf("BuildGrid failed: %v", err)
	}

	idx, err := LoadGrid(gridPath)
	if err != nil {
		t.Fatalf("LoadGrid failed: %v", err)
	}

	query := model.Vector14{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}
	count := idx.PredictRaw(query, 0)
	if count < 0 || count > 5 {
		t.Errorf("PredictRaw = %d, want [0,5]", count)
	}
}

func TestWidestDim(t *testing.T) {
	min := [DIM]int16{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	max := [DIM]int16{100, 200, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}

	d := widestDim(&min, &max)
	if d != 1 {
		t.Errorf("widestDim = %d, want 1", d)
	}
}

func TestWidestDim_AllSame(t *testing.T) {
	min := [DIM]int16{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	max := [DIM]int16{100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100}

	d := widestDim(&min, &max)
	if d != 0 {
		t.Errorf("widestDim = %d, want 0", d)
	}
}

func TestQuantizeQuery_Extreme(t *testing.T) {
	query := model.Vector14{2.0, -2.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	qi := quantizeQuery(query)
	if qi[0] != 20000 {
		t.Errorf("qi[0] = %d, want 20000", qi[0])
	}
	if qi[1] != -20000 {
		t.Errorf("qi[1] = %d, want -20000", qi[1])
	}
}

func TestGridIndex_Predict_EmptyIndex(t *testing.T) {
	idx := &GridIndex{
		n:          0,
		nodes:      make([]gridNode, 0),
		partitions: make([]gridPartition, 0),
		vectors:    make([]int16, 0),
		labels:     make([]byte, 0),
	}
	for i := range idx.partitionsByKey {
		idx.partitionsByKey[i] = -1
	}

	query := model.Vector14{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5}
	score := idx.Predict(query, 5)
	if score != 0 {
		t.Errorf("Predict on empty index = %v, want 0", score)
	}
}

func TestGridIndex_PredictRaw_NoPartitions(t *testing.T) {
	idx := &GridIndex{
		n:          0,
		nodes:      make([]gridNode, 0),
		partitions: make([]gridPartition, 0),
		vectors:    make([]int16, 0),
		labels:     make([]byte, 0),
	}
	for i := range idx.partitionsByKey {
		idx.partitionsByKey[i] = -1
	}

	query := model.Vector14{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5}
	count := idx.PredictRaw(query, 0)
	if count != 0 {
		t.Errorf("PredictRaw on empty index = %d, want 0", count)
	}
}

func TestGridBBoxLowerBound_MultipleDimsOutside(t *testing.T) {
	q := [DIM]int16{20000, 20000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	min := [DIM]int16{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	max := [DIM]int16{10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000}

	got := gridBBoxLowerBound(&q, &min, &max)
	want := uint64(10000*10000 + 10000*10000)
	if got != want {
		t.Errorf("gridBBoxLowerBound = %d, want %d", got, want)
	}
}

func TestBuildGrid_LargeDataset(t *testing.T) {
	n := 1000
	refs := make([]model.Reference, n)
	for i := 0; i < n; i++ {
		for d := 0; d < 14; d++ {
			refs[i].Vector[d] = float32(i%100) / 100.0
		}
		if i%2 == 0 {
			refs[i].Label = "fraud"
		} else {
			refs[i].Label = "legit"
		}
	}

	var buf bytes.Buffer
	gz := gzip.NewWriter(&buf)
	enc := json.NewEncoder(gz)
	if _, err := gz.Write([]byte(`[`)); err != nil {
		t.Fatal(err)
	}
	for i, r := range refs {
		if i > 0 {
			if _, err := gz.Write([]byte(`,`)); err != nil {
				t.Fatal(err)
			}
		}
		vec := make([]float64, 14)
		for d := 0; d < 14; d++ {
			vec[d] = float64(r.Vector[d])
		}
		entry := map[string]interface{}{
			"vector": vec,
			"label":  r.Label,
		}
		if err := enc.Encode(entry); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := gz.Write([]byte(`]`)); err != nil {
		t.Fatal(err)
	}
	if err := gz.Close(); err != nil {
		t.Fatal(err)
	}

	tmpDir := t.TempDir()
	refsPath := tmpDir + "/refs.json.gz"
	gridPath := tmpDir + "/grid.bin"

	if err := os.WriteFile(refsPath, buf.Bytes(), 0644); err != nil {
		t.Fatal(err)
	}

	err := BuildGrid(refsPath, gridPath)
	if err != nil {
		t.Fatalf("BuildGrid failed: %v", err)
	}

	idx, err := LoadGrid(gridPath)
	if err != nil {
		t.Fatalf("LoadGrid failed: %v", err)
	}

	if idx.Count() != n {
		t.Errorf("Count = %d, want %d", idx.Count(), n)
	}
}

func TestGridPartitionKey_AllCombinations(t *testing.T) {
	for hour := int16(0); hour <= 10000; hour += 2500 {
		for installments := int16(0); installments <= 10000; installments += 2500 {
			for online := int16(0); online <= 1; online++ {
				for merchant := int16(0); merchant <= 1; merchant++ {
					for mccVal := int16(0); mccVal <= 8000; mccVal += 2000 {
						v := [DIM]int16{0, installments, 0, hour, 0, 0, 0, 0, 0, online, 0, merchant, mccVal, 0}
						key := gridPartitionKey(&v)

						expected := uint32(0)
						expected |= uint32(hour/1250) << 0
						expected |= uint32(installments/2500) << 3
						if mccVal > 3000 {
							expected |= 1 << 5
						}
						if online > 0 {
							expected |= 1 << 6
						}
						if merchant > 0 {
							expected |= 1 << 7
						}

						if key != expected {
							t.Errorf("key mismatch for hour=%d inst=%d online=%d merch=%d mcc=%d: got %d, want %d",
								hour, installments, online, merchant, mccVal, key, expected)
						}
					}
				}
			}
		}
	}
}

func TestGridScratch_StackCapacity(t *testing.T) {
	var scratch gridScratch
	_ = scratch.stackNodes
	_ = scratch.stackBounds
	if cap(scratch.stackNodes) < 64 {
		t.Errorf("stackNodes capacity = %d, want >= 64", cap(scratch.stackNodes))
	}
	if cap(scratch.stackBounds) < 64 {
		t.Errorf("stackBounds capacity = %d, want >= 64", cap(scratch.stackBounds))
	}
}

func TestExactI64Dist_DifferentVectors(t *testing.T) {
	block := make([]int16, DIM*gridBlockLANES)
	for d := 0; d < DIM; d++ {
		block[d*gridBlockLANES+0] = 0
		block[d*gridBlockLANES+1] = 10000
	}

	var q [DIM]int16
	for d := 0; d < DIM; d++ {
		q[d] = 5000
	}

	dist0 := exactI64Dist(&q, &block[0], 0)
	dist1 := exactI64Dist(&q, &block[0], 1)

	if dist0 != dist1 {
		t.Errorf("distances should be equal: %d vs %d", dist0, dist1)
	}

	expected := uint64(DIM * 5000 * 5000)
	if dist0 != expected {
		t.Errorf("exactI64Dist = %d, want %d", dist0, expected)
	}
}

func TestGridIndex_FraudCount_AllFraud(t *testing.T) {
	n := 16
	labels := make([]byte, (n+7)/8)
	for i := 0; i < n; i++ {
		labels[i>>3] |= 1 << uint(i&7)
	}

	idx := &GridIndex{
		n:      n,
		labels: labels,
	}

	if idx.FraudCount() != n {
		t.Errorf("FraudCount = %d, want %d", idx.FraudCount(), n)
	}
}

func TestGridIndex_FraudCount_None(t *testing.T) {
	n := 16
	labels := make([]byte, (n+7)/8)

	idx := &GridIndex{
		n:      n,
		labels: labels,
	}

	if idx.FraudCount() != 0 {
		t.Errorf("FraudCount = %d, want 0", idx.FraudCount())
	}
}

func TestGridIndex_Predict_ExactMatch(t *testing.T) {
	refs := []model.Reference{
		{Vector: model.Vector14{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5}, Label: "fraud"},
	}

	var buf bytes.Buffer
	gz := gzip.NewWriter(&buf)
	enc := json.NewEncoder(gz)
	if _, err := gz.Write([]byte(`[`)); err != nil {
		t.Fatal(err)
	}
	vec := make([]float64, 14)
	for d := 0; d < 14; d++ {
		vec[d] = float64(refs[0].Vector[d])
	}
	entry := map[string]interface{}{
		"vector": vec,
		"label":  refs[0].Label,
	}
	if err := enc.Encode(entry); err != nil {
		t.Fatal(err)
	}
	if _, err := gz.Write([]byte(`]`)); err != nil {
		t.Fatal(err)
	}
	if err := gz.Close(); err != nil {
		t.Fatal(err)
	}

	tmpDir := t.TempDir()
	refsPath := tmpDir + "/refs.json.gz"
	gridPath := tmpDir + "/grid.bin"

	if err := os.WriteFile(refsPath, buf.Bytes(), 0644); err != nil {
		t.Fatal(err)
	}

	if err := BuildGrid(refsPath, gridPath); err != nil {
		t.Fatalf("BuildGrid failed: %v", err)
	}

	idx, err := LoadGrid(gridPath)
	if err != nil {
		t.Fatalf("LoadGrid failed: %v", err)
	}

	query := model.Vector14{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5}
	score := idx.Predict(query, 5)
	if score <= 0 {
		t.Errorf("Predict = %v, want > 0", score)
	}
}

func TestGridPartitionKey_EdgeCases(t *testing.T) {
	// hour bucket boundaries: v[3]/1250
	v := [DIM]int16{0, 0, 0, 1249, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	key := gridPartitionKey(&v)
	if want := uint32(0); key != want {
		t.Errorf("key for hour=1249 = %d, want %d", key, want)
	}

	v[3] = 1250
	key = gridPartitionKey(&v)
	if want := uint32(1); key != want {
		t.Errorf("key for hour=1250 = %d, want %d", key, want)
	}

	// mcc high threshold at > 3000
	v = [DIM]int16{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3000, 0}
	key = gridPartitionKey(&v)
	if key&(1<<5) != 0 {
		t.Errorf("key for mcc=3000 should not set bit 5")
	}

	v[12] = 3001
	key = gridPartitionKey(&v)
	if key&(1<<5) == 0 {
		t.Errorf("key for mcc=3001 should set bit 5")
	}
}

func TestGridBBoxLowerBound_QueryOnBoundary(t *testing.T) {
	q := [DIM]int16{10000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	min := [DIM]int16{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	max := [DIM]int16{10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000}

	got := gridBBoxLowerBound(&q, &min, &max)
	if got != 0 {
		t.Errorf("gridBBoxLowerBound on boundary = %d, want 0", got)
	}
}

func TestBuildGrid_EmptyDataset(t *testing.T) {
	var buf bytes.Buffer
	gz := gzip.NewWriter(&buf)
	if _, err := gz.Write([]byte(`[]`)); err != nil {
		t.Fatal(err)
	}
	if err := gz.Close(); err != nil {
		t.Fatal(err)
	}

	tmpDir := t.TempDir()
	refsPath := tmpDir + "/refs.json.gz"
	gridPath := tmpDir + "/grid.bin"

	if err := os.WriteFile(refsPath, buf.Bytes(), 0644); err != nil {
		t.Fatal(err)
	}

	err := BuildGrid(refsPath, gridPath)
	if err != nil {
		t.Fatalf("BuildGrid failed: %v", err)
	}

	idx, err := LoadGrid(gridPath)
	if err != nil {
		t.Fatalf("LoadGrid failed: %v", err)
	}

	if idx.Count() != 0 {
		t.Errorf("Count = %d, want 0", idx.Count())
	}
}

func TestGridScratch_PartBoundsSize(t *testing.T) {
	var scratch gridScratch
	_ = scratch.stackNodes
	_ = scratch.stackBounds
	_ = scratch.blockSum
	if len(scratch.stackNodes) != 64 {
		t.Errorf("stackNodes len = %d, want 64", len(scratch.stackNodes))
	}
	if len(scratch.stackBounds) != 64 {
		t.Errorf("stackBounds len = %d, want 64", len(scratch.stackBounds))
	}
	if len(scratch.blockSum) != 8 {
		t.Errorf("blockSum len = %d, want 8", len(scratch.blockSum))
	}
}

func TestGridIndex_Predict_MultipleFraud(t *testing.T) {
	refs := []model.Reference{
		{Vector: model.Vector14{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}, Label: "fraud"},
		{Vector: model.Vector14{0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2}, Label: "fraud"},
		{Vector: model.Vector14{0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3}, Label: "fraud"},
		{Vector: model.Vector14{0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9}, Label: "legit"},
		{Vector: model.Vector14{0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95}, Label: "legit"},
	}

	var buf bytes.Buffer
	gz := gzip.NewWriter(&buf)
	enc := json.NewEncoder(gz)
	if _, err := gz.Write([]byte(`[`)); err != nil {
		t.Fatal(err)
	}
	for i, r := range refs {
		if i > 0 {
			if _, err := gz.Write([]byte(`,`)); err != nil {
				t.Fatal(err)
			}
		}
		vec := make([]float64, 14)
		for d := 0; d < 14; d++ {
			vec[d] = float64(r.Vector[d])
		}
		entry := map[string]interface{}{
			"vector": vec,
			"label":  r.Label,
		}
		if err := enc.Encode(entry); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := gz.Write([]byte(`]`)); err != nil {
		t.Fatal(err)
	}
	if err := gz.Close(); err != nil {
		t.Fatal(err)
	}

	tmpDir := t.TempDir()
	refsPath := tmpDir + "/refs.json.gz"
	gridPath := tmpDir + "/grid.bin"

	if err := os.WriteFile(refsPath, buf.Bytes(), 0644); err != nil {
		t.Fatal(err)
	}

	if err := BuildGrid(refsPath, gridPath); err != nil {
		t.Fatalf("BuildGrid failed: %v", err)
	}

	idx, err := LoadGrid(gridPath)
	if err != nil {
		t.Fatalf("LoadGrid failed: %v", err)
	}

	query := model.Vector14{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}
	score := idx.Predict(query, 5)
	if score < 0.6 {
		t.Errorf("Predict = %v, want >= 0.6 (3/5 fraud)", score)
	}
}

func TestGridPartitionKey_NegativeValues(t *testing.T) {
	v := [DIM]int16{0, 0, 0, -100, 0, 0, 0, 0, 0, 0, 0, 0, -1000, 0}
	key := gridPartitionKey(&v)
	// Negative hour: int16(-100)/1250 = 0 in Go (truncation toward zero)
	if key&7 != 0 {
		t.Errorf("negative hour should give bucket 0, got %d", key&7)
	}
	// Negative mcc: -1000 <= 3000, should not set bit 5
	if key&(1<<5) != 0 {
		t.Error("mcc=-1000 should not set bit 5")
	}
}

func TestGridBBoxLowerBound_AllDimsOutside(t *testing.T) {
	q := [DIM]int16{20000, 20000, 20000, 20000, 20000, 20000, 20000, 20000, 20000, 20000, 20000, 20000, 20000, 20000}
	min := [DIM]int16{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	max := [DIM]int16{10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000}

	got := gridBBoxLowerBound(&q, &min, &max)
	want := uint64(DIM * 10000 * 10000)
	if got != want {
		t.Errorf("gridBBoxLowerBound = %d, want %d", got, want)
	}
}

func TestGridIndex_Predict_DifferentPartitions(t *testing.T) {
	refs := []model.Reference{
		{Vector: model.Vector14{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}, Label: "fraud"},
		{Vector: model.Vector14{0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9}, Label: "legit"},
	}

	var buf bytes.Buffer
	gz := gzip.NewWriter(&buf)
	enc := json.NewEncoder(gz)
	if _, err := gz.Write([]byte(`[`)); err != nil {
		t.Fatal(err)
	}
	for i, r := range refs {
		if i > 0 {
			if _, err := gz.Write([]byte(`,`)); err != nil {
				t.Fatal(err)
			}
		}
		vec := make([]float64, 14)
		for d := 0; d < 14; d++ {
			vec[d] = float64(r.Vector[d])
		}
		entry := map[string]interface{}{
			"vector": vec,
			"label":  r.Label,
		}
		if err := enc.Encode(entry); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := gz.Write([]byte(`]`)); err != nil {
		t.Fatal(err)
	}
	if err := gz.Close(); err != nil {
		t.Fatal(err)
	}

	tmpDir := t.TempDir()
	refsPath := tmpDir + "/refs.json.gz"
	gridPath := tmpDir + "/grid.bin"

	if err := os.WriteFile(refsPath, buf.Bytes(), 0644); err != nil {
		t.Fatal(err)
	}

	if err := BuildGrid(refsPath, gridPath); err != nil {
		t.Fatalf("BuildGrid failed: %v", err)
	}

	idx, err := LoadGrid(gridPath)
	if err != nil {
		t.Fatalf("LoadGrid failed: %v", err)
	}

	query := model.Vector14{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5}
	score := idx.Predict(query, 5)
	if score < 0 || score > 1 {
		t.Errorf("Predict = %v, want [0,1]", score)
	}
}

func TestQuantizeQuery_Zero(t *testing.T) {
	query := model.Vector14{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	qi := quantizeQuery(query)
	for d := 0; d < DIM; d++ {
		if qi[d] != 0 {
			t.Errorf("qi[%d] = %d, want 0", d, qi[d])
		}
	}
}

func TestQuantizeQuery_SmallValues(t *testing.T) {
	query := model.Vector14{0.001, -0.001, 0.001, -0.001, 0.001, -0.001, 0.001, -0.001, 0.001, -0.001, 0.001, -0.001, 0.001, -0.001}
	qi := quantizeQuery(query)
	for d := 0; d < DIM; d++ {
		expected := int16(10)
		if d%2 == 1 {
			expected = -10
		}
		if qi[d] != expected {
			t.Errorf("qi[%d] for small value = %d, want %d", d, qi[d], expected)
		}
	}
}

func BenchmarkGridIndex_Predict(b *testing.B) {
	gridPath := "../../resources/grid.bin"
	idx, err := LoadGrid(gridPath)
	if err != nil {
		b.Fatalf("LoadGrid failed: %v", err)
	}

	query := model.Vector14{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = idx.Predict(query, 5)
	}
}

func BenchmarkGridIndex_PredictRaw(b *testing.B) {
	gridPath := "../../resources/grid.bin"
	idx, err := LoadGrid(gridPath)
	if err != nil {
		b.Fatalf("LoadGrid failed: %v", err)
	}

	query := model.Vector14{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = idx.PredictRaw(query, 0)
	}
}

// BenchmarkGridIndex_PredictRaw_Histogram measures per-call latency of PredictRaw
// with internally recorded μs-resolution histogram, exported via b.ReportMetric.
// Uses a pool of 512 pseudo-random queries to simulate realistic workload.
// Reports overall p50/p95/p99 and per-second p99 (to detect tail stability under
// GOMAXPROCS=1).
func BenchmarkGridIndex_PredictRaw_Histogram(b *testing.B) {
	gridPath := "../../resources/grid.bin"
	idx, err := LoadGrid(gridPath)
	if err != nil {
		b.Fatalf("LoadGrid failed: %v", err)
	}

	// Pre-generate 512 queries spread across the [0,1]ⁱ⁴ hypercube with a bias
	// toward the origin (where most real transactions cluster).
	var queries [512]model.Vector14
	for i := range queries {
		for d := 0; d < DIM; d++ {
			queries[i][d] = float32(math.Abs(math.Sin(float64(i*17+d*31)))) * 0.3
		}
	}
	queryBuf := queries[:]

	// Histogram: 5μs buckets from 0 to 2000μs (more than enough for p99).
	const (
		bucketWidth = 5   // μs per bucket
		maxBucket   = 400 // 400 * 5 = 2000μs
	)

	type secondsHist struct {
		mu   sync.Mutex
		secs [][maxBucket]uint64
	}

	var sh secondsHist

	b.ResetTimer()
	start := time.Now()
	for i := 0; i < b.N; i++ {
		q := queryBuf[i&511]
		t0 := time.Now()
		_ = idx.PredictRaw(q, 0)
		elapsed := time.Since(t0)
		bucket := int(elapsed.Microseconds()) / bucketWidth
		if bucket >= maxBucket {
			bucket = maxBucket - 1
		}
		sec := int(time.Since(start).Seconds())

		sh.mu.Lock()
		for sec >= len(sh.secs) {
			sh.secs = append(sh.secs, [maxBucket]uint64{})
		}
		sh.secs[sec][bucket]++
		sh.mu.Unlock()
	}
	b.StopTimer()

	nsec := len(sh.secs)
	if nsec == 0 {
		return
	}

	// Merge all per-second histograms into the first entry.
	merged := sh.secs[0]
	for s := 1; s < nsec; s++ {
		for bk := range merged {
			merged[bk] += sh.secs[s][bk]
		}
	}

	total := uint64(0)
	for _, c := range merged {
		total += c
	}

	// Compute overall percentiles.
	type pct struct {
		rank  int64 // per-mille rank (500 = p50, 950 = p95, 999 = p99.9)
		label string
	}
	marks := []pct{
		{500, "p50_ms"},
		{950, "p95_ms"},
		{990, "p99_ms"},
		{999, "p99_9_ms"},
	}
	var cumulative uint64
	mi := 0
	for bk, c := range merged {
		if c == 0 {
			continue
		}
		cumulative += c
		for mi < len(marks) && cumulative*1000 >= uint64(marks[mi].rank)*total {
			val := float64(bk*bucketWidth) / 1000
			b.ReportMetric(val, marks[mi].label)
			mi++
		}
	}

	// Per-second p99 — print to b.Log for inspection.
	p99Rank := uint64(990)
	b.Logf("\nper-second p99 (ms):")
	for s := 0; s < nsec; s++ {
		var sTotal, sCum uint64
		for _, c := range sh.secs[s] {
			sTotal += c
		}
		if sTotal == 0 {
			continue
		}
		p99th := (sTotal*p99Rank + 500) / 1000
		for bk, c := range sh.secs[s] {
			sCum += c
			if sCum >= p99th {
				b.Logf("  sec %2d: %5.2fms  (n=%d)", s, float64(bk*bucketWidth)/1000, sTotal)
				break
			}
		}
	}
}
