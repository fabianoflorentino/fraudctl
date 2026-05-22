package knn

import (
	"bytes"
	"compress/gzip"
	"encoding/json"
	"os"
	"testing"

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
			want: 1 << 0,
		},
		{
			name: "last_tx positive",
			v:    [DIM]int16{0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
			want: 1 << 0,
		},
		{
			name: "last_tx negative",
			v:    [DIM]int16{0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0},
			want: 0,
		},
		{
			name: "online positive",
			v:    [DIM]int16{0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
			want: (1 << 0) | (1 << 1),
		},
		{
			name: "card present",
			v:    [DIM]int16{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
			want: (1 << 0) | (1 << 2),
		},
		{
			name: "unknown merchant",
			v:    [DIM]int16{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
			want: (1 << 0) | (1 << 3),
		},
		{
			name: "mcc bucket 0",
			v:    [DIM]int16{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1000, 0},
			want: (1 << 0) | (0 << 4),
		},
		{
			name: "mcc bucket 1",
			v:    [DIM]int16{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3000, 0},
			want: (1 << 0) | (1 << 4),
		},
		{
			name: "mcc bucket 2",
			v:    [DIM]int16{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6000, 0},
			want: (1 << 0) | (2 << 4),
		},
		{
			name: "mcc bucket 3",
			v:    [DIM]int16{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9000, 0},
			want: (1 << 0) | (3 << 4),
		},
		{
			name: "amount high",
			v:    [DIM]int16{0, 0, 6000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			want: (1 << 0) | (1 << 6),
		},
		{
			name: "tx_count high",
			v:    [DIM]int16{0, 0, 0, 0, 0, 0, 0, 0, 3000, 0, 0, 0, 0, 0},
			want: (1 << 0) | (1 << 7),
		},
		{
			name: "all flags",
			v:    [DIM]int16{0, 0, 6000, 0, 0, 1, 0, 0, 3000, 1, 1, 1, 3000, 0},
			want: (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4) | (1 << 6) | (1 << 7),
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
	for lastTx := int16(-1); lastTx <= 1; lastTx += 2 {
		for online := int16(0); online <= 1; online++ {
			for card := int16(0); card <= 1; card++ {
				for merchant := int16(0); merchant <= 1; merchant++ {
					for mccBucket := 0; mccBucket < 4; mccBucket++ {
						for amount := int16(0); amount <= 6000; amount += 6000 {
							for txCount := int16(0); txCount <= 3000; txCount += 3000 {
								v := [DIM]int16{0, 0, amount, 0, 0, lastTx, 0, 0, txCount, online, card, merchant, int16(mccBucket * 2500), 0}
								key := gridPartitionKey(&v)

								expected := uint32(0)
								if lastTx >= 0 {
									expected |= 1 << 0
								}
								if online > 0 {
									expected |= 1 << 1
								}
								if card > 0 {
									expected |= 1 << 2
								}
								if merchant > 0 {
									expected |= 1 << 3
								}
								expected |= uint32(mccBucket) << 4
								if amount > 5000 {
									expected |= 1 << 6
								}
								if txCount > 2500 {
									expected |= 1 << 7
								}

								if key != expected {
									t.Errorf("key mismatch for lastTx=%d online=%d card=%d merchant=%d mcc=%d amount=%d txCount=%d: got %d, want %d",
										lastTx, online, card, merchant, mccBucket, amount, txCount, key, expected)
								}
							}
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
	v := [DIM]int16{0, 0, 5000, 0, 0, 0, 0, 0, 2500, 0, 0, 0, 0, 0}
	key := gridPartitionKey(&v)
	if key != (1 << 0) {
		t.Errorf("key for boundary values = %d, want %d", key, 1<<0)
	}

	v[2] = 5001
	v[8] = 2501
	key = gridPartitionKey(&v)
	if key != (1<<0)|(1<<6)|(1<<7) {
		t.Errorf("key for above-boundary values = %d, want %d", key, (1<<0)|(1<<6)|(1<<7))
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
	v := [DIM]int16{0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, -1000, 0}
	key := gridPartitionKey(&v)
	if key&(1<<0) != 0 {
		t.Error("last_tx=-1 should not set bit 0")
	}
	if key&(0xF<<4) != 0 {
		t.Error("mcc=-1000 should not set mcc bits")
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
