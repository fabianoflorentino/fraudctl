package knn

import (
	"bytes"
	"compress/gzip"
	"container/heap"
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/fabianoflorentino/fraudctl/internal/model"
)

func TestIVFIndex_CountFraudCount(t *testing.T) {
	idx := &IVFIndex{
		nlist:     2,
		centroids: make([]float32, 2*DIM),
		vectors:   make([]int16, 3*DIM),
		labels:    []byte{0b00000101},
		offsets:   []uint32{0, 2, 3},
	}
	if idx.Count() != 3 {
		t.Errorf("Count = %d, want 3", idx.Count())
	}
	if idx.FraudCount() != 2 {
		t.Errorf("FraudCount = %d, want 2", idx.FraudCount())
	}
}

func TestIVFIndex_DebugMethods(t *testing.T) {
	centroids := make([]float32, 3*DIM)
	centroids[0] = 0.1
	centroids[14] = 0.2
	centroids[28] = 0.3

	idx := &IVFIndex{
		nlist:     3,
		centroids: centroids,
		offsets:   []uint32{0, 10, 20, 30},
	}

	nlist := idx.DebugNList()
	if nlist != 3 {
		t.Errorf("DebugNList = %d, want 3", nlist)
	}

	offsets := idx.DebugOffsets()
	if len(offsets) != 4 || offsets[2] != 20 {
		t.Errorf("DebugOffsets = %v, want [0 10 20 30]", offsets)
	}

	centroidsOut := idx.DebugCentroids()
	if len(centroidsOut) != 3*DIM {
		t.Errorf("DebugCentroids len = %d, want %d", len(centroidsOut), 3*DIM)
	}
	if centroidsOut[0] != 0.1 {
		t.Errorf("DebugCentroids[0] = %v, want 0.1", centroidsOut[0])
	}
}

func TestIVFIndex_FraudCount_Empty(t *testing.T) {
	idx := &IVFIndex{
		offsets: []uint32{0},
		labels:  make([]byte, 0),
		nlist:   0,
	}
	if idx.FraudCount() != 0 {
		t.Errorf("FraudCount empty = %d, want 0", idx.FraudCount())
	}
}

func TestMaxHeap_Pop(t *testing.T) {
	h := make(maxHeap, 0, 3)
	heap.Push(&h, candidate{3, 0})
	heap.Push(&h, candidate{1, 1})
	heap.Push(&h, candidate{2, 2})

	popped := heap.Pop(&h).(candidate)
	if popped.dist != 3 {
		t.Errorf("Pop() dist = %v, want 3 (largest)", popped.dist)
	}
	if len(h) != 2 {
		t.Errorf("len after pop = %d, want 2", len(h))
	}
}

func TestBrutePredict_MultipleVectors(t *testing.T) {
	flat := []float32{
		0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0.9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	}
	flags := []bool{true, false, true}
	query := model.Vector14{0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}

	score := brutePredict(flat, flags, 3, query, 3)
	if score < 0 || score > 1 {
		t.Errorf("score = %v, want [0,1]", score)
	}
}

func TestBBoxMayImprove_AllDimsInRange(t *testing.T) {
	nlist := 1
	bboxMin := make([]int16, nlist*DIM)
	bboxMax := make([]int16, nlist*DIM)
	for d := 0; d < DIM; d++ {
		bboxMin[d] = -10000
		bboxMax[d] = 10000
	}

	var qi [DIM]int16
	for d := 0; d < DIM; d++ {
		qi[d] = 0
	}

	if !bboxMayImprove(bboxMin, bboxMax, 0, qi, math.MaxUint64) {
		t.Error("should return true when query is well within bbox")
	}
}

func TestBBoxMayImprove_JustOutside(t *testing.T) {
	nlist := 1
	bboxMin := make([]int16, nlist*DIM)
	bboxMax := make([]int16, nlist*DIM)
	for d := 0; d < DIM; d++ {
		bboxMin[d] = 10
		bboxMax[d] = 20
	}

	var qi [DIM]int16
	for d := 0; d < DIM; d++ {
		qi[d] = 0
	}

	if !bboxMayImprove(bboxMin, bboxMax, 0, qi, 1000000) {
		t.Error("should return true when query is just outside bbox with large worstDist")
	}
}

func TestBBoxMayImprove_VeryFar(t *testing.T) {
	nlist := 1
	bboxMin := make([]int16, nlist*DIM)
	bboxMax := make([]int16, nlist*DIM)
	for d := 0; d < DIM; d++ {
		bboxMin[d] = 10000
		bboxMax[d] = 20000
	}

	var qi [DIM]int16
	for d := 0; d < DIM; d++ {
		qi[d] = 0
	}

	if bboxMayImprove(bboxMin, bboxMax, 0, qi, 100) {
		t.Error("should return false when query is very far from bbox with tight worstDist")
	}
}

func TestBruteIndex_BuildFromGzip(t *testing.T) {
	refs := []model.Reference{
		{Vector: model.Vector14{0.1, 0.1, 0.1}, Label: "fraud"},
		{Vector: model.Vector14{0.9, 0.9, 0.9}, Label: "legit"},
	}

	var buf bytes.Buffer
	gz := gzip.NewWriter(&buf)
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
		if err := json.NewEncoder(gz).Encode(entry); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := gz.Write([]byte(`]`)); err != nil {
		t.Fatal(err)
	}
	if err := gz.Close(); err != nil {
		t.Fatal(err)
	}

	tmpFile := t.TempDir() + "/test_refs.json.gz"
	if err := os.WriteFile(tmpFile, buf.Bytes(), 0644); err != nil {
		t.Fatal(err)
	}

	bi := NewBruteIndex()
	err := bi.BuildFromGzip(tmpFile, 10)
	if err != nil {
		t.Fatalf("BuildFromGzip failed: %v", err)
	}

	if bi.Count() != 2 {
		t.Errorf("Count = %d, want 2", bi.Count())
	}
	if bi.FraudCount() != 1 {
		t.Errorf("FraudCount = %d, want 1", bi.FraudCount())
	}
}

func TestBruteIndex_BuildFromGzip_InvalidFile(t *testing.T) {
	bi := NewBruteIndex()
	err := bi.BuildFromGzip("/nonexistent/file.gz", 10)
	if err == nil {
		t.Fatal("expected error for nonexistent file")
	}
}

func TestBruteIndex_BuildFromGzip_InvalidGzip(t *testing.T) {
	tmpFile := t.TempDir() + "/invalid.gz"
	if err := os.WriteFile(tmpFile, []byte("not gzip"), 0644); err != nil {
		t.Fatal(err)
	}

	bi := NewBruteIndex()
	err := bi.BuildFromGzip(tmpFile, 10)
	if err == nil {
		t.Fatal("expected error for invalid gzip")
	}
}

func TestQuantizeQuery_AllOnes(t *testing.T) {
	query := model.Vector14{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}
	qi := quantizeQuery(query)
	for d := 0; d < DIM; d++ {
		if qi[d] != int16Scale {
			t.Errorf("qi[%d] = %d, want %d", d, qi[d], int16Scale)
		}
	}
}

func TestQuantizeQuery_AllNegOnes(t *testing.T) {
	query := model.Vector14{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}
	qi := quantizeQuery(query)
	for d := 0; d < DIM; d++ {
		if qi[d] != -int16Scale {
			t.Errorf("qi[%d] = %d, want %d", d, qi[d], -int16Scale)
		}
	}
}

func TestQuantizeQuery_OutOfRange(t *testing.T) {
	query := model.Vector14{2.0, -2.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	qi := quantizeQuery(query)
	if qi[0] != int16Scale {
		t.Errorf("qi[0] (clamped) = %d, want %d", qi[0], int16Scale)
	}
	if qi[1] != -int16Scale {
		t.Errorf("qi[1] (clamped) = %d, want %d", qi[1], -int16Scale)
	}
}

func TestIVFIndex_Predict_WithBBox(t *testing.T) {
	centroids := make([]float32, 2*DIM)
	centroids[0] = 1.0

	vectors := make([]int16, 6*DIM)
	labels := make([]byte, 1)
	for i := 0; i < 3; i++ {
		vectors[i*DIM] = 9000
		labels[0] |= 1 << uint(i)
	}

	bboxMin := make([]int16, 2*DIM)
	bboxMax := make([]int16, 2*DIM)
	for d := 0; d < DIM; d++ {
		bboxMin[d] = 0
		bboxMin[DIM+d] = 0
		bboxMax[d] = 10000
		bboxMax[DIM+d] = 0
	}

	idx := &IVFIndex{
		nlist:     2,
		nprobe:    2,
		centroids: centroids,
		vectors:   vectors,
		labels:    labels,
		offsets:   []uint32{0, 3, 6},
		bboxMin:   bboxMin,
		bboxMax:   bboxMax,
	}

	score := idx.Predict(model.Vector14{0.9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 3)
	if score < 0 || score > 1 {
		t.Errorf("Predict = %v, want [0,1]", score)
	}
}

func TestScanCluster_TailOnly(t *testing.T) {
	nVecs := 3
	vectors := make([]int16, nVecs*DIM)
	labels := make([]byte, 1)
	vectors[0*DIM] = 5000
	vectors[1*DIM] = 4000
	vectors[2*DIM] = 6000

	var qi [DIM]int16
	qi[0] = 5000

	h := newTopK5()
	scanCluster(vectors, labels, 0, nVecs, qi, &h)

	if h.count == 0 {
		t.Error("should find at least one neighbor")
	}
}

func TestScanCluster_NoCloseVectors(t *testing.T) {
	nVecs := 10
	vectors := make([]int16, nVecs*DIM)
	labels := make([]byte, 2)
	for i := 0; i < nVecs; i++ {
		vectors[i*DIM] = 10000
	}

	var qi [DIM]int16
	for d := 0; d < DIM; d++ {
		qi[d] = 0
	}

	h := newTopK5()
	scanCluster(vectors, labels, 0, nVecs, qi, &h)
}

func TestSelectProbes_ExactMatch(t *testing.T) {
	nlist := 5
	centroids := make([]float32, nlist*DIM)
	centroids[0] = 0.1
	centroids[DIM] = 0.5
	centroids[2*DIM] = 0.9
	centroids[3*DIM] = 0.3
	centroids[4*DIM] = 0.7

	var query [DIM]float32
	query[0] = 0.1
	out := make([]int, nlist)

	var dist [4096]float32
	accumulateDotProducts(centroids, nlist, query, dist[:nlist])
	{ // convert dot products to L2 distances
		var norms [5]float32
		for ci := 0; ci < nlist; ci++ {
			var s float32
			for d := 0; d < DIM; d++ {
				v := centroids[d*nlist+ci]
				s += v * v
			}
			norms[ci] = s
		}
		qn := computeQueryNorm(query)
		dotToDist(dist[:nlist], nlist, qn, norms[:nlist])
	}
	selectTopN(dist[:nlist], nlist, 1, out[:1])
	if out[0] != 0 {
		t.Errorf("closest centroid = %d, want 0 (d=0)", out[0])
	}
}

func TestQuantizeQuery_ExtremeValues(t *testing.T) {
	query := model.Vector14{1.5, -1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	qi := quantizeQuery(query)
	if qi[0] != int16Scale {
		t.Errorf("qi[0] for 1.5 = %d, want %d", qi[0], int16Scale)
	}
	if qi[1] != -int16Scale {
		t.Errorf("qi[1] for -1.5 = %d, want %d", qi[1], -int16Scale)
	}
}

func TestKMeansAssign_NoChanges(t *testing.T) {
	flat := make([]float32, 2*DIM)
	centroids := make([]float32, 1*DIM)
	assign := []int{0, 0}
	changed := kmeansAssign(flat, 2, centroids, 1, assign)
	if changed != 0 {
		t.Errorf("expected 0 changes (single cluster), got %d", changed)
	}
}

func TestKMeansUpdate_EmptyCluster(t *testing.T) {
	flat := make([]float32, 2*DIM)
	centroids := make([]float32, 2*DIM)
	centroids[DIM] = 0.5
	assign := []int{0, 0}

	kmeansUpdate(flat, 2, centroids, 2, assign)
	if centroids[DIM] != 0.5 {
		t.Errorf("centroid 1 should be unchanged (empty cluster), got %v", centroids[DIM])
	}
}

func TestAccumulateDotProductsGeneric(t *testing.T) {
	nlist := 4
	centroids := make([]float32, DIM*nlist)
	for d := 0; d < DIM; d++ {
		for ci := 0; ci < nlist; ci++ {
			centroids[d*nlist+ci] = float32(d*10 + ci)
		}
	}
	var query [DIM]float32
	for d := 0; d < DIM; d++ {
		query[d] = float32(d + 1)
	}
	out := make([]float32, nlist)

	accumulateDotProductsGeneric(centroids, nlist, query, out)

	for ci := 0; ci < nlist; ci++ {
		var want float32
		for d := 0; d < DIM; d++ {
			want += query[d] * centroids[d*nlist+ci]
		}
		if out[ci] != want {
			t.Errorf("out[%d] = %f, want %f", ci, out[ci], want)
		}
	}
}

func TestScanClusterGeneric(t *testing.T) {
	nVecs := 10
	vectors := make([]int16, nVecs*DIM)
	labels := make([]byte, (nVecs+7)/8)
	for i := 0; i < 5; i++ {
		vectors[i*DIM] = 5000
	}

	var qi [DIM]int16
	qi[0] = 5000

	h := newTopK5()
	scanClusterGeneric(vectors, labels, 0, nVecs, qi, &h)

	if h.count == 0 {
		t.Error("scanClusterGeneric should find at least one close vector")
	}
	if h.dist[0] != 0 {
		t.Errorf("scanClusterGeneric closest dist = %d, want 0", h.dist[0])
	}
	if h.fraudCount(labels) != 0 {
		t.Error("scanClusterGeneric fraudCount should be 0 (no fraud labels)")
	}
}

func TestScanClusterGeneric_EmptyRange(t *testing.T) {
	vectors := make([]int16, 0)
	labels := make([]byte, 0)
	var qi [DIM]int16
	h := newTopK5()
	scanClusterGeneric(vectors, labels, 0, 0, qi, &h)
	if h.count != 0 {
		t.Errorf("count on empty generic scan = %d, want 0", h.count)
	}
}
