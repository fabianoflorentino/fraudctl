package knn

import (
	"container/heap"
	"math"
	"math/rand"
	"testing"

	"github.com/fabianoflorentino/fraudctl/internal/model"
)

func TestQuantizeFloat32(t *testing.T) {
	tests := []struct {
		v    float32
		want int16
	}{
		{0.0, 0},
		{1.0, int16Scale},
		{0.5, int16Scale / 2},
		{-0.5, -int16Scale / 2},
		{-1.0, -int16Scale},
		{1.5, int16Scale},   // clamped
		{-1.5, -int16Scale}, // clamped
	}
	for _, tt := range tests {
		got := quantizeFloat32(tt.v)
		if got != tt.want {
			t.Errorf("quantizeFloat32(%v) = %d, want %d", tt.v, got, tt.want)
		}
	}
}

func TestQuantizeQuery(t *testing.T) {
	query := model.Vector14{0.5, -0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	qi := quantizeQuery(query)
	want0 := quantizeFloat32(0.5)
	if qi[0] != want0 {
		t.Errorf("qi[0] = %d, want %d", qi[0], want0)
	}
	want1 := quantizeFloat32(-0.5)
	if qi[1] != want1 {
		t.Errorf("qi[1] = %d, want %d", qi[1], want1)
	}
}

func TestTopK5(t *testing.T) {
	h := newTopK5()
	if h.count != 0 {
		t.Errorf("initial count = %d, want 0", h.count)
	}
	if h.worstDist() != math.MaxUint64 {
		t.Errorf("initial worstDist = %d, want MaxUint64", h.worstDist())
	}

	h.tryInsert(100, 0)
	if h.count != 1 {
		t.Errorf("count after insert = %d, want 1", h.count)
	}
	if h.dist[0] != 100 {
		t.Errorf("dist[0] = %d, want 100", h.dist[0])
	}

	h.tryInsert(50, 1)
	if h.dist[0] != 50 {
		t.Errorf("dist[0] after second insert = %d, want 50", h.dist[0])
	}
	if h.idx[0] != 1 {
		t.Errorf("idx[0] = %d, want 1", h.idx[0])
	}

	// Fill all K slots
	for i := 2; i < K; i++ {
		h.tryInsert(uint64(200+i), i)
	}
	if h.count != K {
		t.Errorf("count = %d, want %d", h.count, K)
	}

	// Insert worse than worst should be ignored
	h.tryInsert(999999, 99)
	last := h.dist[K-1]
	if last == 999999 {
		t.Errorf("worst insert should not replace")
	}
}

func TestTopK5_FraudCount(t *testing.T) {
	labels := make([]byte, 1)
	labels[0] = 0b00000001 // idx 0 is fraud
	h := newTopK5()

	h.tryInsert(10, 0) // fraud
	h.tryInsert(20, 1) // legit
	h.tryInsert(30, 2) // legit

	if n := h.fraudCount(labels); n != 1 {
		t.Errorf("fraudCount = %d, want 1", n)
	}
}

func TestBruteIndex_PredictRaw(t *testing.T) {
	b := NewBruteIndex()
	vectors := []model.Vector14{
		{0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		{0.9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		{0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		{0.8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		{0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	}
	labels := []bool{false, true, false, true, false}
	b.Build(vectors, labels)

	raw := b.PredictRaw(model.Vector14{0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 0)
	if raw < 0 || raw > 5 {
		t.Errorf("PredictRaw = %d, want [0,5]", raw)
	}
}

func TestBruteIndex_NProbe(t *testing.T) {
	b := NewBruteIndex()
	if b.NProbe() != 0 {
		t.Errorf("NProbe = %d, want 0", b.NProbe())
	}
}

func TestBruteIndex_CountFraudCount(t *testing.T) {
	b := NewBruteIndex()
	if b.Count() != 0 {
		t.Errorf("Count = %d, want 0", b.Count())
	}
	if b.FraudCount() != 0 {
		t.Errorf("FraudCount = %d, want 0", b.FraudCount())
	}

	vectors := []model.Vector14{{0.1}, {0.9}, {0.5}}
	labels := []bool{false, true, false}
	b.Build(vectors, labels)

	if b.Count() != 3 {
		t.Errorf("Count = %d, want 3", b.Count())
	}
	if b.FraudCount() != 1 {
		t.Errorf("FraudCount = %d, want 1", b.FraudCount())
	}
}

func TestBruteIndex_Predict_Empty(t *testing.T) {
	b := NewBruteIndex()
	score := b.Predict(model.Vector14{}, 5)
	if score != 0 {
		t.Errorf("Predict empty = %v, want 0", score)
	}
}

func TestIVFIndex_PredictRaw_Empty(t *testing.T) {
	idx := &IVFIndex{}
	raw := idx.PredictRaw(model.Vector14{}, 0)
	if raw != 0 {
		t.Errorf("PredictRaw empty = %d, want 0", raw)
	}
}

func TestIVFIndex_SetRetry(t *testing.T) {
	idx := NewIVFIndex()
	idx.SetRetry(16, 2, 3)
	if idx.quickProbe != 16 {
		t.Errorf("quickProbe = %d, want 16", idx.quickProbe)
	}
	if idx.boundaryLo != 2 {
		t.Errorf("boundaryLo = %d, want 2", idx.boundaryLo)
	}
	if idx.boundaryHi != 3 {
		t.Errorf("boundaryHi = %d, want 3", idx.boundaryHi)
	}
}

func TestNewBruteIndex(t *testing.T) {
	b := NewBruteIndex()
	if b == nil {
		t.Fatal("NewBruteIndex returned nil")
	}
	if b.flat != nil {
		t.Error("flat should be nil on init")
	}
}

func TestNewIVFIndex(t *testing.T) {
	idx := NewIVFIndex()
	if idx == nil {
		t.Fatal("NewIVFIndex returned nil")
	}
}

func TestBrutePredict_Empty(t *testing.T) {
	score := brutePredict(nil, nil, 0, model.Vector14{}, 5)
	if score != 0 {
		t.Errorf("brutePredict empty = %v, want 0", score)
	}
}

func TestBrutePredict_ExactMatch(t *testing.T) {
	flat := []float32{0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	flags := []bool{true}
	score := brutePredict(flat, flags, 1, model.Vector14{0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 1)
	if score != 1.0 {
		t.Errorf("brutePredict exact match = %v, want 1.0", score)
	}
}

func TestBrutePredict_FarAway(t *testing.T) {
	flat := []float32{0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	flags := []bool{false}
	score := brutePredict(flat, flags, 1, model.Vector14{1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 1)
	if score != 0.0 {
		t.Errorf("brutePredict far = %v, want 0.0", score)
	}
}

func TestMaxHeap(t *testing.T) {
	h := make(maxHeap, 0, 3)
	heap.Push(&h, candidate{5, 0})
	heap.Push(&h, candidate{1, 1})
	heap.Push(&h, candidate{3, 2})

	if h[0].dist != 5 {
		t.Errorf("maxHeap top dist = %v, want 5", h[0].dist)
	}
}

func TestBBoxMayImprove(t *testing.T) {
	nlist := 2
	bboxMin := make([]int16, nlist*DIM)
	bboxMax := make([]int16, nlist*DIM)
	for d := 0; d < DIM; d++ {
		bboxMin[d] = 0
		bboxMax[d] = 100
	}

	var qi [DIM]int16
	qi[5] = 50
	qi[6] = 50

	// Should return true since query is within bbox
	result := bboxMayImprove(bboxMin, bboxMax, 0, qi, math.MaxUint64)
	if !result {
		t.Error("bboxMayImprove should return true when query is within bbox")
	}

	// Query far outside bbox in high-variance dims
	qi[5] = 1000
	result = bboxMayImprove(bboxMin, bboxMax, 0, qi, 0)
	if result {
		t.Error("bboxMayImprove should return false when query is far outside bbox and worstDist=0")
	}
}

func TestSelectProbes(t *testing.T) {
	nlist := 4
	centroids := make([]float32, nlist*DIM)
	centroids[0] = 0.1
	centroids[DIM] = 0.5
	centroids[2*DIM] = 0.9
	centroids[3*DIM] = 0.0

	var query [DIM]float32
	out := make([]int, nlist)

	selectProbes(centroids, nlist, query, 2, out[:2])

	if out[0] != 3 || out[1] != 0 {
		t.Logf("selectProbes result = %v (expected closest to centroids[3]=0.0 and [0]=0.1)", out[:2])
	}
}

func TestSelectProbes_NprobeGreaterThanNlist(t *testing.T) {
	nlist := 3
	centroids := make([]float32, nlist*DIM)
	var query [DIM]float32
	out := make([]int, nlist+2)

	selectProbes(centroids, nlist, query, 10, out)
	for i := 0; i < nlist; i++ {
		if out[i] < 0 || out[i] >= nlist {
			t.Errorf("out[%d] = %d, want [0,%d)", i, out[i], nlist)
		}
	}
}

func TestScanCluster(t *testing.T) {
	nVecs := 20
	vectors := make([]int16, nVecs*DIM)
	labels := make([]byte, (nVecs+7)/8)

	// Put close vectors at indices 0-4 (dim 0 = 5000 = 0.5 * int16Scale)
	for i := 0; i < 5; i++ {
		vectors[i*DIM] = 5000
	}

	var qi [DIM]int16
	qi[0] = 5000

	h := newTopK5()
	scanCluster(vectors, labels, 0, nVecs, qi, &h)

	if h.count == 0 {
		t.Error("scanCluster should find at least one close vector")
	}
	if h.dist[0] != 0 {
		t.Errorf("scanCluster closest dist = %d, want 0 (exact match)", h.dist[0])
	}
}

func TestScanCluster_Empty(t *testing.T) {
	vectors := make([]int16, 0)
	labels := make([]byte, 0)
	var qi [DIM]int16
	h := newTopK5()
	scanCluster(vectors, labels, 0, 0, qi, &h)
	if h.count != 0 {
		t.Errorf("count on empty scan = %d, want 0", h.count)
	}
}

func TestIVFIndex_NProbe(t *testing.T) {
	idx := NewIVFIndex()
	if idx.NProbe() != 0 {
		t.Errorf("default NProbe = %d, want 0", idx.NProbe())
	}
	idx.SetNProbe(48)
	if idx.NProbe() != 48 {
		t.Errorf("NProbe after SetNProbe = %d, want 48", idx.NProbe())
	}
}

func TestIVFIndex_PredictRaw_NoNprobe(t *testing.T) {
	idx := &IVFIndex{
		nlist:     2,
		centroids: make([]float32, 2*DIM),
		vectors:   make([]int16, 6*DIM),
		labels:    make([]byte, 1),
		offsets:   []uint32{0, 3, 6},
	}
	raw := idx.PredictRaw(model.Vector14{}, 0)
	if raw < 0 || raw > 5 {
		t.Errorf("PredictRaw = %d, want [0,5]", raw)
	}
}

func TestIVFIndex_DebugCentroids(t *testing.T) {
	idx := NewIVFIndex()
	c := idx.DebugCentroids()
	if c != nil {
		t.Errorf("DebugCentroids on new index = %v, want nil", c)
	}
}

func TestBuildIVF_NoFile(t *testing.T) {
	err := BuildIVF("/nonexistent/references.json.gz", "/tmp/ivf.bin", 10, 5)
	if err == nil {
		t.Fatal("expected error for nonexistent input")
	}
}

func TestIVFIndex_PredictRaw_WithBBox(t *testing.T) {
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

	raw := idx.PredictRaw(model.Vector14{0.9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 2)
	if raw < 0 || raw > 5 {
		t.Errorf("PredictRaw with bbox = %d, want [0,5]", raw)
	}
}

func TestKMeansInit(t *testing.T) {
	flat := make([]float32, 4*DIM)
	for i := 0; i < 4; i++ {
		flat[i*DIM] = float32(i) * 0.25
	}
	centroids := kmeansInit(flat, 4, 2)
	if len(centroids) != 2*DIM {
		t.Errorf("centroids len = %d, want %d", len(centroids), 2*DIM)
	}
}

func TestKMeansAssign(t *testing.T) {
	flat := make([]float32, 4*DIM)
	flat[0] = 0.1
	flat[DIM] = 0.9
	flat[2*DIM] = 0.2
	flat[3*DIM] = 0.8

	centroids := make([]float32, 2*DIM)
	centroids[0] = 0.1
	centroids[DIM] = 0.9

	assign := make([]int, 4)
	changed := kmeansAssign(flat, 4, centroids, 2, assign)
	if changed <= 0 {
		t.Errorf("expected changes, got %d", changed)
	}
}

func TestKMeansUpdate(t *testing.T) {
	flat := make([]float32, 4*DIM)
	for i := 0; i < 4; i++ {
		flat[i*DIM] = float32(i) * 0.25
	}
	centroids := make([]float32, 2*DIM)
	assign := []int{0, 0, 1, 1}

	kmeansUpdate(flat, 4, centroids, 2, assign)
	if centroids[0] != 0.125 {
		t.Errorf("centroid[0][0] = %v, want 0.125", centroids[0])
	}
}

func BenchmarkBrutePredict(b *testing.B) {
	n := 10000
	flat := make([]float32, n*14)
	flags := make([]bool, n)
	rng := rand.New(rand.NewSource(42))
	for i := 0; i < n; i++ {
		for d := 0; d < 14; d++ {
			flat[i*14+d] = rng.Float32()
		}
		flags[i] = rng.Float32() > 0.5
	}

	var query model.Vector14
	for d := 0; d < 14; d++ {
		query[d] = rng.Float32()
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		brutePredict(flat, flags, n, query, 5)
	}
}

func BenchmarkQuantizeFloat32(b *testing.B) {
	v := float32(0.753)
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		quantizeFloat32(v)
	}
}

func BenchmarkTopK5(b *testing.B) {
	h := newTopK5()
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		h.tryInsert(uint64(i), i)
	}
}
