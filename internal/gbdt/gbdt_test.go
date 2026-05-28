package gbdt

import (
	"math"
	"os"
	"testing"
)

func writeV2Model(t *testing.T, path string, trees []struct {
	nodes []struct {
		feat  uint8
		left  uint16
		right uint16
		leaf  uint8
		value float32
	}
}) {
	t.Helper()
	data := make([]byte, 0, 1024)
	data = append(data, []byte("GBDT")...)
	data = append(data, byte(2), 0, 0, 0)          // version
	data = append(data, byte(len(trees)), 0, 0, 0) // numTrees
	data = append(data, byte(14), 0, 0, 0)         // numFeatures
	data = append(data, 0, 0, 0, 0)                // initPred = 0
	data = append(data, 1, 0, 0, 0)                // sigmoid=1, pad

	for _, tr := range trees {
		data = append(data, byte(len(tr.nodes)), 0, 0, 0)
		for _, n := range tr.nodes {
			data = append(data, n.feat)
			data = append(data, 0) // pad
			data = append(data, byte(n.left), byte(n.left>>8))
			data = append(data, byte(n.right), byte(n.right>>8))
			data = append(data, n.leaf)
			data = append(data, 0) // pad
			v := math.Float32bits(n.value)
			data = append(data, byte(v), byte(v>>8), byte(v>>16), byte(v>>24))
		}
	}
	if err := os.WriteFile(path, data, 0644); err != nil {
		t.Fatalf("writeV2Model: %v", err)
	}
}

func TestLoad_InvalidMagic(t *testing.T) {
	path := t.TempDir() + "/badmagic.bin"
	if err := os.WriteFile(path, []byte("XXXX"), 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	_, err := Load(path)
	if err == nil {
		t.Fatal("expected error for invalid magic")
	}
}

func TestLoad_ShortFile(t *testing.T) {
	path := t.TempDir() + "/short.bin"
	if err := os.WriteFile(path, []byte("G"), 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	_, err := Load(path)
	if err == nil {
		t.Fatal("expected error for short file")
	}
}

func TestLoad_NotFound(t *testing.T) {
	_, err := Load("/nonexistent/path.bin")
	if err == nil {
		t.Fatal("expected error for nonexistent file")
	}
}

func TestLoadV1_ShortFile(t *testing.T) {
	path := t.TempDir() + "/v1short.bin"
	data := make([]byte, 10)
	copy(data, "GBDT")
	if err := os.WriteFile(path, data, 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	_, err := Load(path)
	if err == nil {
		t.Fatal("expected error for short v1 file")
	}
}

func TestLoadV1_TruncatedNode(t *testing.T) {
	path := t.TempDir() + "/v1trunc.bin"
	data := make([]byte, 32)
	copy(data, "GBDT")
	data[8] = 1     // numTrees = 1
	data[12] = 0x40 // lr = 0.1 (approx)
	data[20] = 0    // initPred = 0
	// numNodes = 1 but not enough bytes
	data[28] = 1
	if err := os.WriteFile(path, data, 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	_, err := Load(path)
	if err == nil {
		t.Fatal("expected error for truncated v1 node")
	}
}

func TestLoadV2_ShortFile(t *testing.T) {
	path := t.TempDir() + "/v2short.bin"
	data := make([]byte, 20)
	copy(data, "GBDT")
	data[4] = 2 // version
	if err := os.WriteFile(path, data, 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	_, err := Load(path)
	if err == nil {
		t.Fatal("expected error for short v2 file")
	}
}

func TestLoadV2_TruncatedNode(t *testing.T) {
	path := t.TempDir() + "/v2trunc.bin"
	data := make([]byte, 28)
	copy(data, "GBDT")
	data[4] = 2   // version
	data[8] = 1   // numTrees = 1
	data[12] = 14 // numFeatures
	// numNodes = 1 but no node data
	data[24] = 1
	if err := os.WriteFile(path, data, 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	_, err := Load(path)
	if err == nil {
		t.Fatal("expected error for truncated v2 node")
	}
}

func TestGBDT_Predict_Sigmoid(t *testing.T) {
	// Single tree: single leaf node with value 2.0
	trees := []struct {
		nodes []struct {
			feat  uint8
			left  uint16
			right uint16
			leaf  uint8
			value float32
		}
	}{
		{
			nodes: []struct {
				feat  uint8
				left  uint16
				right uint16
				leaf  uint8
				value float32
			}{
				{feat: 0, left: 0, right: 0, leaf: 1, value: 2.0},
			},
		},
	}

	path := t.TempDir() + "/sigmoid.bin"
	writeV2Model(t, path, trees)

	g, err := Load(path)
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	got := g.Predict([14]float32{0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})
	expected := float32(1.0 / (1.0 + math.Exp(-2.0)))
	if math.Abs(float64(got-expected)) > 0.0001 {
		t.Errorf("Predict = %v, want %v", got, expected)
	}
}

func TestGBDT_Predict_NegativeSigmoid(t *testing.T) {
	trees := []struct {
		nodes []struct {
			feat  uint8
			left  uint16
			right uint16
			leaf  uint8
			value float32
		}
	}{
		{
			nodes: []struct {
				feat  uint8
				left  uint16
				right uint16
				leaf  uint8
				value float32
			}{
				{feat: 0, left: 0, right: 0, leaf: 1, value: -2.0},
			},
		},
	}

	path := t.TempDir() + "/neg_sigmoid.bin"
	writeV2Model(t, path, trees)

	g, err := Load(path)
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	got := g.Predict([14]float32{0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})
	expected := float32(1.0 / (1.0 + math.Exp(2.0)))
	if math.Abs(float64(got-expected)) > 0.0001 {
		t.Errorf("Predict = %v, want %v", got, expected)
	}
}

func TestGBDT_Predict_NoSigmoid(t *testing.T) {
	path := t.TempDir() + "/no_sigmoid.bin"
	data := make([]byte, 0, 256)
	data = append(data, []byte("GBDT")...)
	data = append(data, byte(2), 0, 0, 0)
	data = append(data, byte(1), 0, 0, 0)  // 1 tree
	data = append(data, byte(14), 0, 0, 0) // numFeatures
	data = append(data, 0, 0, 0, 0)        // initPred = 0
	data = append(data, 0, 0, 0, 0)        // sigmoid=0, pad
	data = append(data, byte(1), 0, 0, 0)  // 1 node
	data = append(data, 0)                 // feat=0
	data = append(data, 0)                 // pad
	data = append(data, 0, 0)              // left=0
	data = append(data, 0, 0)              // right=0
	data = append(data, 1)                 // leaf=1
	data = append(data, 0)                 // pad
	v := math.Float32bits(0.75)
	data = append(data, byte(v), byte(v>>8), byte(v>>16), byte(v>>24))
	if err := os.WriteFile(path, data, 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	g, err := Load(path)
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	got := g.Predict([14]float32{0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})
	if math.Abs(float64(got-0.75)) > 0.0001 {
		t.Errorf("Predict = %v, want 0.75", got)
	}
}

func TestGBDT_Predict_Clamp(t *testing.T) {
	g := &GBDT{
		trees:       nil,
		initPred:    1.5,
		sigmoid:     false,
		numFeatures: 14,
		NumTrees:    0,
	}
	got := g.Predict([14]float32{})
	if got != 1.0 {
		t.Errorf("Predict = %v, want 1.0 (clamped)", got)
	}

	g.initPred = -0.5
	got = g.Predict([14]float32{})
	if got != 0.0 {
		t.Errorf("Predict = %v, want 0.0 (clamped)", got)
	}
}

func TestGBDT_Predict_SigmoidClipping(t *testing.T) {
	g := &GBDT{
		trees:       nil,
		initPred:    25,
		sigmoid:     true,
		numFeatures: 14,
		NumTrees:    0,
	}
	got := g.Predict([14]float32{})
	if got != 1.0 {
		t.Errorf("Predict = %v, want 1.0 (sigmoid clipped high)", got)
	}

	g.initPred = -25
	got = g.Predict([14]float32{})
	if got != 0.0 {
		t.Errorf("Predict = %v, want 0.0 (sigmoid clipped low)", got)
	}
}

func TestGBDT_Predict_TreeBranching(t *testing.T) {
	trees := []struct {
		nodes []struct {
			feat  uint8
			left  uint16
			right uint16
			leaf  uint8
			value float32
		}
	}{
		{
			nodes: []struct {
				feat  uint8
				left  uint16
				right uint16
				leaf  uint8
				value float32
			}{
				{feat: 0, left: 1, right: 2, leaf: 0, value: 0.5},
				{feat: 0, left: 0, right: 0, leaf: 1, value: 0.1},
				{feat: 0, left: 0, right: 0, leaf: 1, value: 0.9},
			},
		},
	}

	path := t.TempDir() + "/tree_branch.bin"
	writeV2Model(t, path, trees)

	g, err := Load(path)
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	gotLeft := g.Predict([14]float32{0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})
	expectedLeft := float32(1.0 / (1.0 + math.Exp(-0.1)))
	if math.Abs(float64(gotLeft-expectedLeft)) > 0.0001 {
		t.Errorf("Predict left branch = %v, want %v", gotLeft, expectedLeft)
	}

	gotRight := g.Predict([14]float32{1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})
	expectedRight := float32(1.0 / (1.0 + math.Exp(-0.9)))
	if math.Abs(float64(gotRight-expectedRight)) > 0.0001 {
		t.Errorf("Predict right branch = %v, want %v", gotRight, expectedRight)
	}
}

func TestGBDT_LoadV1(t *testing.T) {
	path := t.TempDir() + "/v1_model.bin"
	data := make([]byte, 0, 256)
	data = append(data, []byte("GBDT")...)
	data = append(data, 0, 0, 0, 0)       // version=0 → triggers loadV1
	data = append(data, byte(1), 0, 0, 0) // 1 tree
	// lr = 0.5 as float64
	lr := math.Float64bits(0.5)
	data = append(data, byte(lr), byte(lr>>8), byte(lr>>16), byte(lr>>24), byte(lr>>32), byte(lr>>40), byte(lr>>48), byte(lr>>56))
	// initPred = 0.0
	init := math.Float64bits(0.0)
	data = append(data, byte(init), byte(init>>8), byte(init>>16), byte(init>>24), byte(init>>32), byte(init>>40), byte(init>>48), byte(init>>56))
	// 1 node
	data = append(data, byte(1), 0, 0, 0)
	// v1 node: feat(1) + thresh(8) + left(2) + right(2) + value(4) + leaf(1) = 18
	data = append(data, 0) // feat=0
	thresh := math.Float64bits(0.5)
	data = append(data, byte(thresh), byte(thresh>>8), byte(thresh>>16), byte(thresh>>24), byte(thresh>>32), byte(thresh>>40), byte(thresh>>48), byte(thresh>>56))
	data = append(data, 0, 0) // left=0
	data = append(data, 0, 0) // right=0
	lv := math.Float32bits(2.0)
	data = append(data, byte(lv), byte(lv>>8), byte(lv>>16), byte(lv>>24))
	data = append(data, 1) // leaf=1
	if err := os.WriteFile(path, data, 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	g, err := Load(path)
	if err != nil {
		t.Fatalf("Load v1 failed: %v", err)
	}

	if g.NumTrees != 1 {
		t.Errorf("NumTrees = %d, want 1", g.NumTrees)
	}
	if !g.sigmoid {
		t.Error("v1 should have sigmoid=true")
	}
}

func TestLoadV1_TruncatedTree(t *testing.T) {
	path := t.TempDir() + "/v1_trunc_tree.bin"
	data := make([]byte, 32)
	copy(data, "GBDT")
	data[8] = 2 // numTrees = 2
	lr := math.Float64bits(0.1)
	copy(data[12:20], data8(lr))
	init := math.Float64bits(0.0)
	copy(data[20:28], data8(init))
	// Only write first tree's numNodes, not enough data for second tree
	data[28] = 1 // tree 0: 1 node
	// need 18 bytes for that node
	copy(data[32:], []byte{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1})
	if err := os.WriteFile(path, data, 50); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	_, err := Load(path)
	if err == nil {
		t.Fatal("expected error for truncated v1 tree")
	}
}
func data8(v uint64) []byte {
	return []byte{byte(v), byte(v >> 8), byte(v >> 16), byte(v >> 24), byte(v >> 32), byte(v >> 40), byte(v >> 48), byte(v >> 56)}
}

func TestGBDT_Sigmoid(t *testing.T) {
	g := &GBDT{sigmoid: true, numFeatures: 14, NumTrees: 1}
	if !g.Sigmoid() {
		t.Error("Sigmoid() = false, want true")
	}
	if g.NumFeatures() != 14 {
		t.Errorf("NumFeatures() = %d, want 14", g.NumFeatures())
	}

	g.sigmoid = false
	if g.Sigmoid() {
		t.Error("Sigmoid() = true, want false")
	}
}

func TestGBDT_Predict_RegressionMultiTree(t *testing.T) {
	// Two trees, each a single leaf node, sigmoid=false.
	// Tree 0: value 0.2, Tree 1: value 0.6
	// Ensemble average: (0.2 + 0.6) / 2 = 0.4
	path := t.TempDir() + "/rf_avg.bin"
	data := make([]byte, 0, 256)
	data = append(data, []byte("GBDT")...)
	data = append(data, byte(2), 0, 0, 0)  // version=2
	data = append(data, byte(2), 0, 0, 0)  // numTrees=2
	data = append(data, byte(14), 0, 0, 0) // numFeatures=14
	data = append(data, 0, 0, 0, 0)        // initPred=0
	data = append(data, 0, 0, 0, 0)        // sigmoid=0, pad

	writeTreeNode := func(value float32) {
		data = append(data, byte(1), 0, 0, 0) // 1 node
		data = append(data, 0)                // feat=0
		data = append(data, 0)                // pad
		data = append(data, 0, 0)             // left=0
		data = append(data, 0, 0)             // right=0
		data = append(data, 1)                // leaf=1
		data = append(data, 0)                // pad
		v := math.Float32bits(value)
		data = append(data, byte(v), byte(v>>8), byte(v>>16), byte(v>>24))
	}
	writeTreeNode(0.2)
	writeTreeNode(0.6)

	if err := os.WriteFile(path, data, 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	g, err := Load(path)
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	got := g.Predict([14]float32{})
	want := float32(0.4)
	if math.Abs(float64(got)-float64(want)) > 0.0001 {
		t.Errorf("Predict = %v, want %v", got, want)
	}
}

// testTrainAndExport is a helper that trains a decision tree on the given
// samples and exports it, then loads the model back.
func testTrainAndExport(t *testing.T, samples []Sample, maxDepth, minSamples int, minGain float64) *GBDT {
	t.Helper()
	path := t.TempDir() + "/trained.bin"
	if err := TrainAndExport(samples, path, maxDepth, minSamples, minGain); err != nil {
		t.Fatalf("TrainAndExport: %v", err)
	}
	g, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	return g
}

func TestTrainAndExport_SmallTree(t *testing.T) {
	// 200 samples: first 100 fraud (label=true), last 100 legit (label=false)
	// Feature 0 is the only discriminant: fraud has high value (0.9), legit has low value (0.1).
	// All other features are set to 0 to avoid accidental separation.
	samples := make([]Sample, 200)
	for i := range 100 {
		samples[i] = Sample{
			Vector: [14]float32{0.9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			Label:  true,
		}
	}
	for i := 100; i < 200; i++ {
		samples[i] = Sample{
			Vector: [14]float32{0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			Label:  false,
		}
	}

	g := testTrainAndExport(t, samples, 4, 10, 0.01)

	// Fraud sample should predict > 0.5
	gotFraud := g.Predict([14]float32{0.9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})
	if gotFraud <= 0.5 {
		t.Errorf("fraud sample predicted %v, want > 0.5", gotFraud)
	}

	// Legit sample should predict < 0.5
	gotLegit := g.Predict([14]float32{0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})
	if gotLegit >= 0.5 {
		t.Errorf("legit sample predicted %v, want < 0.5", gotLegit)
	}
}

func TestTrainAndExport_UniformLabel(t *testing.T) {
	// All fraud — tree should be a single leaf with value 1.0
	samples := make([]Sample, 50)
	for i := range 50 {
		samples[i] = Sample{Label: true}
	}
	g := testTrainAndExport(t, samples, 4, 5, 0.01)
	got := g.Predict([14]float32{})
	if got != 1.0 {
		t.Errorf("uniform fraud predicted %v, want 1.0", got)
	}
}

func TestTrainRandomForest(t *testing.T) {
	samples := make([]Sample, 200)
	for i := range 100 {
		samples[i] = Sample{
			Vector: [14]float32{0.9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			Label:  true,
		}
	}
	for i := 100; i < 200; i++ {
		samples[i] = Sample{
			Vector: [14]float32{0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			Label:  false,
		}
	}

	path := t.TempDir() + "/rf_trained.bin"
	if err := TrainRandomForest(samples, path, 3, 3, 10, 0.01, 42, 100); err != nil {
		t.Fatalf("TrainRandomForest: %v", err)
	}
	g, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if g.NumTrees != 3 {
		t.Errorf("NumTrees = %d, want 3", g.NumTrees)
	}

	// Fraud sample
	gotFraud := g.Predict([14]float32{0.9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})
	if gotFraud <= 0.5 {
		t.Errorf("fraud sample predicted %v, want > 0.5", gotFraud)
	}
}

func TestGiniImpurity(t *testing.T) {
	tests := []struct {
		fraud, total int
		want         float64
	}{
		{0, 100, 0.0},    // pure legit
		{100, 100, 0.0},  // pure fraud
		{50, 100, 0.5},   // 50/50 split -> 1 - 0.25 - 0.25 = 0.5
		{25, 100, 0.375}, // 1 - 0.0625 - 0.5625 = 0.375
		{0, 0, 0.0},      // edge: empty
	}
	for _, tt := range tests {
		got := giniImpurity(tt.fraud, tt.total)
		if math.Abs(got-tt.want) > 1e-10 {
			t.Errorf("giniImpurity(%d,%d) = %v, want %v", tt.fraud, tt.total, got, tt.want)
		}
	}
}

func TestExportV2Multi_RoundTrip(t *testing.T) {
	// Create a flat node structure matching what flattenTree produces
	// One tree: internal node split on feat=0 at thresh=0.5, left leaf=0.2, right leaf=0.8
	flatTree := []flatNode{
		{feat: 0, thresh: 0.5, left: 1, right: 2}, // root
		{leaf: true, value: 0.2},                  // left child
		{leaf: true, value: 0.8},                  // right child
	}

	path := t.TempDir() + "/rt.bin"
	if err := exportV2Multi(path, [][]flatNode{flatTree}); err != nil {
		t.Fatalf("exportV2Multi: %v", err)
	}

	g, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	gotLeft := g.Predict([14]float32{0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})
	if gotLeft != 0.2 {
		t.Errorf("left branch predict = %v, want 0.2", gotLeft)
	}
	gotRight := g.Predict([14]float32{1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})
	if gotRight != 0.8 {
		t.Errorf("right branch predict = %v, want 0.8", gotRight)
	}
}

func TestTrainAndExport_Empty(t *testing.T) {
	// Empty sample set produces a single leaf with NaN value (0/0).
	path := t.TempDir() + "/empty.bin"
	if err := TrainAndExport(nil, path, 4, 5, 0.01); err != nil {
		t.Fatalf("TrainAndExport nil: %v", err)
	}
	g, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	got := g.Predict([14]float32{})
	if !math.IsNaN(float64(got)) {
		t.Errorf("expected NaN for empty training set, got %v", got)
	}
}

func TestFindBestSplit_AllSameLabel(t *testing.T) {
	samples := make([]Sample, 20)
	for i := range 20 {
		samples[i] = Sample{Label: true}
	}
	indices := make([]int, 20)
	for i := range indices {
		indices[i] = i
	}
	feat, thresh, gain := findBestSplit(samples, indices)
	if feat != -1 {
		t.Errorf("findBestSplit with uniform labels: feat=%d, want -1", feat)
	}
	if thresh != 0 {
		t.Errorf("findBestSplit with uniform labels: thresh=%v, want 0", thresh)
	}
	if gain != 0 {
		t.Errorf("findBestSplit with uniform labels: gain=%v, want 0", gain)
	}
}

func TestPartition(t *testing.T) {
	samples := []Sample{
		{Vector: [14]float32{0.1}},
		{Vector: [14]float32{0.9}},
		{Vector: [14]float32{0.3}},
		{Vector: [14]float32{0.7}},
	}
	indices := []int{0, 1, 2, 3}
	split := partition(indices, samples, 0, 0.5)
	// After partition: indices[:split] should have values <= 0.5
	for i := range split {
		if samples[indices[i]].Vector[0] > 0.5 {
			t.Errorf("index %d (val=%v) should be in left (<=0.5) but is in right", i, samples[indices[i]].Vector[0])
		}
	}
	for i := split; i < len(indices); i++ {
		if samples[indices[i]].Vector[0] <= 0.5 {
			t.Errorf("index %d (val=%v) should be in right (>0.5) but is in left", i, samples[indices[i]].Vector[0])
		}
	}
}
