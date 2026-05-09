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
	data = append(data, byte(2), 0, 0, 0) // version
	data = append(data, byte(len(trees)), 0, 0, 0) // numTrees
	data = append(data, byte(14), 0, 0, 0) // numFeatures
	data = append(data, 0, 0, 0, 0) // initPred = 0
	data = append(data, 1, 0, 0, 0) // sigmoid=1, pad

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
	os.WriteFile(path, []byte("XXXX"), 0644)
	_, err := Load(path)
	if err == nil {
		t.Fatal("expected error for invalid magic")
	}
}

func TestLoad_ShortFile(t *testing.T) {
	path := t.TempDir() + "/short.bin"
	os.WriteFile(path, []byte("G"), 0644)
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
	os.WriteFile(path, data, 0644)
	_, err := Load(path)
	if err == nil {
		t.Fatal("expected error for short v1 file")
	}
}

func TestLoadV1_TruncatedNode(t *testing.T) {
	path := t.TempDir() + "/v1trunc.bin"
	data := make([]byte, 32)
	copy(data, "GBDT")
	data[8] = 1 // numTrees = 1
	data[12] = 0x40 // lr = 0.1 (approx)
	data[20] = 0 // initPred = 0
	// numNodes = 1 but not enough bytes
	data[28] = 1
	os.WriteFile(path, data, 0644)
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
	os.WriteFile(path, data, 0644)
	_, err := Load(path)
	if err == nil {
		t.Fatal("expected error for short v2 file")
	}
}

func TestLoadV2_TruncatedNode(t *testing.T) {
	path := t.TempDir() + "/v2trunc.bin"
	data := make([]byte, 28)
	copy(data, "GBDT")
	data[4] = 2 // version
	data[8] = 1 // numTrees = 1
	data[12] = 14 // numFeatures
	// numNodes = 1 but no node data
	data[24] = 1
	os.WriteFile(path, data, 0644)
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
	data = append(data, byte(1), 0, 0, 0) // 1 tree
	data = append(data, byte(14), 0, 0, 0) // numFeatures
	data = append(data, 0, 0, 0, 0) // initPred = 0
	data = append(data, 0, 0, 0, 0) // sigmoid=0, pad
	data = append(data, byte(1), 0, 0, 0) // 1 node
	data = append(data, 0) // feat=0
	data = append(data, 0) // pad
	data = append(data, 0, 0) // left=0
	data = append(data, 0, 0) // right=0
	data = append(data, 1) // leaf=1
	data = append(data, 0) // pad
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
	data = append(data, 0, 0, 0, 0) // version=0 → triggers loadV1
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
	os.WriteFile(path, data, 50)
	_, err := Load(path)
	if err == nil {
		t.Fatal("expected error for truncated v1 tree")
	}
}
func data8(v uint64) []byte {
	return []byte{byte(v), byte(v >> 8), byte(v >> 16), byte(v >> 24), byte(v >> 32), byte(v >> 40), byte(v >> 48), byte(v >> 56)}
}
