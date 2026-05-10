package logreg

import (
	"bytes"
	"math"
	"os"
	"testing"

	"github.com/fabianoflorentino/fraudctl/internal/model"
)

func TestNewModel(t *testing.T) {
	m := NewModel()
	if m.dim != dimExpanded {
		t.Errorf("dim = %d, want %d", m.dim, dimExpanded)
	}
	if len(m.weights) != dimExpanded {
		t.Errorf("len(weights) = %d, want %d", len(m.weights), dimExpanded)
	}
	if m.bias != 0 {
		t.Errorf("bias = %v, want 0", m.bias)
	}
}

func TestSigmoid(t *testing.T) {
	tests := []struct {
		x    float64
		want float64
	}{
		{0, 0.5},
		{20, 1.0},
		{-20, 0.0},
		{1, 1 / (1 + math.Exp(-1))},
		{-1, 1 / (1 + math.Exp(1))},
		{100, 1.0},
		{-100, 0.0},
	}
	for _, tt := range tests {
		got := sigmoid(tt.x)
		if math.Abs(got-tt.want) > 0.0001 {
			t.Errorf("sigmoid(%v) = %v, want %v", tt.x, got, tt.want)
		}
	}
}

func TestExpand(t *testing.T) {
	v := [dimInput]float32{1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	feat := expand(v)
	if len(feat) != dimExpanded {
		t.Fatalf("expand len = %d, want %d", len(feat), dimExpanded)
	}
	// First 14: original values
	for d := 0; d < dimInput; d++ {
		if feat[d] != float64(v[d]) {
			t.Errorf("feat[%d] = %v, want %v", d, feat[d], v[d])
		}
	}
	// Check quadratic features
	idx := dimInput
	for a := 0; a < dimInput; a++ {
		for b := a; b < dimInput; b++ {
			expected := float64(v[a]) * float64(v[b])
			if math.Abs(feat[idx]-expected) > 0.0001 {
				t.Errorf("expand[%d] (a=%d,b=%d) = %v, want %v", idx, a, b, feat[idx], expected)
			}
			idx++
		}
	}
}

func TestPredictBeforeTrain(t *testing.T) {
	m := NewModel()
	v := [dimInput]float32{0.5, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	p := m.Predict(v)
	if p < 0 || p > 1 {
		t.Errorf("Predict = %v, want [0,1]", p)
	}
}

func TestWriteToAndLoadFrom(t *testing.T) {
	m := NewModel()
	m.bias = 0.5
	m.weights[0] = 0.1
	m.weights[1] = -0.2

	var buf bytes.Buffer
	n, err := m.WriteTo(&buf)
	if err != nil {
		t.Fatalf("WriteTo: %v", err)
	}
	if n <= 0 {
		t.Fatalf("WriteTo wrote %d bytes", n)
	}

	loaded, err := LoadFrom(&buf)
	if err != nil {
		t.Fatalf("LoadFrom: %v", err)
	}
	if loaded.bias != m.bias {
		t.Errorf("bias = %v, want %v", loaded.bias, m.bias)
	}
	if loaded.dim != m.dim {
		t.Errorf("dim = %d, want %d", loaded.dim, m.dim)
	}
	for i := range m.weights {
		if loaded.weights[i] != m.weights[i] {
			t.Errorf("weights[%d] = %v, want %v", i, loaded.weights[i], m.weights[i])
		}
	}
}

func TestLoadFrom_InvalidMagic(t *testing.T) {
	buf := bytes.NewReader([]byte{0, 0, 0, 0, 1, 0, 0, 0, 14, 0, 0, 0})
	_, err := LoadFrom(buf)
	if err != errInvalidMagic {
		t.Errorf("expected errInvalidMagic, got %v", err)
	}
}

func TestLoadFrom_InvalidVersion(t *testing.T) {
	data := make([]byte, 20)
	data[0] = 0x52 // magic byte 0
	data[1] = 0x47 // magic byte 1
	data[2] = 0x4F // magic byte 2
	data[3] = 0x4C // magic byte 3 -> 0x4C4F4752 = "LOGR"
	data[4] = 99   // version = 99 (invalid)
	data[8] = 14   // dim = 14
	buf := bytes.NewReader(data)
	_, err := LoadFrom(buf)
	if err != errInvalidVersion {
		t.Errorf("expected errInvalidVersion, got %v", err)
	}
}

func TestLoadFrom_EmptyReader(t *testing.T) {
	buf := bytes.NewReader(nil)
	_, err := LoadFrom(buf)
	if err == nil {
		t.Fatal("expected error for empty reader")
	}
}

func TestEvaluate(t *testing.T) {
	m := NewModel()
	vectors := []float32{0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	labels := []bool{true}
	tp, tn, fp, fn := m.Evaluate(vectors, labels, 1)
	if tp+tn+fp+fn != 1 {
		t.Errorf("total evaluations = %d, want 1", tp+tn+fp+fn)
	}
}

func TestLoadPredictor_NotFound(t *testing.T) {
	_, err := LoadPredictor("/nonexistent/path.bin")
	if err == nil {
		t.Fatal("expected error for nonexistent path")
	}
}

func TestLoadPredictor_InvalidFile(t *testing.T) {
	path := t.TempDir() + "/invalid.bin"
	if err := os.WriteFile(path, []byte("invalid"), 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	_, err := LoadPredictor(path)
	if err == nil {
		t.Fatal("expected error for invalid file")
	}
}

func TestLoadPredictor_RoundTrip(t *testing.T) {
	m := NewModel()
	m.bias = 0.3
	m.weights[0] = 0.5

	var buf bytes.Buffer
	if _, err := m.WriteTo(&buf); err != nil {
		t.Fatalf("WriteTo: %v", err)
	}

	path := t.TempDir() + "/predictor.bin"
	if err := os.WriteFile(path, buf.Bytes(), 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	p, err := LoadPredictor(path)
	if err != nil {
		t.Fatalf("LoadPredictor: %v", err)
	}
	if p == nil {
		t.Fatal("LoadPredictor returned nil")
	}

	v := model.Vector14{0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	score := p.Predict(v, 5)
	if score < 0 || score > 1 {
		t.Errorf("Predict = %v, want [0,1]", score)
	}
}

func TestPredictor_NilModel(t *testing.T) {
	p := &Predictor{model: nil}
	v := model.Vector14{0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	score := p.Predict(v, 5)
	if score != 0 {
		t.Errorf("Predict with nil model = %v, want 0", score)
	}
}

func TestPredictor_CountFraudCount(t *testing.T) {
	p := &Predictor{}
	if p.Count() != 0 {
		t.Errorf("Count = %d, want 0", p.Count())
	}
	if p.FraudCount() != 0 {
		t.Errorf("FraudCount = %d, want 0", p.FraudCount())
	}
}

func TestExists(t *testing.T) {
	if Exists("/nonexistent/path.bin") {
		t.Error("Exists should return false for nonexistent path")
	}

	path := t.TempDir() + "/exists.bin"
	if err := os.WriteFile(path, []byte("test"), 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	if !Exists(path) {
		t.Error("Exists should return true for existing path")
	}
}

func TestRNG(t *testing.T) {
	r := randNew(42)
	if r.state != 42 {
		t.Errorf("initial state = %d, want 42", r.state)
	}
	seen := make(map[int]bool)
	for i := 0; i < 100; i++ {
		n := r.Intn(100)
		if n < 0 || n >= 100 {
			t.Errorf("Intn(100) = %d, out of range", n)
		}
		seen[n] = true
	}
	if len(seen) < 50 {
		t.Errorf("only %d unique values from 100 Intn calls, expected >= 50", len(seen))
	}
}

func TestTrain(t *testing.T) {
	m := NewModel()
	// Small dataset: 2 fraud (first 2), 2 legit (next 2)
	vectors := []float32{
		0.9, 0.9, 0.9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0.8, 0.8, 0.8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0.2, 0.2, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	}
	labels := []bool{true, true, false, false}
	m.Train(vectors, labels, 4)

	vFraud := [dimInput]float32{0.9, 0.9, 0.9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	vLegit := [dimInput]float32{0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}

	pFraud := m.Predict(vFraud)
	pLegit := m.Predict(vLegit)

	if pFraud <= pLegit {
		t.Errorf("fraud score %v should be > legit score %v after training", pFraud, pLegit)
	}
}

func TestEvaluate_Results(t *testing.T) {
	m := NewModel()
	m.bias = 0.5
	m.weights[0] = 1.0

	vectors := []float32{
		0.9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	}
	labels := []bool{true, false}

	tp, tn, fp, fn := m.Evaluate(vectors, labels, 2)
	if tp+tn+fp+fn != 2 {
		t.Errorf("total = %d, want 2", tp+tn+fp+fn)
	}
}

func TestTrain_LargeBatch(t *testing.T) {
	m := NewModel()
	n := 300001
	vectors := make([]float32, n*dimInput)
	labels := make([]bool, n)
	for i := 0; i < n; i++ {
		if i < n/2 {
			vectors[i*dimInput] = 0.9
			labels[i] = true
		} else {
			vectors[i*dimInput] = 0.1
			labels[i] = false
		}
	}

	m.Train(vectors, labels, n)

	vFraud := [dimInput]float32{0.9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	vLegit := [dimInput]float32{0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}

	if m.Predict(vFraud) <= m.Predict(vLegit) {
		t.Error("fraud prediction should be higher than legit after training")
	}
}
