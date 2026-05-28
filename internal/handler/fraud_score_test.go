package handler

import (
	"math"
	"os"
	"testing"

	"github.com/fabianoflorentino/fraudctl/internal/gbdt"
	"github.com/fabianoflorentino/fraudctl/internal/knn"
	"github.com/fabianoflorentino/fraudctl/internal/model"
	"github.com/fabianoflorentino/fraudctl/internal/vectorizer"
)

type mockVec struct{}

func (m *mockVec) Vectorize(req *model.FraudScoreRequest) model.Vector14 {
	return model.Vector14{0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
}

func (m *mockVec) VectorizeJSON(data []byte) (model.Vector14, error) {
	for _, b := range data {
		if b == '{' {
			return model.Vector14{0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, nil
		}
		if b != ' ' && b != '\t' && b != '\n' && b != '\r' {
			break
		}
	}
	return model.Vector14{}, errInvalidJSON
}

var errInvalidJSON = &invalidJSONError{}

type invalidJSONError struct{}

func (e *invalidJSONError) Error() string { return "invalid json" }

type mockKNN struct{ score float64 }

func (m *mockKNN) Predict(_ model.Vector14, _ int) float64 { return m.score }
func (m *mockKNN) PredictRaw(_ model.Vector14, _ int) int {
	return int(math.Round(m.score * float64(knnK)))
}
func (m *mockKNN) NProbe() int { return 8 }

const validPayload = `{
	"id": "tx-123",
	"transaction": {"amount": 100, "installments": 1, "requested_at": "2026-03-11T10:00:00Z"},
	"customer": {"avg_amount": 100, "tx_count_24h": 5, "known_merchants": ["m1"]},
	"merchant": {"id": "m1", "mcc": "5411", "avg_amount": 50},
	"terminal": {"is_online": false, "card_present": true, "km_from_home": 10}
}`

func TestHandleFraudScore_ValidJSON_Approved(t *testing.T) {
	h := NewFraudScoreHandler(&mockVec{}, &mockKNN{0.4})
	count, _ := h.HandleFraudScore([]byte(validPayload))
	if count >= 3 {
		t.Errorf("expected approved (count < 3), got count=%d", count)
	}
	resp := h.ResponseForCount(count)
	if len(resp) == 0 {
		t.Error("expected non-empty response")
	}
}

func TestHandleFraudScore_ValidJSON_Denied(t *testing.T) {
	h := NewFraudScoreHandler(&mockVec{}, &mockKNN{0.6})
	count, _ := h.HandleFraudScore([]byte(validPayload))
	if count < 3 {
		t.Errorf("expected denied (count >= 3), got count=%d", count)
	}
	resp := h.ResponseForCount(count)
	if len(resp) == 0 {
		t.Error("expected non-empty response")
	}
}

func TestHandleFraudScore_InvalidJSON(t *testing.T) {
	h := NewFraudScoreHandler(&mockVec{}, &mockKNN{0.0})
	count, parseErr := h.HandleFraudScore([]byte("not json"))
	if count != 0 {
		t.Errorf("expected fallback count=0, got %d", count)
	}
	if !parseErr {
		t.Error("expected parseErr=true for invalid JSON")
	}
}

func TestHandleFraudScore_EmptyBody(t *testing.T) {
	h := NewFraudScoreHandler(&mockVec{}, &mockKNN{0.0})
	count, parseErr := h.HandleFraudScore([]byte{})
	if count != 0 {
		t.Errorf("expected fallback count=0, got %d", count)
	}
	if !parseErr {
		t.Error("expected parseErr=true for empty body")
	}
}

func TestResponseForCount(t *testing.T) {
	h := NewFraudScoreHandler(&mockVec{}, &mockKNN{0.0})
	for count := 0; count <= 5; count++ {
		resp := h.ResponseForCount(count)
		if string(resp) != string(precomputedResp[count]) {
			t.Errorf("ResponseForCount(%d) = %s; want %s", count, resp, precomputedResp[count])
		}
	}
}

func TestResponseForCount_OutOfRange(t *testing.T) {
	h := NewFraudScoreHandler(&mockVec{}, &mockKNN{0.0})
	tests := []struct {
		count int
		want  int
	}{
		{-1, 0},
		{99, 0},
	}
	for _, tt := range tests {
		resp := h.ResponseForCount(tt.count)
		if string(resp) != string(precomputedResp[tt.want]) {
			t.Errorf("ResponseForCount(%d) = %s; want %s", tt.count, resp, precomputedResp[tt.want])
		}
	}
}

func TestNewFraudScoreHandler(t *testing.T) {
	vec := &mockVec{}
	knn := &mockKNN{0.5}
	h := NewFraudScoreHandler(vec, knn)
	if h == nil {
		t.Fatal("NewFraudScoreHandler returned nil")
	}
}

func TestFraudScoreHandler_Fields(t *testing.T) {
	vec := &mockVec{}
	knn := &mockKNN{0.5}
	h := NewFraudScoreHandler(vec, knn)
	if h.vec != vec {
		t.Error("vectorizer not set")
	}
	if h.knn != knn {
		t.Error("knn not set")
	}
}

func TestHandleFraudScore_BoundaryScore_Low(t *testing.T) {
	h := NewFraudScoreHandler(&mockVec{}, &mockKNN{0.0})
	count, _ := h.HandleFraudScore([]byte(validPayload))
	if count < 0 || count > 5 {
		t.Errorf("count out of range: %d", count)
	}
}

func TestHandleFraudScore_BoundaryScore_High(t *testing.T) {
	h := NewFraudScoreHandler(&mockVec{}, &mockKNN{1.0})
	count, _ := h.HandleFraudScore([]byte(validPayload))
	if count < 0 || count > 5 {
		t.Errorf("count out of range: %d", count)
	}
}

func TestHandleFraudScore_MidScore(t *testing.T) {
	h := NewFraudScoreHandler(&mockVec{}, &mockKNN{0.5})
	count, _ := h.HandleFraudScore([]byte(validPayload))
	if count < 0 || count > 5 {
		t.Errorf("count out of range: %d", count)
	}
}

func TestResponseForCount_AllValues(t *testing.T) {
	h := NewFraudScoreHandler(&mockVec{}, &mockKNN{0.0})
	for count := 0; count <= 5; count++ {
		resp := h.ResponseForCount(count)
		if len(resp) == 0 {
			t.Errorf("ResponseForCount(%d) returned empty response", count)
		}
	}
}

func TestHandleFraudScore_SetsCount(t *testing.T) {
	h := NewFraudScoreHandler(&mockVec{}, &mockKNN{0.7})
	count, _ := h.HandleFraudScore([]byte(validPayload))
	// Score 0.7 * 5 (knnK) = 3.5, rounds to 4
	if count == -1 {
		t.Error("count not set")
	}
}

func TestResponseForCount_Consistency(t *testing.T) {
	h := NewFraudScoreHandler(&mockVec{}, &mockKNN{0.0})

	// Same count should always return same response
	resp1 := h.ResponseForCount(3)
	resp2 := h.ResponseForCount(3)
	if string(resp1) != string(resp2) {
		t.Error("inconsistent responses for same count")
	}
}

func TestPrecomputedResponses_Length(t *testing.T) {
	if len(precomputedResp) < 6 {
		t.Errorf("precomputedResp length = %d, want >= 6", len(precomputedResp))
	}
}

func TestKNNThreshold(t *testing.T) {
	if knnK <= 0 {
		t.Errorf("knnK = %d, want > 0", knnK)
	}
}
func TestMockVectorizer(t *testing.T) {
	vec := &mockVec{}
	req := &model.FraudScoreRequest{}
	v := vec.Vectorize(req)
	if v[0] != 0.1 {
		t.Errorf("Vectorize() first element = %f, want 0.1", v[0])
	}

	// Verify it's a 14D vector
	if len(v) != 14 {
		t.Errorf("vector length = %d, want 14", len(v))
	}
}

func TestMockVectorizer_VectorizeJSON_Valid(t *testing.T) {
	vec := &mockVec{}
	_, err := vec.VectorizeJSON([]byte(`{}`))
	if err != nil {
		t.Errorf("VectorizeJSON valid json returned error: %v", err)
	}
}

func TestMockVectorizer_VectorizeJSON_Invalid(t *testing.T) {
	vec := &mockVec{}
	_, err := vec.VectorizeJSON([]byte(`not json`))
	if err == nil {
		t.Error("VectorizeJSON invalid json should return error")
	}
}

func TestMockKNN(t *testing.T) {
	knn := &mockKNN{0.42}
	score := knn.Predict(model.Vector14{}, 5)
	if score != 0.42 {
		t.Errorf("Predict returned %f, want 0.42", score)
	}
}

func TestMockKNN_PredictRaw(t *testing.T) {
	knn := &mockKNN{0.5}
	raw := knn.PredictRaw(model.Vector14{}, 5)
	// 0.5 * knnK rounded should be calculated
	if raw <= 0 {
		t.Errorf("PredictRaw returned %d, want > 0", raw)
	}
}

func TestMockKNN_NProbe(t *testing.T) {
	knn := &mockKNN{0.5}
	nprobe := knn.NProbe()
	if nprobe != 8 {
		t.Errorf("NProbe returned %d, want 8", nprobe)
	}
}

func TestInvalidJSONError(t *testing.T) {
	err := &invalidJSONError{}
	if err.Error() != "invalid json" {
		t.Errorf("Error() = %q, want 'invalid json'", err.Error())
	}
}

// gbdtHelper creates a temporary GBDT model file with a single leaf node
// returning the given score in regression mode (sigmoid=false).
func gbdtHelper(t *testing.T, score float32) string {
	t.Helper()
	path := t.TempDir() + "/gbdt_test.bin"
	data := make([]byte, 0, 64)
	data = append(data, []byte("GBDT")...)
	data = append(data, byte(2), 0, 0, 0)  // version=2
	data = append(data, byte(1), 0, 0, 0)  // numTrees=1
	data = append(data, byte(14), 0, 0, 0) // numFeatures=14
	data = append(data, 0, 0, 0, 0)        // initPred=0
	data = append(data, 0, 0, 0, 0)        // sigmoid=0, pad
	data = append(data, byte(1), 0, 0, 0)  // 1 node
	data = append(data, 0)                 // feat=0
	data = append(data, 0)                 // pad
	data = append(data, 0, 0)              // left=0
	data = append(data, 0, 0)              // right=0
	data = append(data, 1)                 // leaf=1
	data = append(data, 0)                 // pad
	v := math.Float32bits(score)
	data = append(data, byte(v), byte(v>>8), byte(v>>16), byte(v>>24))
	if err := os.WriteFile(path, data, 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	return path
}

func TestHandleFraudScore_GBDT_Prefilter_Low(t *testing.T) {
	// GBDT predicts score <= confLow (0.08) — should return 0 without calling KNN
	path := gbdtHelper(t, 0.0)
	g, err := gbdt.Load(path)
	if err != nil {
		t.Fatalf("LoadGBDT: %v", err)
	}
	h := NewFraudScoreHandler(&mockVec{}, &mockKNN{1.0}) // KNN would return 5 if called
	h.SetGBDT(g)
	count, parseErr := h.HandleFraudScore([]byte(validPayload))
	if count != 0 {
		t.Errorf("expected count=0 (prefilter low), got %d", count)
	}
	if parseErr {
		t.Error("expected parseErr=false")
	}
}

func TestHandleFraudScore_GBDT_Prefilter_High(t *testing.T) {
	// GBDT predicts score >= confHigh (0.92) — should return 5 without calling KNN
	path := gbdtHelper(t, 1.0)
	g, err := gbdt.Load(path)
	if err != nil {
		t.Fatalf("LoadGBDT: %v", err)
	}
	h := NewFraudScoreHandler(&mockVec{}, &mockKNN{0.0}) // KNN would return 0 if called
	h.SetGBDT(g)
	count, parseErr := h.HandleFraudScore([]byte(validPayload))
	if count != 5 {
		t.Errorf("expected count=5 (prefilter high), got %d", count)
	}
	if parseErr {
		t.Error("expected parseErr=false")
	}
}

func TestHandleFraudScore_GBDT_Prefilter_Mid(t *testing.T) {
	// GBDT predicts score in (confLow, confHigh) — KNN should be called
	path := gbdtHelper(t, 0.5)
	g, err := gbdt.Load(path)
	if err != nil {
		t.Fatalf("LoadGBDT: %v", err)
	}
	h := NewFraudScoreHandler(&mockVec{}, &mockKNN{0.6}) // KNN returns score=0.6 → count=3
	h.SetGBDT(g)
	count, parseErr := h.HandleFraudScore([]byte(validPayload))
	if count != 3 {
		t.Errorf("expected count=3 (KNN called), got %d", count)
	}
	if parseErr {
		t.Error("expected parseErr=false")
	}
}

func TestSetGBDT_Nil(t *testing.T) {
	h := NewFraudScoreHandler(&mockVec{}, &mockKNN{0.5})
	h.SetGBDT(nil)
	// Should still work without GBDT
	count, parseErr := h.HandleFraudScore([]byte(validPayload))
	if parseErr {
		t.Error("expected parseErr=false")
	}
	if count == -1 {
		t.Error("count not set")
	}
}

func TestHandleFraudScore_GBDT_Prefilter_InvalidJSON(t *testing.T) {
	// Even with GBDT prefilter, invalid JSON should return parse error
	path := gbdtHelper(t, 0.0)
	g, err := gbdt.Load(path)
	if err != nil {
		t.Fatalf("LoadGBDT: %v", err)
	}
	h := NewFraudScoreHandler(&mockVec{}, &mockKNN{0.0})
	h.SetGBDT(g)
	count, parseErr := h.HandleFraudScore([]byte("not json"))
	if count != 0 {
		t.Errorf("expected fallback count=0, got %d", count)
	}
	if !parseErr {
		t.Error("expected parseErr=true for invalid JSON")
	}
}

func BenchmarkHandleFraudScore(b *testing.B) {
	h := NewFraudScoreHandler(&mockVec{}, &mockKNN{0.5})
	payload := []byte(validPayload)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = h.HandleFraudScore(payload)
	}
}

func BenchmarkResponseForCount(b *testing.B) {
	h := NewFraudScoreHandler(&mockVec{}, &mockKNN{0.5})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = h.ResponseForCount(3)
	}
}

func BenchmarkHandleFraudScore_FullPipeline(b *testing.B) {
	gridPath := "../../resources/grid.bin"
	if _, err := os.Stat(gridPath); os.IsNotExist(err) {
		b.Skipf("grid.bin not found at %s (build with: make grid)", gridPath)
	}

	idx, err := knn.LoadGrid(gridPath)
	if err != nil {
		b.Fatalf("LoadGrid: %v", err)
	}

	norm := model.NormalizationConstants{
		MaxAmount:            100000,
		MaxInstallments:      12,
		AmountVsAvgRatio:     50,
		MaxMinutes:           10080,
		MaxKm:                10000,
		MaxTxCount24h:        100,
		MaxMerchantAvgAmount: 100000,
	}
	vec := vectorizer.New(norm, model.MCCRisk{})

	h := NewFraudScoreHandler(vec, idx)
	payload := []byte(validPayload)

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = h.HandleFraudScore(payload)
	}
}

func BenchmarkHandleFraudScore_FullPipeline_Parallel(b *testing.B) {
	gridPath := "../../resources/grid.bin"
	if _, err := os.Stat(gridPath); os.IsNotExist(err) {
		b.Skipf("grid.bin not found at %s (build with: make grid)", gridPath)
	}

	idx, err := knn.LoadGrid(gridPath)
	if err != nil {
		b.Fatalf("LoadGrid: %v", err)
	}

	norm := model.NormalizationConstants{
		MaxAmount:            100000,
		MaxInstallments:      12,
		AmountVsAvgRatio:     50,
		MaxMinutes:           10080,
		MaxKm:                10000,
		MaxTxCount24h:        100,
		MaxMerchantAvgAmount: 100000,
	}
	vec := vectorizer.New(norm, model.MCCRisk{})

	h := NewFraudScoreHandler(vec, idx)
	payload := []byte(validPayload)

	b.ResetTimer()
	b.ReportAllocs()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			_, _ = h.HandleFraudScore(payload)
		}
	})
}
