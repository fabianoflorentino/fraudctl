package handler

import (
	"math"
	"testing"

	"github.com/fabianoflorentino/fraudctl/internal/model"
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
	count := h.HandleFraudScore([]byte(validPayload))
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
	count := h.HandleFraudScore([]byte(validPayload))
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
	count := h.HandleFraudScore([]byte("not json"))
	if count != 0 {
		t.Errorf("expected fallback count=0, got %d", count)
	}
}

func TestHandleFraudScore_EmptyBody(t *testing.T) {
	h := NewFraudScoreHandler(&mockVec{}, &mockKNN{0.0})
	count := h.HandleFraudScore([]byte{})
	if count != 0 {
		t.Errorf("expected fallback count=0, got %d", count)
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
	count := h.HandleFraudScore([]byte(validPayload))
	if count < 0 || count > 5 {
		t.Errorf("count out of range: %d", count)
	}
}

func TestHandleFraudScore_BoundaryScore_High(t *testing.T) {
	h := NewFraudScoreHandler(&mockVec{}, &mockKNN{1.0})
	count := h.HandleFraudScore([]byte(validPayload))
	if count < 0 || count > 5 {
		t.Errorf("count out of range: %d", count)
	}
}

func TestHandleFraudScore_MidScore(t *testing.T) {
	h := NewFraudScoreHandler(&mockVec{}, &mockKNN{0.5})
	count := h.HandleFraudScore([]byte(validPayload))
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
	count := h.HandleFraudScore([]byte(validPayload))
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

func BenchmarkHandleFraudScore(b *testing.B) {
	h := NewFraudScoreHandler(&mockVec{}, &mockKNN{0.5})
	payload := []byte(validPayload)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = h.HandleFraudScore(payload)
	}
}

func BenchmarkResponseForCount(b *testing.B) {
	h := NewFraudScoreHandler(&mockVec{}, &mockKNN{0.5})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = h.ResponseForCount(3)
	}
}
