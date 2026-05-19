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
