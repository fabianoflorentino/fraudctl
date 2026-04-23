package handler

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/fabianoflorentino/fraudctl/internal/model"
	"github.com/fabianoflorentino/fraudctl/internal/vectorizer"
)

type mockVec struct{}

func (m *mockVec) Vectorize(req *model.FraudScoreRequest) vectorizer.Vector {
	return vectorizer.Vector{Dimensions: []float64{0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}
}

type mockKNN struct{}

func (m *mockKNN) Predict(vector []float64) (float64, bool) {
	if len(vector) > 0 && vector[0] > 0.8 {
		return 0.8, false
	}
	return 0.2, true
}

type mockRespWriter struct {
	code int
	body string
}

func (m *mockRespWriter) WriteHeader(statusCode int) {
	m.code = statusCode
}

func (m *mockRespWriter) Write(body []byte) (int, error) {
	m.body = string(body)
	return len(body), nil
}

type errorReader struct{}

func (e *errorReader) Read(p []byte) (n int, err error) {
	return 0, io.ErrUnexpectedEOF
}

type mockDataset struct {
	cache map[string]model.FraudScoreResponse
}

func (m *mockDataset) GetCachedAnswer(id string) (model.FraudScoreResponse, bool) {
	resp, ok := m.cache[id]
	return resp, ok
}

func (m *mockDataset) CachedAnswers() int {
	return len(m.cache)
}

func TestFraudScoreHandler_Handle_MethodNotAllowed(t *testing.T) {
	handler := NewFraudScoreHandler(nil, &mockVec{}, &mockKNN{})

	req := httptest.NewRequest(http.MethodGet, "/fraud-score", nil)
	w := httptest.NewRecorder()

	err := handler.Handle(&ResponseWriterAdapter{W: w}, req)

	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if w.Code != http.StatusMethodNotAllowed {
		t.Errorf("expected status 405, got %d", w.Code)
	}
}

func TestFraudScoreHandler_Handle_ReadError(t *testing.T) {
	handler := NewFraudScoreHandler(nil, &mockVec{}, &mockKNN{})

	req := httptest.NewRequest(http.MethodPost, "/fraud-score", &errorReader{})

	err := handler.Handle(&ResponseWriterAdapter{W: httptest.NewRecorder()}, req)

	if err != nil {
		t.Errorf("expected nil error, got %v", err)
	}
}

func TestFraudScoreHandler_Handle_InvalidJSON(t *testing.T) {
	handler := NewFraudScoreHandler(nil, &mockVec{}, &mockKNN{})

	req := httptest.NewRequest(http.MethodPost, "/fraud-score", strings.NewReader("invalid json"))
	w := httptest.NewRecorder()

	err := handler.Handle(&ResponseWriterAdapter{W: w}, req)

	if err != nil {
		t.Errorf("expected nil error, got %v", err)
	}
	if w.Body.String() != `{"approved":true,"fraud_score":0.0}` {
		t.Errorf("unexpected fallback response: %s", w.Body.String())
	}
}

func TestFraudScoreHandler_sendFallback(t *testing.T) {
	handler := &FraudScoreHandler{}

	w := &mockRespWriter{}
	err := handler.sendFallback(w)

	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if w.code != http.StatusOK {
		t.Errorf("expected status 200, got %d", w.code)
	}
	if w.body != `{"approved":true,"fraud_score":0.0}` {
		t.Errorf("unexpected body: %s", w.body)
	}
}

func TestFraudScoreHandler_Handle_Success(t *testing.T) {
	handler := NewFraudScoreHandler(nil, &mockVec{}, &mockKNN{})

	payload := `{
		"id": "tx-123",
		"transaction": {"amount": 100, "installments": 1, "requested_at": "2026-03-11T10:00:00Z"},
		"customer": {"avg_amount": 100, "tx_count_24h": 5, "known_merchants": ["m1"]},
		"merchant": {"id": "m1", "mcc": "5411", "avg_amount": 50},
		"terminal": {"is_online": false, "card_present": true, "km_from_home": 10}
	}`

	req := httptest.NewRequest(http.MethodPost, "/fraud-score", strings.NewReader(payload))
	w := httptest.NewRecorder()

	err := handler.Handle(&ResponseWriterAdapter{W: w}, req)

	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if w.Code != http.StatusOK {
		t.Errorf("expected status 200, got %d", w.Code)
	}

	body := w.Body.String()
	if !strings.Contains(body, `"approved":`) || !strings.Contains(body, `"fraud_score":`) {
		t.Errorf("unexpected body: %s", body)
	}
}

func TestRouter_HandleFunc(t *testing.T) {
	router := NewRouter()

	called := false
	router.HandleFunc("/test", func(w ResponseWriter, r *http.Request) error {
		called = true
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
		return nil
	})

	req := httptest.NewRequest(http.MethodGet, "/test", nil)
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	if !called {
		t.Error("handler was not called")
	}
	if w.Code != http.StatusOK {
		t.Errorf("expected status 200, got %d", w.Code)
	}
	if w.Body.String() != "ok" {
		t.Errorf("expected 'ok', got %s", w.Body.String())
	}
}

func TestFraudScoreHandler_Handle_CacheHit(t *testing.T) {
	cache := &mockDataset{
		cache: map[string]model.FraudScoreResponse{
			"tx-cached": {Approved: false, FraudScore: 0.95},
		},
	}

	handler := NewFraudScoreHandler(cache, &mockVec{}, &mockKNN{})

	payload := `{
		"id": "tx-cached",
		"transaction": {"amount": 100, "installments": 1, "requested_at": "2026-03-11T10:00:00Z"},
		"customer": {"avg_amount": 100, "tx_count_24h": 5, "known_merchants": ["m1"]},
		"merchant": {"id": "m1", "mcc": "5411", "avg_amount": 50},
		"terminal": {"is_online": false, "card_present": true, "km_from_home": 10}
	}`

	req := httptest.NewRequest(http.MethodPost, "/fraud-score", strings.NewReader(payload))
	w := httptest.NewRecorder()

	err := handler.Handle(&ResponseWriterAdapter{W: w}, req)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if w.Code != http.StatusOK {
		t.Errorf("expected status 200, got %d", w.Code)
	}

	body := w.Body.String()
	if !strings.Contains(body, `"approved":false`) {
		t.Errorf("expected approved=false from cache, got %s", body)
	}
	if !strings.Contains(body, `"fraud_score":0.95`) {
		t.Errorf("expected fraud_score=0.95 from cache, got %s", body)
	}
}

func TestFraudScoreHandler_Handle_CacheMiss(t *testing.T) {
	cache := &mockDataset{
		cache: map[string]model.FraudScoreResponse{
			"tx-other": {Approved: true, FraudScore: 0.1},
		},
	}

	handler := NewFraudScoreHandler(cache, &mockVec{}, &mockKNN{})

	payload := `{
		"id": "tx-unknown",
		"transaction": {"amount": 100, "installments": 1, "requested_at": "2026-03-11T10:00:00Z"},
		"customer": {"avg_amount": 100, "tx_count_24h": 5, "known_merchants": ["m1"]},
		"merchant": {"id": "m1", "mcc": "5411", "avg_amount": 50},
		"terminal": {"is_online": false, "card_present": true, "km_from_home": 10}
	}`

	req := httptest.NewRequest(http.MethodPost, "/fraud-score", strings.NewReader(payload))
	w := httptest.NewRecorder()

	err := handler.Handle(&ResponseWriterAdapter{W: w}, req)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if w.Code != http.StatusOK {
		t.Errorf("expected status 200, got %d", w.Code)
	}

	body := w.Body.String()
	if !strings.Contains(body, `"approved":true`) {
		t.Errorf("expected approved=true from KNN fallback, got %s", body)
	}
}
