package handler

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/fabianoflorentino/fraudctl/internal/model"
)

// mockVec Mock implementation of Vectorizer interface for testing.
type mockVec struct{}

// Vectorize Returns a fixed mock vector for testing.
func (m *mockVec) Vectorize(req *model.FraudScoreRequest) model.Vector14 {
	return model.Vector14{0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
}

// mockKNN Mock implementation of KNN interface for testing.
type mockKNN struct{}

// Predict Returns mock prediction results based on vector values.
func (m *mockKNN) Predict(vec model.Vector14, k int) float64 {
	if vec[0] > 0.8 {
		return 0.8
	}
	return 0.2
}

// Count returns the number of vectors in the mock.
func (m *mockKNN) Count() int {
	return 100
}

// errorReader Mock implementation of io.Reader that returns an error.
type errorReader struct{}

// Read Simulates a read error.
func (e *errorReader) Read(p []byte) (n int, err error) {
	return 0, io.ErrUnexpectedEOF
}

// TestFraudScoreHandler_Handle_MethodNotAllowed Verifies that GET requests return 405 Method Not Allowed.
func TestFraudScoreHandler_Handle_MethodNotAllowed(t *testing.T) {
	handler := NewFraudScoreHandler(&mockVec{}, &mockKNN{})

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

// TestFraudScoreHandler_Handle_ReadError Verifies that read errors return a fallback response.
func TestFraudScoreHandler_Handle_ReadError(t *testing.T) {
	handler := NewFraudScoreHandler(&mockVec{}, &mockKNN{})

	req := httptest.NewRequest(http.MethodPost, "/fraud-score", &errorReader{})

	err := handler.Handle(&ResponseWriterAdapter{W: httptest.NewRecorder()}, req)

	if err != nil {
		t.Errorf("expected nil error, got %v", err)
	}
}

// TestFraudScoreHandler_Handle_InvalidJSON Verifies that invalid JSON returns a fallback response.
func TestFraudScoreHandler_Handle_InvalidJSON(t *testing.T) {
	handler := NewFraudScoreHandler(&mockVec{}, &mockKNN{})

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

// TestFraudScoreHandler_sendFallback Verifies that sendFallback returns the correct default response.
func TestFraudScoreHandler_sendFallback(t *testing.T) {
	handler := &FraudScoreHandler{}

	w := httptest.NewRecorder()
	err := handler.sendFallback(&ResponseWriterAdapter{W: w})

	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if w.Code != http.StatusOK {
		t.Errorf("expected status 200, got %d", w.Code)
	}
	if w.Body.String() != `{"approved":true,"fraud_score":0.0}` {
		t.Errorf("unexpected body: %s", w.Body.String())
	}
}

// TestFraudScoreHandler_Handle_Success Verifies successful fraud score calculation for valid requests.
func TestFraudScoreHandler_Handle_Success(t *testing.T) {
	handler := NewFraudScoreHandler(&mockVec{}, &mockKNN{})

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

// TestFraudScoreHandler_Handle_ApprovedFalse Verifies that high fraud scores result in approved=false.
func TestFraudScoreHandler_Handle_ApprovedFalse(t *testing.T) {
	highVec := &mockVecHigh{}
	handler := NewFraudScoreHandler(highVec, &mockKNN{})

	payload := `{
		"id": "tx-456",
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
		t.Errorf("expected approved=false, got %s", body)
	}
}

// mockVecHigh Returns a vector that triggers high fraud score.
type mockVecHigh struct{}

func (m *mockVecHigh) Vectorize(req *model.FraudScoreRequest) model.Vector14 {
	return model.Vector14{0.9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
}

// TestRouter_HandleFunc Verifies that router correctly registers and executes handlers.
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
