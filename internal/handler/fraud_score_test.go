package handler

import (
	"strings"
	"testing"

	"github.com/valyala/fasthttp"

	"github.com/fabianoflorentino/fraudctl/internal/model"
)

type mockVec struct{}

func (m *mockVec) Vectorize(req *model.FraudScoreRequest) model.Vector14 {
	return model.Vector14{0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
}

type mockVecHigh struct{}

func (m *mockVecHigh) Vectorize(req *model.FraudScoreRequest) model.Vector14 {
	return model.Vector14{0.9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
}

func TestFraudScoreHandler_MethodNotAllowed(t *testing.T) {
	ctx := &fasthttp.RequestCtx{}
	ctx.Request.SetRequestURI("/fraud-score")
	ctx.Request.Header.SetMethod("GET")
	HandleFraudScore(ctx, &mockVec{})

	if ctx.Response.StatusCode() != fasthttp.StatusMethodNotAllowed {
		t.Errorf("expected 405, got %d", ctx.Response.StatusCode())
	}
}

func TestFraudScoreHandler_InvalidJSON(t *testing.T) {
	ctx := &fasthttp.RequestCtx{}
	ctx.Request.SetRequestURI("/fraud-score")
	ctx.Request.Header.SetMethod("POST")
	ctx.Request.SetBody([]byte("invalid json"))
	HandleFraudScore(ctx, &mockVec{})

	if ctx.Response.StatusCode() != fasthttp.StatusOK {
		t.Errorf("expected 200, got %d", ctx.Response.StatusCode())
	}
	body := string(ctx.Response.Body())
	if body != `{"approved":true,"fraud_score":0}` {
		t.Errorf("unexpected fallback: %s", body)
	}
}

func TestFraudScoreHandler_Success(t *testing.T) {
	payload := `{
		"id": "tx-123",
		"transaction": {"amount": 100, "installments": 1, "requested_at": "2026-03-11T10:00:00Z"},
		"customer": {"avg_amount": 100, "tx_count_24h": 5, "known_merchants": ["m1"]},
		"merchant": {"id": "m1", "mcc": "5411", "avg_amount": 50},
		"terminal": {"is_online": false, "card_present": true, "km_from_home": 10}
	}`

	ctx := &fasthttp.RequestCtx{}
	ctx.Request.SetRequestURI("/fraud-score")
	ctx.Request.Header.SetMethod("POST")
	ctx.Request.SetBodyString(payload)
	HandleFraudScore(ctx, &mockVec{})

	if ctx.Response.StatusCode() != fasthttp.StatusOK {
		t.Errorf("expected 200, got %d", ctx.Response.StatusCode())
	}
	body := string(ctx.Response.Body())
	if !strings.Contains(body, `"approved":`) || !strings.Contains(body, `"fraud_score":`) {
		t.Errorf("unexpected body: %s", body)
	}
}

func TestFraudScoreHandler_HighFraud(t *testing.T) {
	payload := `{
		"id": "tx-456",
		"transaction": {"amount": 100, "installments": 1, "requested_at": "2026-03-11T10:00:00Z"},
		"customer": {"avg_amount": 100, "tx_count_24h": 5, "known_merchants": ["m1"]},
		"merchant": {"id": "m1", "mcc": "5411", "avg_amount": 50},
		"terminal": {"is_online": false, "card_present": true, "km_from_home": 10}
	}`

	ctx := &fasthttp.RequestCtx{}
	ctx.Request.SetRequestURI("/fraud-score")
	ctx.Request.Header.SetMethod("POST")
	ctx.Request.SetBodyString(payload)
	HandleFraudScore(ctx, &mockVecHigh{})

	if ctx.Response.StatusCode() != fasthttp.StatusOK {
		t.Errorf("expected 200, got %d", ctx.Response.StatusCode())
	}
	body := string(ctx.Response.Body())
	if !strings.Contains(body, `"approved":false`) {
		t.Errorf("expected approved=false, got %s", body)
	}
}
