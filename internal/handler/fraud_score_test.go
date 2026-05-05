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

// mockKNN returns a fixed score regardless of the query vector.
type mockKNN struct{ score float64 }

func (m *mockKNN) Predict(_ model.Vector14, _ int) float64 { return m.score }

func newCtx(method, body string) *fasthttp.RequestCtx {
	ctx := &fasthttp.RequestCtx{}
	ctx.Request.SetRequestURI("/fraud-score")
	ctx.Request.Header.SetMethod(method)
	if body != "" {
		ctx.Request.SetBodyString(body)
	}
	return ctx
}

const validPayload = `{
	"id": "tx-123",
	"transaction": {"amount": 100, "installments": 1, "requested_at": "2026-03-11T10:00:00Z"},
	"customer": {"avg_amount": 100, "tx_count_24h": 5, "known_merchants": ["m1"]},
	"merchant": {"id": "m1", "mcc": "5411", "avg_amount": 50},
	"terminal": {"is_online": false, "card_present": true, "km_from_home": 10}
}`

func TestFraudScoreHandler_MethodNotAllowed(t *testing.T) {
	h := NewFraudScoreHandler(&mockVec{}, &mockKNN{0.0})
	ctx := newCtx("GET", "")
	h.Handle(ctx)
	if ctx.Response.StatusCode() != fasthttp.StatusMethodNotAllowed {
		t.Errorf("expected 405, got %d", ctx.Response.StatusCode())
	}
}

func TestFraudScoreHandler_InvalidJSON(t *testing.T) {
	h := NewFraudScoreHandler(&mockVec{}, &mockKNN{0.0})
	ctx := newCtx("POST", "invalid json")
	h.Handle(ctx)
	if ctx.Response.StatusCode() != fasthttp.StatusOK {
		t.Errorf("expected 200, got %d", ctx.Response.StatusCode())
	}
	body := string(ctx.Response.Body())
	if !strings.Contains(body, `"approved":true`) {
		t.Errorf("expected approved=true fallback, got %s", body)
	}
}

func TestFraudScoreHandler_LegitScore(t *testing.T) {
	// KNN returns 0.4 → below threshold → approved
	h := NewFraudScoreHandler(&mockVec{}, &mockKNN{0.4})
	ctx := newCtx("POST", validPayload)
	h.Handle(ctx)
	if ctx.Response.StatusCode() != fasthttp.StatusOK {
		t.Errorf("expected 200, got %d", ctx.Response.StatusCode())
	}
	body := string(ctx.Response.Body())
	if !strings.Contains(body, `"approved":true`) {
		t.Errorf("expected approved=true, got %s", body)
	}
}

func TestFraudScoreHandler_FraudScore(t *testing.T) {
	// KNN returns 0.6 → at threshold → denied (majority vote >=3/5)
	h := NewFraudScoreHandler(&mockVec{}, &mockKNN{0.6})
	ctx := newCtx("POST", validPayload)
	h.Handle(ctx)
	if ctx.Response.StatusCode() != fasthttp.StatusOK {
		t.Errorf("expected 200, got %d", ctx.Response.StatusCode())
	}
	body := string(ctx.Response.Body())
	if !strings.Contains(body, `"approved":false`) {
		t.Errorf("expected approved=false, got %s", body)
	}
}
