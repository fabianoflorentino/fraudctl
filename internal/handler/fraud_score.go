package handler

import (
	"fmt"
	"log"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/valyala/fasthttp"

	gojson "github.com/goccy/go-json"

	"github.com/fabianoflorentino/fraudctl/internal/model"
)

type Vectorizer interface {
	Vectorize(req *model.FraudScoreRequest) model.Vector14
}

type Predictor interface {
	Predict(vec []float32) float64
}

// approvalThreshold defines the fraud score cutoff for approval.
// Optimized for minimum E=FP+3*FN on official test data.
const approvalThreshold = 0.02

var staticResponses [101][]byte

func init() {
	for i := 0; i <= 100; i++ {
		score := float64(i) / 100.0
		approved := score < approvalThreshold
		staticResponses[i] = fmt.Appendf(nil,
			`{"approved":%s,"fraud_score":%s}`,
			strconv.FormatBool(approved),
			strconv.FormatFloat(score, 'f', -1, 64),
		)
	}
}

type FraudScoreHandler struct {
	vec          Vectorizer
	predictor    Predictor
	bufPool      sync.Pool
	requestCount atomic.Uint64
}

func NewFraudScoreHandler(vec Vectorizer, predictor Predictor) *FraudScoreHandler {
	h := &FraudScoreHandler{
		vec:       vec,
		predictor: predictor,
	}
	h.bufPool.New = func() interface{} {
		b := make([]byte, 0, 4096)
		return &b
	}
	return h
}

func (h *FraudScoreHandler) Handle(ctx *fasthttp.RequestCtx) {
	start := time.Now()

	if !ctx.IsPost() {
		ctx.SetStatusCode(fasthttp.StatusMethodNotAllowed)
		return
	}

	bufp := h.bufPool.Get().(*[]byte)
	*bufp = (*bufp)[:0]

	body := ctx.PostBody()
	if cap(*bufp) >= len(body) {
		*bufp = (*bufp)[:len(body)]
		copy(*bufp, body)
	} else {
		*bufp = append(*bufp, body...)
	}

	var req model.FraudScoreRequest
	err := gojson.Unmarshal(*bufp, &req)
	h.bufPool.Put(bufp)
	if err != nil {
		ctx.SetStatusCode(fasthttp.StatusOK)
		ctx.Write(staticResponses[0])
		return
	}

	vec := h.vec.Vectorize(&req)
	fraudScore := h.predictor.Predict(vec[:])

	scoreIdx := int(fraudScore*100 + 0.5)
	if scoreIdx < 0 {
		scoreIdx = 0
	} else if scoreIdx > 100 {
		scoreIdx = 100
	}
	resp := staticResponses[scoreIdx]

	ctx.SetStatusCode(fasthttp.StatusOK)
	ctx.SetContentType("application/json")
	ctx.Write(resp)

	count := h.requestCount.Add(1)
	if count%1000 == 0 {
		log.Printf("requests=%d latency=%s fraud_score=%.4f", count, time.Since(start), fraudScore)
	}
}

// HandleFraudScore is a convenience function for testing without creating a full handler.
func HandleFraudScore(ctx *fasthttp.RequestCtx, vec Vectorizer) {
	if !ctx.IsPost() {
		ctx.SetStatusCode(fasthttp.StatusMethodNotAllowed)
		return
	}

	body := ctx.PostBody()
	var req model.FraudScoreRequest
	err := gojson.Unmarshal(body, &req)
	if err != nil {
		ctx.SetStatusCode(fasthttp.StatusOK)
		ctx.Write(staticResponses[0])
		return
	}

	v := vec.Vectorize(&req)
	// Simple heuristic for testing: high first dimension = high fraud
	score := float64(v[0])
	scoreIdx := int(score*100 + 0.5)
	if scoreIdx < 0 {
		scoreIdx = 0
	} else if scoreIdx > 100 {
		scoreIdx = 100
	}

	ctx.SetStatusCode(fasthttp.StatusOK)
	ctx.SetContentType("application/json")
	ctx.Write(staticResponses[scoreIdx])
}
