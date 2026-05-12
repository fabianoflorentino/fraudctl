package handler

import (
	"time"

	"github.com/valyala/fasthttp"

	"github.com/fabianoflorentino/fraudctl/internal/middleware"
	"github.com/fabianoflorentino/fraudctl/internal/model"
)

type Vectorizer interface {
	Vectorize(req *model.FraudScoreRequest) model.Vector14
	VectorizeJSON(data []byte) (model.Vector14, error)
}

type KNNIndex interface {
	Predict(query model.Vector14, k int) float64
	PredictRaw(query model.Vector14, nprobe int) int
	NProbe() int
}

const (
	knnK              = 5
	knnFraudThreshold = 0.6
	knnNeighbors      = knnK
)

var precomputedResp [knnK + 1][]byte

func init() {
	precomputedResp[0] = []byte(`{"approved":true,"fraud_score":0.0}`)
	precomputedResp[1] = []byte(`{"approved":true,"fraud_score":0.2}`)
	precomputedResp[2] = []byte(`{"approved":true,"fraud_score":0.4}`)
	precomputedResp[3] = []byte(`{"approved":false,"fraud_score":0.6}`)
	precomputedResp[4] = []byte(`{"approved":false,"fraud_score":0.8}`)
	precomputedResp[5] = []byte(`{"approved":false,"fraud_score":1.0}`)
}

type FraudScoreHandler struct {
	vec Vectorizer
	knn KNNIndex
}

func NewFraudScoreHandler(vec Vectorizer, knn KNNIndex) *FraudScoreHandler {
	return &FraudScoreHandler{vec: vec, knn: knn}
}

func (h *FraudScoreHandler) Handle(ctx *fasthttp.RequestCtx) {
	start := time.Now()

	if !ctx.IsPost() {
		ctx.SetStatusCode(fasthttp.StatusMethodNotAllowed)
		return
	}

	vecStart := time.Now()
	vec, err := h.vec.VectorizeJSON(ctx.PostBody())
	vecDur := time.Since(vecStart)

	if err != nil {
		ctx.SetStatusCode(fasthttp.StatusOK)
		ctx.SetContentType("application/json")
		_, _ = ctx.Write(precomputedResp[0])
		middleware.Record(time.Since(start), vecDur, 0, 0, true, true)
		return
	}

	knnStart := time.Now()
	fraudCount := h.knn.PredictRaw(vec, h.knn.NProbe())
	knnDur := time.Since(knnStart)

	resp := precomputedResp[fraudCount]

	ctx.SetStatusCode(fasthttp.StatusOK)
	ctx.SetContentType("application/json")
	_, _ = ctx.Write(resp)

	middleware.Record(time.Since(start), vecDur, knnDur, fraudCount, fraudCount < 3, false)
}
