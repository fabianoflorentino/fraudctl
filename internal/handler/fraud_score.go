package handler

import (
	"github.com/valyala/fasthttp"

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

var (
	approvedBody    = []byte(`{"approved":true,"fraud_score":0.0}`)
	disapprovedBody = []byte(`{"approved":false,"fraud_score":1.0}`)
)

type FraudScoreHandler struct {
	vec Vectorizer
	knn KNNIndex
}

func NewFraudScoreHandler(vec Vectorizer, knn KNNIndex) *FraudScoreHandler {
	return &FraudScoreHandler{vec: vec, knn: knn}
}

func (h *FraudScoreHandler) Handle(ctx *fasthttp.RequestCtx) {
	if !ctx.IsPost() {
		ctx.SetStatusCode(fasthttp.StatusMethodNotAllowed)
		return
	}

	vec, err := h.vec.VectorizeJSON(ctx.PostBody())
	if err != nil {
		ctx.SetStatusCode(fasthttp.StatusOK)
		ctx.SetContentType("application/json")
		_, _ = ctx.Write(approvedBody)
		return
	}

	fraudCount := h.knn.PredictRaw(vec, h.knn.NProbe())
	score := float64(fraudCount) / float64(knnNeighbors)

	resp := approvedBody
	if score >= knnFraudThreshold {
		resp = disapprovedBody
	}

	ctx.SetStatusCode(fasthttp.StatusOK)
	ctx.SetContentType("application/json")
	_, _ = ctx.Write(resp)
}
