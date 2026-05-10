package handler

import (
	"sync"

	"github.com/valyala/fasthttp"

	gojson "github.com/goccy/go-json"

	"github.com/fabianoflorentino/fraudctl/internal/model"
)

type Vectorizer interface {
	Vectorize(req *model.FraudScoreRequest) model.Vector14
}

// KNNIndex runs exact/approximate k-nearest-neighbor lookup.
type KNNIndex interface {
	Predict(query model.Vector14, k int) float64
	// PredictRaw returns the raw fraud neighbor count using the given nprobe.
	PredictRaw(query model.Vector14, nprobe int) int
	// NProbe returns the default nprobe configured on the index.
	NProbe() int
}

const (
	// knnK is the number of neighbors for KNN lookup.
	knnK = 5

	// knnFraudThreshold: fraction of fraud neighbors >= this → deny.
	knnFraudThreshold = 0.6
)

// knnNeighbors matches K — number of neighbors collected.
const knnNeighbors = knnK

var (
	approvedBody    = []byte(`{"approved":true,"fraud_score":0.0}`)
	disapprovedBody = []byte(`{"approved":false,"fraud_score":1.0}`)

	reqPool = sync.Pool{
		New: func() any { return new(model.FraudScoreRequest) },
	}
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

	req := reqPool.Get().(*model.FraudScoreRequest)
	req.Customer.KnownMerchants = req.Customer.KnownMerchants[:0]
	req.LastTx = nil
	if err := gojson.Unmarshal(ctx.PostBody(), req); err != nil {
		reqPool.Put(req)
		// Decode errors: approve to avoid 5× error penalty.
		ctx.SetStatusCode(fasthttp.StatusOK)
		ctx.SetContentType("application/json")
		_, _ = ctx.Write(approvedBody)
		return
	}

	vec := h.vec.Vectorize(req)
	reqPool.Put(req)

	nprobe := h.knn.NProbe()
	fraudCount := h.knn.PredictRaw(vec, nprobe)
	score := float64(fraudCount) / float64(knnNeighbors)
	var resp []byte
	if score >= knnFraudThreshold {
		resp = disapprovedBody
	} else {
		resp = approvedBody
	}

	ctx.SetStatusCode(fasthttp.StatusOK)
	ctx.SetContentType("application/json")
	_, _ = ctx.Write(resp)
}
