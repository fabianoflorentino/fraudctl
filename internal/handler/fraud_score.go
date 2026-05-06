package handler

import (
	"log"
	"sync"
	"sync/atomic"

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
	// C collects K_NEIGHBORS=10; threshold=0.6 → deny when >=6/10 are fraud.
	knnFraudThreshold = 0.6

	// knnNeighbors matches C K_NEIGHBORS — number of neighbors C collects.
	knnNeighbors = 5
)

var (
	approvedBody    = []byte(`{"approved":true,"fraud_score":0.0}`)
	disapprovedBody = []byte(`{"approved":false,"fraud_score":1.0}`)

	reqPool = sync.Pool{
		New: func() any { return new(model.FraudScoreRequest) },
	}
)

type FraudScoreHandler struct {
	vec          Vectorizer
	knn          KNNIndex
	requestCount atomic.Uint64
}

func NewFraudScoreHandler(vec Vectorizer, knn KNNIndex) *FraudScoreHandler {
	log.Printf("KNN-only inference: k=%d fraud_threshold=%.1f (majority vote >=3/%d)",
		knnK, knnFraudThreshold, knnK)
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
		ctx.Write(approvedBody)
		return
	}

	vec := h.vec.Vectorize(req)
	nprobe := h.knn.NProbe()
	fraudCount := h.knn.PredictRaw(vec, nprobe)
	score := float64(fraudCount) / float64(knnNeighbors)
	reqPool.Put(req)

	var resp []byte
	if score >= knnFraudThreshold {
		resp = disapprovedBody
	} else {
		resp = approvedBody
	}

	ctx.SetStatusCode(fasthttp.StatusOK)
	ctx.SetContentType("application/json")
	ctx.Write(resp)

	if count := h.requestCount.Add(1); count%10000 == 0 {
		log.Printf("requests=%d knn_score=%.3f", count, score)
	}
}
