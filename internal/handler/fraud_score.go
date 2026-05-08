package handler

import (
	"log"
	"sync/atomic"

	"github.com/valyala/fasthttp"

	"github.com/fabianoflorentino/fraudctl/internal/model"
)

type Vectorizer interface {
	Vectorize(req *model.FraudScoreRequest) model.Vector14
	VectorizeRaw(req *model.RawRequest) model.Vector14
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

	// knnNeighbors matches C K_NEIGHBORS — number of neighbors C collects.
	knnNeighbors = 5
)

var (
	approvedBody    = []byte(`{"approved":true,"fraud_score":0.0}`)
	disapprovedBody = []byte(`{"approved":false,"fraud_score":1.0}`)
	contentTypeJSON = []byte("application/json")

	// responses indexed by fraudCount (0-5).
	responses = [6][]byte{
		approvedBody,    // 0 → approved
		approvedBody,    // 1 → approved
		approvedBody,    // 2 → approved
		disapprovedBody, // 3 → denied
		disapprovedBody, // 4 → denied
		disapprovedBody, // 5 → denied
	}
)

type FraudScoreHandler struct {
	vec          Vectorizer
	knn          KNNIndex
	requestCount atomic.Uint64
}

func NewFraudScoreHandler(vec Vectorizer, knn KNNIndex) *FraudScoreHandler {
	log.Printf("KNN inference: k=%d fraud_threshold=%.1f (majority vote >=3/%d)",
		knnK, knnFraudThreshold, knnK)
	return &FraudScoreHandler{vec: vec, knn: knn}
}

func (h *FraudScoreHandler) Handle(ctx *fasthttp.RequestCtx) {
	if !ctx.IsPost() {
		ctx.SetStatusCode(fasthttp.StatusMethodNotAllowed)
		return
	}

	body := ctx.PostBody()
	if len(body) == 0 {
		writeJSON(ctx, approvedBody)
		return
	}

	parsed := parseFraudRequest(body)

	vec := h.vec.VectorizeRaw(&parsed)

	nprobe := h.knn.NProbe()
	fraudCount := h.knn.PredictRaw(vec, nprobe)

	writeJSON(ctx, responses[fraudCount])

	if count := h.requestCount.Add(1); count%10000 == 0 {
		log.Printf("requests=%d", count)
	}
}

func writeJSON(ctx *fasthttp.RequestCtx, body []byte) {
	ctx.Response.Header.SetStatusCode(fasthttp.StatusOK)
	ctx.Response.Header.SetContentTypeBytes(contentTypeJSON)
	ctx.Response.SetBody(body)
}
