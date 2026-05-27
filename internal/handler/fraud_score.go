// Package handler provides HTTP request handlers for the fraud detection API.
//
// The main handler, FraudScoreHandler, processes incoming fraud score requests
// by vectorizing the JSON payload and querying the KNN index to determine
// the fraud score based on the 5 nearest neighbors.
package handler

import (
	"github.com/fabianoflorentino/fraudctl/internal/model"
)

// Vectorizer defines the interface for converting transaction data into
// a 14-dimensional feature vector used by the KNN index.
type Vectorizer interface {
	Vectorize(req *model.FraudScoreRequest) model.Vector14
	VectorizeJSON(data []byte) (model.Vector14, error)
}

// KNNIndex defines the interface for the k-nearest neighbors search index.
// Implementations can be GridIndex (KD-tree), IVFIndex, or BruteIndex.
type KNNIndex interface {
	Predict(query model.Vector14, k int) float64
	PredictRaw(query model.Vector14, nprobe int) int
}

const knnK = 5

var precomputedResp [knnK + 1][]byte

func init() {
	precomputedResp[0] = []byte(`{"approved":true,"fraud_score":0.0}`)
	precomputedResp[1] = []byte(`{"approved":true,"fraud_score":0.2}`)
	precomputedResp[2] = []byte(`{"approved":true,"fraud_score":0.4}`)
	precomputedResp[3] = []byte(`{"approved":false,"fraud_score":0.6}`)
	precomputedResp[4] = []byte(`{"approved":false,"fraud_score":0.8}`)
	precomputedResp[5] = []byte(`{"approved":false,"fraud_score":1.0}`)
}

// FraudScoreHandler handles fraud score requests by vectorizing the input
// and querying the KNN index. It returns the count of fraudulent neighbors
// among the K=5 nearest neighbors.
type FraudScoreHandler struct {
	vec Vectorizer
	knn KNNIndex
}

// NewFraudScoreHandler creates a new FraudScoreHandler with the given
// vectorizer and KNN index.
func NewFraudScoreHandler(vec Vectorizer, knn KNNIndex) *FraudScoreHandler {
	return &FraudScoreHandler{vec: vec, knn: knn}
}

// HandleFraudScore processes a raw JSON request body and returns the count
// of fraudulent neighbors (0-5) and whether a parse error occurred.
// Returns 0 on parse errors as a fallback.
func (h *FraudScoreHandler) HandleFraudScore(body []byte) (int, bool) {
	vec, err := h.vec.VectorizeJSON(body)
	if err != nil {
		return 0, true
	}
	return h.knn.PredictRaw(vec, 0), false
}

// ResponseForCount returns the precomputed JSON response for the given
// fraud count. Returns the approved/0.0 response for out-of-range counts.
func (h *FraudScoreHandler) ResponseForCount(count int) []byte {
	if count < 0 || count > knnK {
		return precomputedResp[0]
	}
	return precomputedResp[count]
}
