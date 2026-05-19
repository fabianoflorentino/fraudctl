package handler

import (
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

type FraudScoreHandler struct {
	vec Vectorizer
	knn KNNIndex
}

func NewFraudScoreHandler(vec Vectorizer, knn KNNIndex) *FraudScoreHandler {
	return &FraudScoreHandler{vec: vec, knn: knn}
}

func (h *FraudScoreHandler) HandleFraudScore(body []byte) int {
	vec, err := h.vec.VectorizeJSON(body)
	if err != nil {
		return 0
	}
	return h.knn.PredictRaw(vec, h.knn.NProbe())
}

func (h *FraudScoreHandler) ResponseForCount(count int) []byte {
	if count < 0 || count > knnK {
		return precomputedResp[0]
	}
	return precomputedResp[count]
}
