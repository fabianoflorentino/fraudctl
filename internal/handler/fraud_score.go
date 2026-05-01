package handler

import (
	"log"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	gojson "github.com/goccy/go-json"

	"github.com/fabianoflorentino/fraudctl/internal/knn"
	"github.com/fabianoflorentino/fraudctl/internal/model"
)

// Vectorizer interface for converting requests to vectors.
type Vectorizer interface {
	Vectorize(req *model.FraudScoreRequest) model.Vector14
}

// KNNPredictor interface for fraud score prediction.
type KNNPredictor interface {
	Predict(vector model.Vector14, k int) float64
	Count() int
}

// staticResponses holds pre-computed JSON response bytes for all possible
// fraud_score values. With K=5 and binary labels, fraud_score ∈ {0.0, 0.2,
// 0.4, 0.6, 0.8, 1.0}. Index is fraudCount (0..K). The threshold is 0.6:
// approved=true when fraudScore < 0.6 (i.e. fraudCount < 3).
//
//	index 0 → fraudCount=0 → score=0.0 → approved=true
//	index 1 → fraudCount=1 → score=0.2 → approved=true
//	index 2 → fraudCount=2 → score=0.4 → approved=true
//	index 3 → fraudCount=3 → score=0.6 → approved=false
//	index 4 → fraudCount=4 → score=0.8 → approved=false
//	index 5 → fraudCount=5 → score=1.0 → approved=false
var staticResponses = [knn.K + 1][]byte{
	[]byte(`{"approved":true,"fraud_score":0}`),
	[]byte(`{"approved":true,"fraud_score":0.2}`),
	[]byte(`{"approved":true,"fraud_score":0.4}`),
	[]byte(`{"approved":false,"fraud_score":0.6}`),
	[]byte(`{"approved":false,"fraud_score":0.8}`),
	[]byte(`{"approved":false,"fraud_score":1}`),
}

// FraudScoreHandler handles POST /fraud-score requests.
type FraudScoreHandler struct {
	vec          Vectorizer
	knn          KNNPredictor
	bufPool      sync.Pool
	requestCount atomic.Uint64
}

// NewFraudScoreHandler creates a new handler with the given vectorizer and KNN predictor.
func NewFraudScoreHandler(vec Vectorizer, knn KNNPredictor) *FraudScoreHandler {
	h := &FraudScoreHandler{
		vec: vec,
		knn: knn,
	}
	h.bufPool.New = func() interface{} {
		b := make([]byte, 0, 4096)
		return &b
	}
	return h
}

// Handle processes a fraud score request.
func (h *FraudScoreHandler) Handle(w ResponseWriter, r *http.Request) error {
	start := time.Now()

	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return nil
	}

	bufp := h.bufPool.Get().(*[]byte)
	*bufp = (*bufp)[:0]

	var n int
	var readErr error
	for {
		if len(*bufp) == cap(*bufp) {
			*bufp = append(*bufp, 0)[:len(*bufp)]
		}
		n, readErr = r.Body.Read((*bufp)[len(*bufp):cap(*bufp)])
		*bufp = (*bufp)[:len(*bufp)+n]
		if readErr != nil {
			break
		}
	}

	var req model.FraudScoreRequest
	err := gojson.Unmarshal(*bufp, &req)
	h.bufPool.Put(bufp)
	if err != nil {
		return h.sendFallback(w)
	}

	vec := h.vec.Vectorize(&req)
	fraudScore := h.knn.Predict(vec, knn.K)

	// Convert score to fraudCount index (0..K) for static response lookup.
	// fraudScore = fraudCount/K, so fraudCount = round(fraudScore * K).
	fraudCount := int(fraudScore*float64(knn.K) + 0.5)
	if fraudCount < 0 {
		fraudCount = 0
	} else if fraudCount > knn.K {
		fraudCount = knn.K
	}
	resp := staticResponses[fraudCount]

	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(resp)

	count := h.requestCount.Add(1)
	if count%1000 == 0 {
		log.Printf("requests=%d latency=%s fraud_score=%.2f", count, time.Since(start), fraudScore)
	}

	return nil
}

// sendFallback returns a default response on error to avoid HTTP 500.
func (h *FraudScoreHandler) sendFallback(w ResponseWriter) error {
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(staticResponses[0])
	return nil
}
