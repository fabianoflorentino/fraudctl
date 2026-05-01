package handler

import (
	"encoding/json"
	"io"
	"log"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

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

// FraudScoreHandler handles POST /fraud-score requests.
type FraudScoreHandler struct {
	vec          Vectorizer
	knn          KNNPredictor
	responsePool sync.Pool
	requestCount atomic.Uint64
}

// NewFraudScoreHandler creates a new handler with the given vectorizer and KNN predictor.
func NewFraudScoreHandler(vec Vectorizer, knn KNNPredictor) *FraudScoreHandler {
	h := &FraudScoreHandler{
		vec: vec,
		knn: knn,
	}
	h.responsePool.New = func() interface{} {
		return &model.FraudScoreResponse{}
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

	body, err := io.ReadAll(r.Body)
	if err != nil {
		log.Printf("Error reading request body: %v", err)
		return h.sendFallback(w)
	}
	r.Body.Close()

	var req model.FraudScoreRequest
	if err := json.Unmarshal(body, &req); err != nil {
		log.Printf("Error parsing JSON: %v", err)
		return h.sendFallback(w)
	}

	vec := h.vec.Vectorize(&req)
	fraudScore := h.knn.Predict(vec, 5)
	approved := fraudScore < 0.6

	resp := h.responsePool.Get().(*model.FraudScoreResponse)
	resp.Approved = approved
	resp.FraudScore = fraudScore

	respBytes, err := json.Marshal(resp)
	if err != nil {
		log.Printf("Error marshaling response: %v", err)
		h.responsePool.Put(resp)
		return h.sendFallback(w)
	}

	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(respBytes)

	h.responsePool.Put(resp)

	count := h.requestCount.Add(1)
	if count%100 == 0 {
		log.Printf("requests=%d latency=%s approved=%v fraud_score=%.2f", count, time.Since(start), approved, fraudScore)
	}

	return nil
}

// sendFallback returns a default response on error to avoid HTTP 500.
func (h *FraudScoreHandler) sendFallback(w ResponseWriter) error {
	w.WriteHeader(http.StatusOK)
	resp := []byte(`{"approved":true,"fraud_score":0.0}`)
	_, _ = w.Write(resp)
	return nil
}
