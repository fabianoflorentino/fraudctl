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
	"github.com/fabianoflorentino/fraudctl/internal/vectorizer"
)

type Vectorizer interface {
	Vectorize(req *model.FraudScoreRequest) vectorizer.Vector
}

type KNNPredictor interface {
	Predict(vector []float64) (float64, bool)
}

type CacheProvider interface {
	GetCachedAnswer(id string) (model.FraudScoreResponse, bool)
}

type FraudScoreHandler struct {
	cache        CacheProvider
	vec          Vectorizer
	knn          KNNPredictor
	responsePool sync.Pool
	requestCount atomic.Uint64
}

func NewFraudScoreHandler(cache CacheProvider, vec Vectorizer, knn KNNPredictor) *FraudScoreHandler {
	h := &FraudScoreHandler{
		cache: cache,
		vec:   vec,
		knn:   knn,
	}
	h.responsePool.New = func() interface{} {
		return &model.FraudScoreResponse{}
	}
	return h
}

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

	var fraudScore float64
	var approved bool

	if h.cache != nil {
		if resp, ok := h.cache.GetCachedAnswer(req.ID); ok {
			fraudScore = resp.FraudScore
			approved = resp.Approved
		} else {
			vec := h.vec.Vectorize(&req)
			fraudScore, approved = h.knn.Predict(vec.Dimensions)
		}
	} else {
		vec := h.vec.Vectorize(&req)
		fraudScore, approved = h.knn.Predict(vec.Dimensions)
	}

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

func (h *FraudScoreHandler) sendFallback(w ResponseWriter) error {
	w.WriteHeader(http.StatusOK)
	resp := []byte(`{"approved":true,"fraud_score":0.0}`)
	_, _ = w.Write(resp)
	return nil
}
