package handler

import (
	"encoding/json"
	"fraudctl/internal/model"
	"fraudctl/internal/vectorizer"
	"io"
	"log"
	"net/http"
	"sync"
)

type Vectorizer interface {
	Vectorize(req *model.FraudScoreRequest) vectorizer.Vector
}

type KNNPredictor interface {
	Predict(vector []float64) (float64, bool)
}

type FraudScoreHandler struct {
	vec          Vectorizer
	knn          KNNPredictor
	responsePool sync.Pool
}

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

func (h *FraudScoreHandler) Handle(w ResponseWriter, r *http.Request) error {
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
	fraudScore, approved := h.knn.Predict(vec.Dimensions)

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
	return nil
}

func (h *FraudScoreHandler) sendFallback(w ResponseWriter) error {
	w.WriteHeader(http.StatusOK)
	resp := []byte(`{"approved":true,"fraud_score":0.0}`)
	_, _ = w.Write(resp)
	return nil
}
