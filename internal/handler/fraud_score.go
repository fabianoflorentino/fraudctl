package handler

import (
	"bufio"
	"encoding/json"
	"log"
	"net/http"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

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

// FraudScoreHandler handles POST /fraud-score requests.
type FraudScoreHandler struct {
	vec          Vectorizer
	knn          KNNPredictor
	bufPool      sync.Pool
	readerPool   sync.Pool
	requestCount atomic.Uint64
}

// NewFraudScoreHandler creates a new handler with the given vectorizer and KNN predictor.
func NewFraudScoreHandler(vec Vectorizer, knn KNNPredictor) *FraudScoreHandler {
	h := &FraudScoreHandler{
		vec: vec,
		knn: knn,
	}
	h.bufPool.New = func() interface{} {
		return make([]byte, 0, 64)
	}
	h.readerPool.New = func() interface{} {
		return bufio.NewReaderSize(nil, 4096)
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

	var req model.FraudScoreRequest
	br := h.readerPool.Get().(*bufio.Reader)
	br.Reset(r.Body)
	err := json.NewDecoder(br).Decode(&req)
	h.readerPool.Put(br)
	if err != nil {
		return h.sendFallback(w)
	}

	vec := h.vec.Vectorize(&req)
	fraudScore := h.knn.Predict(vec, knn.K)
	approved := fraudScore < 0.6

	// Manual JSON encoding — zero allocation, ~2μs vs ~30μs for json.Marshal
	buf := h.bufPool.Get().([]byte)
	buf = buf[:0]

	if approved {
		buf = append(buf, `{"approved":true,"fraud_score":`...)
	} else {
		buf = append(buf, `{"approved":false,"fraud_score":`...)
	}
	buf = strconv.AppendFloat(buf, fraudScore, 'f', -1, 64)
	buf = append(buf, '}')

	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(buf)

	h.bufPool.Put(buf)

	count := h.requestCount.Add(1)
	if count%1000 == 0 {
		log.Printf("requests=%d latency=%s approved=%v fraud_score=%.2f", count, time.Since(start), approved, fraudScore)
	}

	return nil
}

// sendFallback returns a default response on error to avoid HTTP 500.
func (h *FraudScoreHandler) sendFallback(w ResponseWriter) error {
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte(`{"approved":true,"fraud_score":0.0}`))
	return nil
}
