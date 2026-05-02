package handler

import (
	"log"
	"sync"
	"sync/atomic"
	"time"

	"github.com/valyala/fasthttp"

	gojson "github.com/goccy/go-json"

	"github.com/fabianoflorentino/fraudctl/internal/knn"
	"github.com/fabianoflorentino/fraudctl/internal/model"
)

type Vectorizer interface {
	Vectorize(req *model.FraudScoreRequest) model.Vector14
}

type KNNPredictor interface {
	Predict(vector model.Vector14, k int) float64
	Count() int
}

var staticResponses = [knn.K + 1][]byte{
	[]byte(`{"approved":true,"fraud_score":0}`),
	[]byte(`{"approved":true,"fraud_score":0.03}`),
	[]byte(`{"approved":true,"fraud_score":0.06}`),
	[]byte(`{"approved":true,"fraud_score":0.1}`),
	[]byte(`{"approved":true,"fraud_score":0.13}`),
	[]byte(`{"approved":true,"fraud_score":0.16}`),
	[]byte(`{"approved":true,"fraud_score":0.19}`),
	[]byte(`{"approved":true,"fraud_score":0.23}`),
	[]byte(`{"approved":true,"fraud_score":0.26}`),
	[]byte(`{"approved":true,"fraud_score":0.29}`),
	[]byte(`{"approved":true,"fraud_score":0.32}`),
	[]byte(`{"approved":true,"fraud_score":0.35}`),
	[]byte(`{"approved":true,"fraud_score":0.39}`),
	[]byte(`{"approved":true,"fraud_score":0.42}`),
	[]byte(`{"approved":true,"fraud_score":0.45}`),
	[]byte(`{"approved":true,"fraud_score":0.48}`),
	[]byte(`{"approved":false,"fraud_score":0.52}`),
	[]byte(`{"approved":false,"fraud_score":0.55}`),
	[]byte(`{"approved":false,"fraud_score":0.58}`),
	[]byte(`{"approved":false,"fraud_score":0.61}`),
	[]byte(`{"approved":false,"fraud_score":0.65}`),
	[]byte(`{"approved":false,"fraud_score":0.68}`),
	[]byte(`{"approved":false,"fraud_score":0.71}`),
	[]byte(`{"approved":false,"fraud_score":0.74}`),
	[]byte(`{"approved":false,"fraud_score":0.77}`),
	[]byte(`{"approved":false,"fraud_score":0.81}`),
	[]byte(`{"approved":false,"fraud_score":0.84}`),
	[]byte(`{"approved":false,"fraud_score":0.87}`),
	[]byte(`{"approved":false,"fraud_score":0.9}`),
	[]byte(`{"approved":false,"fraud_score":0.94}`),
	[]byte(`{"approved":false,"fraud_score":0.97}`),
	[]byte(`{"approved":false,"fraud_score":1}`),
}

type FraudScoreHandler struct {
	vec          Vectorizer
	knn          KNNPredictor
	bufPool      sync.Pool
	requestCount atomic.Uint64
}

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

func (h *FraudScoreHandler) Handle(ctx *fasthttp.RequestCtx) {
	start := time.Now()

	if !ctx.IsPost() {
		ctx.SetStatusCode(fasthttp.StatusMethodNotAllowed)
		return
	}

	bufp := h.bufPool.Get().(*[]byte)
	*bufp = (*bufp)[:0]

	body := ctx.PostBody()
	if cap(*bufp) >= len(body) {
		*bufp = (*bufp)[:len(body)]
		copy(*bufp, body)
	} else {
		*bufp = append(*bufp, body...)
	}

	var req model.FraudScoreRequest
	err := gojson.Unmarshal(*bufp, &req)
	h.bufPool.Put(bufp)
	if err != nil {
		ctx.SetStatusCode(fasthttp.StatusOK)
		ctx.Write(staticResponses[0])
		return
	}

	vec := h.vec.Vectorize(&req)
	fraudScore := h.knn.Predict(vec, knn.K)

	fraudCount := int(fraudScore*float64(knn.K) + 0.5)
	if fraudCount < 0 {
		fraudCount = 0
	} else if fraudCount > knn.K {
		fraudCount = knn.K
	}
	resp := staticResponses[fraudCount]

	ctx.SetStatusCode(fasthttp.StatusOK)
	ctx.SetContentType("application/json")
	ctx.Write(resp)

	count := h.requestCount.Add(1)
	if count%1000 == 0 {
		log.Printf("requests=%d latency=%s fraud_score=%.2f", count, time.Since(start), fraudScore)
	}
}
