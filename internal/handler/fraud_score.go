package handler

import (
	"fmt"
	"strconv"
	"sync"
	"sync/atomic"

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

// approvalThreshold defines the fraud score cutoff for approval.
// Set below 0.5 because false negatives (missed fraud) cost 3× more than
// false positives, shifting the Bayes-optimal boundary toward 1/(1+3)=0.25.
const approvalThreshold = 0.5

var staticResponses [knn.K + 1][]byte

func init() {
	for i := 0; i <= knn.K; i++ {
		score := float64(i) / float64(knn.K)
		approved := score < approvalThreshold
		staticResponses[i] = fmt.Appendf(nil,
			`{"approved":%s,"fraud_score":%s}`,
			strconv.FormatBool(approved),
			strconv.FormatFloat(score, 'f', -1, 64),
		)
	}
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

	h.requestCount.Add(1)
}
