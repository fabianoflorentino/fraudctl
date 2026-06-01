// Package knn implements k-nearest-neighbor search over 14-dimensional vectors.
package knn

import (
	"math"

	"github.com/fabianoflorentino/fraudctl/internal/model"
)

// K is the number of nearest neighbors used for fraud scoring.
const K = 5

// DIM is the dimensionality of the feature vectors.
const DIM = 14

// int16Scale is the scale factor for quantizing float32 vectors to int16.
const int16Scale = 10000

// quantizeQuery converts a 14D float32 vector to int16 using int16Scale.
func quantizeQuery(query model.Vector14) [DIM]int16 {
	var q [DIM]int16
	for i := 0; i < DIM; i++ {
		v := query[i]
		if v < 0 {
			q[i] = int16(v*int16Scale - 0.5)
		} else {
			q[i] = int16(v*int16Scale + 0.5)
		}
	}
	return q
}

// quantizeFloat32 converts a single float32 to int16 using int16Scale.
func quantizeFloat32(v float32) int16 {
	if v > 1.0 {
		return int16Scale
	}
	if v < -1.0 {
		return -int16Scale
	}
	return int16(math.Round(float64(v * int16Scale)))
}

// topK5 is a fixed-size sorted array tracking the K=5 nearest neighbours.
type topK5 struct {
	dist  [K]uint64
	idx   [K]int
	count int
}

//go:inline
func newTopK5() topK5 {
	var h topK5
	for i := range h.dist {
		h.dist[i] = math.MaxUint64
	}
	return h
}

//go:inline
func (h *topK5) worstDist() uint64 { return h.dist[K-1] }

//go:inline
func (h *topK5) tryInsert(dist uint64, vecIdx int) {
	if dist >= h.worstDist() {
		return
	}
	i := h.count
	if i == K {
		i = K - 1
	}
	for i > 0 && dist < h.dist[i-1] {
		h.dist[i] = h.dist[i-1]
		h.idx[i] = h.idx[i-1]
		i--
	}
	h.dist[i] = dist
	h.idx[i] = vecIdx
	if h.count < K {
		h.count++
	}
}

func (h *topK5) fraudCount(labels []byte) int {
	n := 0
	for i := 0; i < K; i++ {
		if h.dist[i] == math.MaxUint64 {
			break
		}
		vi := h.idx[i]
		if (labels[vi>>3] & (1 << uint(vi&7))) != 0 {
			n++
		}
	}
	return n
}
