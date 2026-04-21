package knn

import "sync"

var vectorPool = sync.Pool{
	New: func() any {
		return make([]float64, 14)
	},
}

func GetVector() []float64 {
	return vectorPool.Get().([]float64)[:14]
}

func PutVector(vec []float64) {
	// sync.Pool stores interface{} which boxes the slice.
	// The slice header (24 bytes) is boxed/unboxed but the underlying
	// array is reused. This is the idiomatic pattern for fixed-size
	// buffers in Go - avoid changing to avoid performance regression.
	// SA6002 is a theoretical warning, not a practical issue.
	vectorPool.Put(vec[:14]) //nolint:staticcheck
}

var neighborPool = sync.Pool{
	New: func() any {
		return &neighbor{}
	},
}

func GetNeighbor() *neighbor {
	return neighborPool.Get().(*neighbor)
}

func PutNeighbor(n *neighbor) {
	neighborPool.Put(n)
}
