package knn

import "sync"

var vectorPool = sync.Pool{
	New: func() interface{} {
		return make([]float64, 14)
	},
}

func GetVector() []float64 {
	return vectorPool.Get().([]float64)[:14]
}

func PutVector(vec []float64) {
	vectorPool.Put(vec[:14])
}

var neighborPool = sync.Pool{
	New: func() interface{} {
		return &neighbor{}
	},
}

func GetNeighbor() *neighbor {
	return neighborPool.Get().(*neighbor)
}

func PutNeighbor(n *neighbor) {
	neighborPool.Put(n)
}