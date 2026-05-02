package knn

import (
	"container/heap"

	"github.com/fabianoflorentino/fraudctl/internal/model"
)

type candidate struct {
	dist float32
	idx  int
}

type maxHeap []candidate

func (h maxHeap) Len() int            { return len(h) }
func (h maxHeap) Less(i, j int) bool  { return h[i].dist > h[j].dist }
func (h maxHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *maxHeap) Push(x interface{}) { *h = append(*h, x.(candidate)) }
func (h *maxHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

func brutePredict(flat []float32, fraudFlags []bool, n int, query model.Vector14, k int) float64 {
	h := make(maxHeap, 0, k+1)
	for i := 0; i < n; i++ {
		base := i * 14
		var sum float32
		for d := 0; d < 14; d++ {
			diff := query[d] - flat[base+d]
			sum += diff * diff
		}
		if len(h) < k {
			heap.Push(&h, candidate{sum, i})
		} else if sum < h[0].dist {
			h[0] = candidate{sum, i}
			heap.Fix(&h, 0)
		}
	}
	fraudCount := 0
	for _, c := range h {
		if c.idx < len(fraudFlags) && fraudFlags[c.idx] {
			fraudCount++
		}
	}
	if len(h) == 0 {
		return 0
	}
	return float64(fraudCount) / float64(len(h))
}
