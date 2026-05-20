//go:build amd64

package knn

func init() {
	useAVX2 = true
	useAVX2Scan = true
}

//go:noescape
//go:nosplit
func accumulateDotProductsAVX2(centroids []float32, nlist int, query [14]float32, out []float32)

//go:noescape
//go:nosplit
func vecSqDistAVX2(vec, q *int16) uint64

func scanClusterAVX2(vectors []int16, labels []byte, start, end int, q [DIM]int16, h *topK5) {
	if start >= end {
		return
	}
	_ = vectors[(end-1)*DIM+13]
	worst := h.worstDist()
	for i := start; i < end; i++ {
		dist := vecSqDistAVX2(&vectors[i*DIM], &q[0])
		if dist < worst {
			h.tryInsert(dist, i)
			worst = h.worstDist()
		}
	}
}
