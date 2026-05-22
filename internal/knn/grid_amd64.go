//go:build amd64

package knn

func init() {
	useAVX2 = true
	useAVX2Scan = true
}

//go:noescape
//go:nosplit
func ScanBlock8AVX2(q *float32, block *int16, worst float32, sum *[8]float32) bool
