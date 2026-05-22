//go:build amd64

package knn

func init() {
	useAVX2 = true
	useAVX2Scan = true
}

// ScanBlock8AVX2 computes squared L2 distances between a query vector and 8
// reference vectors stored in block-major int16 layout. It uses AVX2 SIMD
// instructions to process 8 vectors simultaneously with early exit at dim 8.
// Returns true if at least one vector has distance < worst.
//
//go:noescape
//go:nosplit
func ScanBlock8AVX2(q *float32, block *int16, worst float32, sum *[8]float32) bool
