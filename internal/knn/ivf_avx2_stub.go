//go:build !amd64

package knn

import "unsafe"

// useAVX2 is always false on non-amd64 architectures.
var useAVX2 = false

const blockSize = 8
const blockStride = DIM * blockSize * 2 // 224 bytes

// scanBlock8AVX2 is not available on this architecture; stub panics if called.
func scanBlock8AVX2(query *int16, block unsafe.Pointer, out *uint64) {
	panic("AVX2 not available on this architecture")
}

// scanBlock32AVX2 is not available on this architecture; stub panics if called.
func scanBlock32AVX2(query *int16, block unsafe.Pointer, out *uint64) {
	panic("AVX2 not available on this architecture")
}
