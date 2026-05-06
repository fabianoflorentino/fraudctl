//go:build amd64

package knn

import (
	"unsafe"

	"golang.org/x/sys/cpu"
)

// useAVX2 is set at init time based on CPU feature detection.
var useAVX2 = cpu.X86.HasAVX2

// blockSize is the number of vectors packed in one SoA block.
const blockSize = 8

// blockStride is the byte size of one SoA block: DIM * blockSize * sizeof(int16).
// Layout: block[d*blockSize + lane] for d in [0,DIM), lane in [0,blockSize).
const blockStride = DIM * blockSize * 2 // 14 * 8 * 2 = 224 bytes

// scanBlock8AVX2 computes squared L2 distances from query to 8 vectors packed
// in SoA layout and writes them to out[0..7]. No return value; caller reads out.
//
//go:noescape
func scanBlock8AVX2(query *int16, block unsafe.Pointer, out *uint64)

// scanBlock32AVX2 processes 4 consecutive SoA blocks (32 vectors) in one call.
//
//go:noescape
func scanBlock32AVX2(query *int16, block unsafe.Pointer, out *uint64)
