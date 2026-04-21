// Package vectorizer provides functionality to convert transaction requests
// into 14-dimensional normalized vectors for fraud detection using KNN.
//
// This file provides object pooling for memory efficiency.
package vectorizer

import "sync"

// VectorSize is the number of dimensions in the feature vector.
const VectorSize = 14

// vectorPool is a sync.Pool for reusing float64 slices.
// This reduces GC pressure in high-throughput scenarios.
var vectorPool = sync.Pool{
	New: func() any {
		return make([]float64, VectorSize)
	},
}

// GetVector retrieves a vector from the pool.
// Always returns a slice of length VectorSize.
func GetVector() []float64 {
	return vectorPool.Get().([]float64)[:VectorSize]
}

// PutVector returns a vector to the pool for reuse.
func PutVector(vec []float64) {
	vectorPool.Put(vec[:VectorSize])
}

// VectorF32 is a fixed-size vector using float32 for lower memory usage.
type VectorF32 struct {
	Dimensions [VectorSize]float32
}

// vectorF32Pool is a sync.Pool for VectorF32 structs.
var vectorF32Pool = sync.Pool{
	New: func() any {
		return new(VectorF32)
	},
}

// GetVectorF32 retrieves a VectorF32 from the pool.
func GetVectorF32() *VectorF32 {
	return vectorF32Pool.Get().(*VectorF32)
}

// PutVectorF32 returns a VectorF32 to the pool for reuse.
func PutVectorF32(v *VectorF32) {
	vectorF32Pool.Put(v)
}

// VectorToF32 converts a float64 Vector to a float32 VectorF32.
func VectorToF32(vec Vector) VectorF32 {
	var result VectorF32
	for i := 0; i < VectorSize; i++ {
		result.Dimensions[i] = float32(vec.Dimensions[i])
	}
	return result
}
