package vectorizer

import "sync"

const VectorSize = 14

var vectorPool = sync.Pool{
	New: func() interface{} {
		return make([]float64, VectorSize)
	},
}

func GetVector() []float64 {
	return vectorPool.Get().([]float64)[:VectorSize]
}

func PutVector(vec []float64) {
	vectorPool.Put(vec[:VectorSize])
}

type VectorF32 struct {
	Dimensions [VectorSize]float32
}

var vectorF32Pool = sync.Pool{
	New: func() interface{} {
		return new(VectorF32)
	},
}

func GetVectorF32() *VectorF32 {
	return vectorF32Pool.Get().(*VectorF32)
}

func PutVectorF32(v *VectorF32) {
	vectorF32Pool.Put(v)
}

func VectorToF32(vec Vector) VectorF32 {
	var result VectorF32
	for i := 0; i < VectorSize; i++ {
		result.Dimensions[i] = float32(vec.Dimensions[i])
	}
	return result
}
