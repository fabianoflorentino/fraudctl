package knn

import (
	"fmt"
	"fraudctl/internal/vectorizer"
	"math"
	"math/rand"
	"sync"
	"testing"
)

func generateTestDataset(size int, numWorkers int) *Dataset {
	references := make([]Reference, size)
	r := rand.New(rand.NewSource(42))

	for i := 0; i < size; i++ {
		vec := vectorizer.Vector{
			Dimensions: make([]float64, 14),
		}
		for j := 0; j < 14; j++ {
			vec.Dimensions[j] = r.Float64()
		}
		references[i] = Reference{
			Vector:  vec,
			IsFraud: r.Float64() > 0.7,
		}
	}

	return NewDataset(references, numWorkers)
}

func BenchmarkKNN_Predict_1k(b *testing.B) {
	dataset := generateTestDataset(1000, 4)
	query := make([]float64, 14)
	for i := range query {
		query[i] = 0.5
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = dataset.Predict(query)
	}
}

func BenchmarkKNN_Predict_10k(b *testing.B) {
	dataset := generateTestDataset(10000, 4)
	query := make([]float64, 14)
	for i := range query {
		query[i] = 0.5
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = dataset.Predict(query)
	}
}

func BenchmarkKNN_Predict_100k(b *testing.B) {
	dataset := generateTestDataset(100000, 4)
	query := make([]float64, 14)
	for i := range query {
		query[i] = 0.5
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = dataset.Predict(query)
	}
}

func BenchmarkKNN_WorkerScaling(b *testing.B) {
	sizes := []int{1, 2, 4, 8}
	datasetSize := 100000
	query := make([]float64, 14)
	for i := range query {
		query[i] = 0.5
	}

	for _, workers := range sizes {
		b.Run(fmt.Sprintf("workers=%d", workers), func(b *testing.B) {
			dataset := generateTestDataset(datasetSize, workers)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _ = dataset.Predict(query)
			}
		})
	}
}

func BenchmarkEuclideanDistance(b *testing.B) {
	a := make([]float64, 14)
	bVec := make([]float64, 14)
	r := rand.New(rand.NewSource(42))
	for i := range a {
		a[i] = r.Float64()
		bVec[i] = r.Float64()
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = euclideanDistance(a, bVec)
	}
}

func BenchmarkTopK(b *testing.B) {
	results := make([]distanceResult, 100000)
	r := rand.New(rand.NewSource(42))
	for i := range results {
		results[i] = distanceResult{
			index:    i,
			distance: r.Float64(),
			isFraud:  r.Float64() > 0.7,
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = topK(results, 5)
	}
}

func BenchmarkPool_GetPut(b *testing.B) {
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			vec := GetVector()
			PutVector(vec)
		}
	})
}

var sink bool

func BenchmarkParallelKNN(b *testing.B) {
	dataset := generateTestDataset(100000, 4)
	query := make([]float64, 14)
	for i := range query {
		query[i] = 0.5
	}

	var mu sync.Mutex
	var total float64

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			score, approved := dataset.Predict(query)
			mu.Lock()
			if approved {
				total += score
			}
			mu.Unlock()
		}
	})

	sink = total > 0
	_ = sink
}

func BenchmarkMathSqrt(b *testing.B) {
	x := 123.456

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = math.Sqrt(x)
	}
}

func BenchmarkEuclideanSquared(b *testing.B) {
	a := make([]float64, 14)
	bVec := make([]float64, 14)
	r := rand.New(rand.NewSource(42))
	for i := range a {
		a[i] = r.Float64()
		bVec[i] = r.Float64()
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var sum float64
		for j := 0; j < 14; j++ {
			diff := a[j] - bVec[j]
			sum += diff * diff
		}
		_ = sum
	}
}
