package knn

import (
	"fraudctl/internal/vectorizer"
	"math/rand"
	"sync"
	"testing"
)

func generateTestDataset(size int) *Dataset {
	references := make([]Reference, size)
	r := rand.New(rand.NewSource(42))

	for i := 0; i < size; i++ {
		vec := vectorizer.Vector{
			Dimensions: make([]float64, Dimensions),
		}
		for j := 0; j < Dimensions; j++ {
			vec.Dimensions[j] = r.Float64()
		}
		references[i] = Reference{
			Vector:  vec,
			IsFraud: r.Float64() > 0.7,
		}
	}

	return NewDataset(references, 1)
}

func BenchmarkKNN_Predict_1k(b *testing.B) {
	dataset := generateTestDataset(1000)
	query := make([]float64, Dimensions)
	for i := range query {
		query[i] = 0.5
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = dataset.Predict(query)
	}
}

func BenchmarkKNN_Predict_10k(b *testing.B) {
	dataset := generateTestDataset(10000)
	query := make([]float64, Dimensions)
	for i := range query {
		query[i] = 0.5
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = dataset.Predict(query)
	}
}

func BenchmarkKNN_Predict_100k(b *testing.B) {
	dataset := generateTestDataset(100000)
	query := make([]float64, Dimensions)
	for i := range query {
		query[i] = 0.5
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = dataset.Predict(query)
	}
}

func BenchmarkEuclideanDistanceSquared(b *testing.B) {
	a := make([]float64, Dimensions)
	bVec := make([]float64, Dimensions)
	r := rand.New(rand.NewSource(42))
	for i := range a {
		a[i] = r.Float64()
		bVec[i] = r.Float64()
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = euclideanDistanceSquared(a, bVec)
	}
}

func BenchmarkParallelKNN(b *testing.B) {
	dataset := generateTestDataset(100000)
	query := make([]float64, Dimensions)
	for i := range query {
		query[i] = 0.5
	}

	var mu sync.Mutex
	var total float64

	b.ResetTimer()
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

	_ = total
}
