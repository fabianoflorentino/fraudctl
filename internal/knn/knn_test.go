package knn

import (
	"testing"

	"github.com/fabianoflorentino/fraudctl/internal/model"
)

func TestBruteForce_Predict(t *testing.T) {
	vectors := []model.Vector14{
		{0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		{0.2, 0.2, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		{0.3, 0.3, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		{0.9, 0.9, 0.9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		{0.95, 0.95, 0.95, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	}
	labels := []bool{true, true, true, false, false}

	bf := NewBruteForce()
	bf.Build(vectors, labels)

	tests := []struct {
		name    string
		query   model.Vector14
		k       int
		wantMin float64
		wantMax float64
	}{
		{
			name:    "close to fraud vectors",
			query:   model.Vector14{0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			k:       3,
			wantMin: 0.66,
			wantMax: 1.0,
		},
		{
			name:    "close to legit vectors",
			query:   model.Vector14{0.9, 0.9, 0.9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			k:       3,
			wantMin: 0.0,
			wantMax: 0.34,
		},
		{
			name:    "empty predictor",
			query:   model.Vector14{0.5, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			k:       5,
			wantMin: 0.0,
			wantMax: 1.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.name == "empty predictor" {
				empty := NewBruteForce()
				got := empty.Predict(tt.query, tt.k)
				if got != 0.0 {
					t.Errorf("Predict() = %v, want 0.0", got)
				}
				return
			}

			got := bf.Predict(tt.query, tt.k)
			if got < tt.wantMin || got > tt.wantMax {
				t.Errorf("Predict() = %v, want in [%v, %v]", got, tt.wantMin, tt.wantMax)
			}
		})
	}
}

func TestBruteForce_Count(t *testing.T) {
	vectors := []model.Vector14{
		{0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		{0.2, 0.2, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	}
	labels := []bool{true, false}

	bf := NewBruteForce()
	bf.Build(vectors, labels)

	if got := bf.Count(); got != 2 {
		t.Errorf("Count() = %v, want 2", got)
	}
}

func BenchmarkBruteForce_Predict_10k(b *testing.B) {
	vectors := make([]model.Vector14, 10000)
	labels := make([]bool, 10000)
	for i := range vectors {
		for j := 0; j < 14; j++ {
			vectors[i][j] = float32(i%100) / 100.0
		}
		labels[i] = i%3 == 0
	}

	bf := NewBruteForce()
	bf.Build(vectors, labels)

	query := model.Vector14{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = bf.Predict(query, 5)
	}
}

func BenchmarkBruteForce_Predict_100k(b *testing.B) {
	vectors := make([]model.Vector14, 100000)
	labels := make([]bool, 100000)
	for i := range vectors {
		for j := 0; j < 14; j++ {
			vectors[i][j] = float32(i%100) / 100.0
		}
		labels[i] = i%3 == 0
	}

	bf := NewBruteForce()
	bf.Build(vectors, labels)

	query := model.Vector14{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = bf.Predict(query, 5)
	}
}

func BenchmarkBruteForce_Predict_Parallel(b *testing.B) {
	vectors := make([]model.Vector14, 100000)
	labels := make([]bool, 100000)
	for i := range vectors {
		for j := 0; j < 14; j++ {
			vectors[i][j] = float32(i%100) / 100.0
		}
		labels[i] = i%3 == 0
	}

	bf := NewBruteForce()
	bf.Build(vectors, labels)

	query := model.Vector14{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5}

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			_ = bf.Predict(query, 5)
		}
	})
}
