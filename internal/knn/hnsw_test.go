package knn

import (
	"testing"

	"github.com/fabianoflorentino/fraudctl/internal/model"
)

func TestHNSW_Predict(t *testing.T) {
	vectors := make([]model.Vector14, 10)
	labels := make([]bool, 10)

	for i := 0; i < 5; i++ {
		vectors[i] = model.Vector14{0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
		labels[i] = true
	}
	for i := 5; i < 10; i++ {
		vectors[i] = model.Vector14{0.9, 0.9, 0.9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
		labels[i] = false
	}

	index := NewHNSWIndex()
	index.Build(vectors, labels)
	defer index.Free()

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
			k:       5,
			wantMin: 0.6,
			wantMax: 1.0,
		},
		{
			name:    "close to legit vectors",
			query:   model.Vector14{0.9, 0.9, 0.9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			k:       5,
			wantMin: 0.0,
			wantMax: 0.4,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := index.Predict(tt.query, tt.k)
			if got < tt.wantMin || got > tt.wantMax {
				t.Errorf("Predict() = %v, want in [%v, %v]", got, tt.wantMin, tt.wantMax)
			}
		})
	}
}

func TestHNSW_Count(t *testing.T) {
	vectors := []model.Vector14{
		{0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		{0.2, 0.2, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	}
	labels := []bool{true, false}

	index := NewHNSWIndex()
	index.Build(vectors, labels)
	defer index.Free()

	if got := index.Count(); got != 2 {
		t.Errorf("Count() = %v, want 2", got)
	}
}

func BenchmarkHNSW_Predict(b *testing.B) {
	vectors := make([]model.Vector14, 100000)
	labels := make([]bool, 100000)
	for i := range vectors {
		for j := 0; j < 14; j++ {
			vectors[i][j] = float32(i%100) / 100.0
		}
		labels[i] = i%3 == 0
	}

	index := NewHNSWIndex()
	index.Build(vectors, labels)
	defer index.Free()

	query := model.Vector14{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = index.Predict(query, 5)
	}
}

func BenchmarkHNSW_Predict_Parallel(b *testing.B) {
	vectors := make([]model.Vector14, 100000)
	labels := make([]bool, 100000)
	for i := range vectors {
		for j := 0; j < 14; j++ {
			vectors[i][j] = float32(i%100) / 100.0
		}
		labels[i] = i%3 == 0
	}

	index := NewHNSWIndex()
	index.Build(vectors, labels)
	defer index.Free()

	query := model.Vector14{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5}

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			_ = index.Predict(query, 5)
		}
	})
}
