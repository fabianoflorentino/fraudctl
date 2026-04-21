package knn

import (
	"fraudctl/internal/vectorizer"
	"testing"
)

func TestDataset_Predict(t *testing.T) {
	tests := []struct {
		name         string
		references   []Reference
		query        []float64
		workers      int
		wantApproved bool
	}{
		{
			name: "all fraud neighbors",
			references: []Reference{
				{Vector: vectorizer.Vector{Dimensions: []float64{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}}, IsFraud: true},
				{Vector: vectorizer.Vector{Dimensions: []float64{0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2}}, IsFraud: true},
				{Vector: vectorizer.Vector{Dimensions: []float64{0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3}}, IsFraud: true},
				{Vector: vectorizer.Vector{Dimensions: []float64{0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4}}, IsFraud: true},
				{Vector: vectorizer.Vector{Dimensions: []float64{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5}}, IsFraud: true},
				{Vector: vectorizer.Vector{Dimensions: []float64{0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6}}, IsFraud: false},
			},
			query:        []float64{0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15},
			workers:      2,
			wantApproved: false,
		},
		{
			name: "all legit neighbors",
			references: []Reference{
				{Vector: vectorizer.Vector{Dimensions: []float64{0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9}}, IsFraud: false},
				{Vector: vectorizer.Vector{Dimensions: []float64{0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8}}, IsFraud: false},
				{Vector: vectorizer.Vector{Dimensions: []float64{0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7}}, IsFraud: false},
				{Vector: vectorizer.Vector{Dimensions: []float64{0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6}}, IsFraud: false},
				{Vector: vectorizer.Vector{Dimensions: []float64{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5}}, IsFraud: false},
				{Vector: vectorizer.Vector{Dimensions: []float64{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}}, IsFraud: true},
			},
			query:        []float64{0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85},
			workers:      2,
			wantApproved: true,
		},
		{
			name: "threshold 0.6 borderline",
			references: []Reference{
				{Vector: vectorizer.Vector{Dimensions: []float64{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}}, IsFraud: true},
				{Vector: vectorizer.Vector{Dimensions: []float64{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}}, IsFraud: true},
				{Vector: vectorizer.Vector{Dimensions: []float64{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}}, IsFraud: false},
				{Vector: vectorizer.Vector{Dimensions: []float64{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}}, IsFraud: false},
				{Vector: vectorizer.Vector{Dimensions: []float64{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}}, IsFraud: false},
				{Vector: vectorizer.Vector{Dimensions: []float64{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}}, IsFraud: true},
			},
			query:        []float64{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1},
			workers:      2,
			wantApproved: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dataset := NewDataset(tt.references, tt.workers)
			_, approved := dataset.Predict(tt.query)

			if approved != tt.wantApproved {
				t.Errorf("Predict() approved = %v, want %v", approved, tt.wantApproved)
			}
		})
	}
}

func TestEuclideanDistance(t *testing.T) {
	tests := []struct {
		name     string
		a        []float64
		b        []float64
		wantDist float64
	}{
		{
			name:     "identical vectors",
			a:        []float64{1, 2, 3, 4},
			b:        []float64{1, 2, 3, 4},
			wantDist: 0,
		},
		{
			name:     "simple difference",
			a:        []float64{0, 0},
			b:        []float64{3, 4},
			wantDist: 25,
		},
		{
			name:     "one dimension only",
			a:        []float64{0},
			b:        []float64{3},
			wantDist: 9,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := euclideanDistance(tt.a, tt.b)
			if got != tt.wantDist {
				t.Errorf("euclideanDistance() = %v, want %v", got, tt.wantDist)
			}
		})
	}
}

func TestTopK(t *testing.T) {
	results := []distanceResult{
		{index: 0, distance: 5.0, isFraud: false},
		{index: 1, distance: 3.0, isFraud: true},
		{index: 2, distance: 1.0, isFraud: true},
		{index: 3, distance: 4.0, isFraud: false},
		{index: 4, distance: 2.0, isFraud: true},
		{index: 5, distance: 6.0, isFraud: false},
	}

	neighbors := topK(results, 5)

	if len(neighbors) != 5 {
		t.Errorf("topK() len = %d, want 5", len(neighbors))
	}

	for _, n := range neighbors {
		if n.Index < 0 || n.Index > 5 {
			t.Errorf("topK() invalid index %d", n.Index)
		}
	}
}

func TestNewDataset(t *testing.T) {
	references := []Reference{
		{Vector: vectorizer.Vector{Dimensions: []float64{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}}, IsFraud: true},
		{Vector: vectorizer.Vector{Dimensions: []float64{0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9}}, IsFraud: false},
	}

	dataset := NewDataset(references, 2)

	if len(dataset.vectors) != 2 {
		t.Errorf("NewDataset() vectors len = %d, want 2", len(dataset.vectors))
	}

	if len(dataset.fraudFlags) != 2 {
		t.Errorf("NewDataset() fraudFlags len = %d, want 2", len(dataset.fraudFlags))
	}
}

func TestKNNPool(t *testing.T) {
	vec := GetVector()
	if len(vec) != 14 {
		t.Errorf("GetVector() len = %d, want 14", len(vec))
	}

	vec[0] = 1.0
	PutVector(vec)

	vec2 := GetVector()
	vec2[0] = 0.5
	if vec2[0] != 0.5 {
		t.Error("Pool should return usable vector")
	}
	PutVector(vec2)
}
