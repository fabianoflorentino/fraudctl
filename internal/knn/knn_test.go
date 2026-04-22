package knn

import (
	"testing"

	"github.com/fabianoflorentino/fraudctl/internal/vectorizer"
)

func TestDataset_Predict(t *testing.T) {
	tests := []struct {
		name         string
		references   []Reference
		query        []float64
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
			wantApproved: true,
		},
		{
			name: "threshold 0.6 borderline - approved",
			references: []Reference{
				{Vector: vectorizer.Vector{Dimensions: []float64{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}}, IsFraud: false},
				{Vector: vectorizer.Vector{Dimensions: []float64{0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2}}, IsFraud: false},
				{Vector: vectorizer.Vector{Dimensions: []float64{0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3}}, IsFraud: false},
				{Vector: vectorizer.Vector{Dimensions: []float64{0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4}}, IsFraud: false},
				{Vector: vectorizer.Vector{Dimensions: []float64{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5}}, IsFraud: false},
				{Vector: vectorizer.Vector{Dimensions: []float64{0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6}}, IsFraud: true},
			},
			query:        []float64{0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1},
			wantApproved: true,
		},
		{
			name: "threshold 0.6 borderline - denied",
			references: []Reference{
				{Vector: vectorizer.Vector{Dimensions: []float64{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}}, IsFraud: true},
				{Vector: vectorizer.Vector{Dimensions: []float64{0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2}}, IsFraud: true},
				{Vector: vectorizer.Vector{Dimensions: []float64{0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3}}, IsFraud: true},
				{Vector: vectorizer.Vector{Dimensions: []float64{0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4}}, IsFraud: true},
				{Vector: vectorizer.Vector{Dimensions: []float64{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5}}, IsFraud: false},
				{Vector: vectorizer.Vector{Dimensions: []float64{0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6}}, IsFraud: false},
			},
			query:        []float64{0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1},
			wantApproved: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dataset := NewDataset(tt.references, 1)
			_, approved := dataset.Predict(tt.query)

			if approved != tt.wantApproved {
				t.Errorf("Predict() approved = %v, want %v", approved, tt.wantApproved)
			}
		})
	}
}

func TestEuclideanDistanceSquared(t *testing.T) {
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
			got := euclideanDistanceSquared(tt.a, tt.b)
			if got != tt.wantDist {
				t.Errorf("euclideanDistanceSquared() = %v, want %v", got, tt.wantDist)
			}
		})
	}
}

func TestNewDataset(t *testing.T) {
	references := []Reference{
		{Vector: vectorizer.Vector{Dimensions: []float64{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}}, IsFraud: true},
		{Vector: vectorizer.Vector{Dimensions: []float64{0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9}}, IsFraud: false},
	}

	dataset := NewDataset(references, 1)

	if len(dataset.vectors) != 2 {
		t.Errorf("NewDataset() vectors len = %d, want 2", len(dataset.vectors))
	}

	if len(dataset.fraudFlags) != 2 {
		t.Errorf("NewDataset() fraudFlags len = %d, want 2", len(dataset.fraudFlags))
	}
}
