// Package dataset provides functionality for loading and managing the reference dataset.
//
// This package handles loading of the reference data files required for the fraud
// detection KNN algorithm, and provides a Dataset type that can be used to create
// vectorizers and KNN predictors.
package dataset

import (
	"github.com/fabianoflorentino/fraudctl/internal/knn"
	"github.com/fabianoflorentino/fraudctl/internal/model"
	"github.com/fabianoflorentino/fraudctl/internal/vectorizer"
)

// Dataset holds the reference data in memory for fraud detection.
type Dataset struct {
	references []model.Reference

	vectors []model.Vector14
	labels  []bool

	norm    model.NormalizationConstants
	mccRisk model.MCCRisk
}

// NewDataset creates a new Dataset from a slice of Reference vectors.
func NewDataset(references []model.Reference) *Dataset {
	count := len(references)

	vectors := make([]model.Vector14, count)
	labels := make([]bool, count)

	for i, ref := range references {
		vectors[i] = ref.Vector
		labels[i] = ref.Label == "fraud"
	}

	return &Dataset{
		references: references,
		vectors:    vectors,
		labels:     labels,
	}
}

// Vectorizer creates a new Vectorizer configured with the dataset's
// normalization constants and MCC risk scores.
func (d *Dataset) Vectorizer() *vectorizer.Vectorizer {
	return vectorizer.New(d.norm, d.mccRisk)
}

// KNN creates a new HNSW-based KNN predictor using hnswlib (C++).
// Build time is ~10s for 3M vectors with parallel insertion.
func (d *Dataset) KNN() *knn.HNSWIndex {
	index := knn.NewHNSWIndex()
	index.Build(d.vectors, d.labels)
	return index
}

// Count returns the total number of reference vectors in the dataset.
func (d *Dataset) Count() int {
	return len(d.vectors)
}

// FraudCount returns the number of fraudulent transactions in the reference dataset.
func (d *Dataset) FraudCount() int {
	count := 0
	for _, label := range d.labels {
		if label {
			count++
		}
	}
	return count
}

// LegitCount returns the number of legitimate transactions in the reference dataset.
func (d *Dataset) LegitCount() int {
	count := 0
	for _, label := range d.labels {
		if !label {
			count++
		}
	}
	return count
}

// SetConfig sets the normalization constants and MCC risk scores for the dataset.
func (d *Dataset) SetConfig(norm model.NormalizationConstants, mccRisk model.MCCRisk) {
	d.norm = norm
	d.mccRisk = mccRisk
}

// LoadDefault is a convenience function that loads the default dataset
// from the specified path. It creates a Loader and calls LoadAll.
func LoadDefault(path string) (*Dataset, error) {
	loader := NewLoader(path)
	return loader.LoadAll()
}
