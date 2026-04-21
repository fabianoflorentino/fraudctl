// Package dataset provides functionality for loading and managing the reference dataset.
//
// This package handles loading of the reference data files required for the fraud
// detection KNN algorithm, and provides a Dataset type that can be used to create
// vectorizers and KNN predictors.
package dataset

import (
	"fraudctl/internal/knn"
	"fraudctl/internal/model"
	"fraudctl/internal/vectorizer"
)

// Dataset holds the reference data in memory for fraud detection.
// It pre-allocates slices for vectors and labels for efficient memory usage.
type Dataset struct {
	references []model.Reference

	vectors [][]float64
	labels  []bool

	norm    model.NormalizationConstants
	mccRisk model.MCCRisk
}

// NewDataset creates a new Dataset from a slice of Reference vectors.
// It pre-allocates memory for vectors and labels for optimal performance.
//
// The references slice is processed to extract:
//   - vectors: 14-dimensional normalized vectors
//   - labels: boolean fraud indicators (true = fraud, false = legit)
func NewDataset(references []model.Reference) *Dataset {
	count := len(references)

	vectors := make([][]float64, count)
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
//
// The returned Vectorizer can be used to convert incoming transaction
// requests into 14-dimensional vectors for KNN prediction.
func (d *Dataset) Vectorizer() *vectorizer.Vectorizer {
	return vectorizer.New(d.norm, d.mccRisk)
}

// KNN creates a new KNN predictor configured with the dataset's reference vectors.
// The numWorkers parameter specifies the number of goroutines to use for
// parallel distance computation. Use 1 for single-threaded operation.
func (d *Dataset) KNN(numWorkers int) *knn.Dataset {
	references := make([]knn.Reference, len(d.vectors))

	for i := range d.vectors {
		references[i] = knn.Reference{
			Vector: vectorizer.Vector{
				Dimensions: d.vectors[i],
			},
			IsFraud: d.labels[i],
		}
	}

	return knn.NewDataset(references, numWorkers)
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
// This is typically called after loading the data from files.
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
