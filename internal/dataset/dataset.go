package dataset

import (
	"fraudctl/internal/knn"
	"fraudctl/internal/model"
	"fraudctl/internal/vectorizer"
)

type Dataset struct {
	references []model.Reference

	vectors [][]float64
	labels  []bool

	norm    model.NormalizationConstants
	mccRisk model.MCCRisk
}

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

func (d *Dataset) Vectorizer() *vectorizer.Vectorizer {
	return vectorizer.New(d.norm, d.mccRisk)
}

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

func (d *Dataset) Count() int {
	return len(d.vectors)
}

func (d *Dataset) FraudCount() int {
	count := 0
	for _, label := range d.labels {
		if label {
			count++
		}
	}
	return count
}

func (d *Dataset) LegitCount() int {
	count := 0
	for _, label := range d.labels {
		if !label {
			count++
		}
	}
	return count
}

func (d *Dataset) SetConfig(norm model.NormalizationConstants, mccRisk model.MCCRisk) {
	d.norm = norm
	d.mccRisk = mccRisk
}

func LoadDefault(path string) (*Dataset, error) {
	loader := NewLoader(path)
	return loader.LoadAll()
}
