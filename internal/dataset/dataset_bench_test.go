package dataset

import (
	"testing"

	"github.com/fabianoflorentino/fraudctl/internal/model"
)

func BenchmarkNewDataset(b *testing.B) {
	references := make([]model.Reference, 100000)
	for i := range references {
		refs := make([]float64, 14)
		for j := range refs {
			refs[j] = float64(i%100) / 100.0
		}
		references[i] = model.Reference{
			Vector: refs,
			Label:  "fraud",
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = NewDataset(references)
	}
}

func BenchmarkDataset_KNN(b *testing.B) {
	references := make([]model.Reference, 10000)
	for i := range references {
		refs := make([]float64, 14)
		for j := range refs {
			refs[j] = float64(i%100) / 100.0
		}
		references[i] = model.Reference{
			Vector: refs,
			Label:  "fraud",
		}
	}

	ds := NewDataset(references)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = ds.KNN(1)
	}
}

func BenchmarkDataset_Count(b *testing.B) {
	references := make([]model.Reference, 100000)
	for i := range references {
		refs := make([]float64, 14)
		for j := range refs {
			refs[j] = float64(i%100) / 100.0
		}
		references[i] = model.Reference{
			Vector: refs,
			Label:  "fraud",
		}
	}

	ds := NewDataset(references)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = ds.Count()
	}
}

func BenchmarkDataset_FraudCount(b *testing.B) {
	references := make([]model.Reference, 100000)
	for i := range references {
		refs := make([]float64, 14)
		label := "legit"
		if i%3 == 0 {
			label = "fraud"
		}
		for j := range refs {
			refs[j] = float64(i%100) / 100.0
		}
		references[i] = model.Reference{
			Vector: refs,
			Label:  label,
		}
	}

	ds := NewDataset(references)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = ds.FraudCount()
	}
}

func BenchmarkLoadNormalization(b *testing.B) {
	loader := NewLoader("")
	path := "../../resources/normalization.json"

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = loader.LoadNormalization(path)
	}
}

func BenchmarkLoadMCCRisk(b *testing.B) {
	loader := NewLoader("")
	path := "../../resources/mcc_risk.json"

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = loader.LoadMCCRisk(path)
	}
}
