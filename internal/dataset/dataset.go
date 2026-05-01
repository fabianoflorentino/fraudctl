// Package dataset provides functionality for loading and managing the reference dataset.
package dataset

import (
	"os"
	"path/filepath"
	"runtime"

	"github.com/fabianoflorentino/fraudctl/internal/knn"
	"github.com/fabianoflorentino/fraudctl/internal/model"
	"github.com/fabianoflorentino/fraudctl/internal/vectorizer"
)

// Dataset holds config and the KNN index needed to serve requests.
type Dataset struct {
	norm    model.NormalizationConstants
	mccRisk model.MCCRisk
	index   KNNIndex
}

// KNNIndex is implemented by both BruteIndex and IVFIndex.
type KNNIndex interface {
	Predict(query model.Vector14, k int) float64
	Count() int
	FraudCount() int
}

// NewDataset creates a Dataset with an empty BruteIndex (for testing).
func NewDataset(refs []model.Reference) *Dataset {
	idx := knn.NewBruteIndex()
	if len(refs) > 0 {
		vectors := make([]model.Vector14, len(refs))
		labels := make([]bool, len(refs))
		for i, r := range refs {
			vectors[i] = r.Vector
			labels[i] = r.Label == "fraud"
		}
		idx.Build(vectors, labels)
	}
	return &Dataset{index: idx}
}

// SetConfig sets normalization constants and MCC risk scores.
func (d *Dataset) SetConfig(norm model.NormalizationConstants, mccRisk model.MCCRisk) {
	d.norm = norm
	d.mccRisk = mccRisk
}

// Vectorizer returns a Vectorizer configured with the dataset's constants.
func (d *Dataset) Vectorizer() *vectorizer.Vectorizer {
	return vectorizer.New(d.norm, d.mccRisk)
}

// Index returns the KNN index.
func (d *Dataset) Index() KNNIndex { return d.index }

// KNN returns the KNN index (backward-compat alias).
func (d *Dataset) KNN() KNNIndex { return d.index }

// Count returns total reference vectors.
func (d *Dataset) Count() int { return d.index.Count() }

// FraudCount returns fraud reference count.
func (d *Dataset) FraudCount() int { return d.index.FraudCount() }

// LegitCount returns legit reference count.
func (d *Dataset) LegitCount() int { return d.index.Count() - d.index.FraudCount() }

// LoadDefault loads the dataset from path.
// If ivf.bin exists, loads the pre-built IVF index (fast, low memory).
// Otherwise, streams references.json.gz into a BruteIndex (slower queries).
func LoadDefault(path string) (*Dataset, error) {
	loader := NewLoader(path)

	norm, err := loader.LoadNormalization("")
	if err != nil {
		return nil, err
	}

	mccRisk, err := loader.LoadMCCRisk("")
	if err != nil {
		return nil, err
	}

	var idx KNNIndex

	ivfPath := filepath.Join(path, "ivf.bin")
	if _, err := os.Stat(ivfPath); err == nil {
		// Fast path: load pre-built IVF index.
		ivf, err := knn.LoadIVF(ivfPath)
		if err != nil {
			return nil, err
		}
		ivf.SetNProbe(8)
		idx = ivf
	} else {
		// Fallback: stream gzip directly into BruteIndex.
		brute := knn.NewBruteIndex()
		gzPath := filepath.Join(path, "references.json.gz")
		if err := brute.BuildFromGzip(gzPath, 3_000_000); err != nil {
			return nil, err
		}
		idx = brute
	}

	runtime.GC()

	return &Dataset{
		norm:    norm,
		mccRisk: mccRisk,
		index:   idx,
	}, nil
}
