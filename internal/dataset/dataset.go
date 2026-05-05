// Package dataset provides functionality for loading and managing the reference dataset.
package dataset

import (
	"os"
	"path/filepath"

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

// KNNIndex is implemented by IVFIndex, BruteIndex and BruteAVX2Index.
type KNNIndex interface {
	Predict(query model.Vector14, k int) float64
	PredictRaw(query model.Vector14, nprobe int) int
	NProbe() int
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

// LoadVectorizerOnly loads only the normalization constants and MCC risk scores
// needed for the vectorizer, without loading the KNN index.
// Use this when the predictor is a standalone model (GBDT/LightGBM) that does
// not need the KNN index at runtime. This avoids loading the ~95MB ivf.bin.
func LoadVectorizerOnly(path string) (*Dataset, error) {
	loader := NewLoader(path)

	norm, err := loader.LoadNormalization("")
	if err != nil {
		return nil, err
	}

	mccRisk, err := loader.LoadMCCRisk("")
	if err != nil {
		return nil, err
	}

	// Use an empty KNN index (never queried in production with GBDT/LightGBM)
	return &Dataset{
		norm:    norm,
		mccRisk: mccRisk,
		index:   knn.NewBruteIndex(),
	}, nil
}

// LoadDefault loads the dataset from path.
// Priority: brute.bin (exact brute-force KNN) > ivf.bin (IVF approx) > references.json.gz (fallback).
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

	// Priority 1: Brute-force AVX2 index (exact KNN, zero approximation error)
	// NOTE: Disabled - brute force O(N) is too slow at 3M vectors (causes timeouts).
	// Use IVF (Priority 2) as the primary index.
	brutePath := filepath.Join(path, "brute.bin")
	if false && knn.ExistsBrute(brutePath) {
		brute, err := knn.LoadBruteAVX2(brutePath)
		if err == nil {
			idx = brute
		}
	}

	// Priority 2: IVF index (primary - fast and accurate)
	if idx == nil {
		ivfPath := filepath.Join(path, "ivf.bin")
		if _, err := os.Stat(ivfPath); err == nil {
			ivf, err := knn.LoadIVF(ivfPath)
			if err == nil {
				ivf.SetNProbe(8)
				idx = ivf
			}
		}
	}

	// Priority 3: Brute force from gzip
	if idx == nil {
		brute := knn.NewBruteIndex()
		gzPath := filepath.Join(path, "references.json.gz")
		if err := brute.BuildFromGzip(gzPath, 3_000_000); err != nil {
			return nil, err
		}
		idx = brute
	}

	return &Dataset{
		norm:    norm,
		mccRisk: mccRisk,
		index:   idx,
	}, nil
}
