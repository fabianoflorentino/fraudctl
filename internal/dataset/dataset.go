// Package dataset provides functionality for loading and managing the reference dataset.
package dataset

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/fabianoflorentino/fraudctl/internal/gbdt"
	"github.com/fabianoflorentino/fraudctl/internal/knn"
	"github.com/fabianoflorentino/fraudctl/internal/model"
	"github.com/fabianoflorentino/fraudctl/internal/vectorizer"
)

// Dataset holds config and the KNN index needed to serve requests.
type Dataset struct {
	norm    model.NormalizationConstants
	mccRisk model.MCCRisk
	index   KNNIndex
	gbdt    *gbdt.GBDT
}

// KNNIndex is implemented by GridIndex.
type KNNIndex interface {
	Predict(query model.Vector14, k int) float64
	PredictRaw(query model.Vector14, nprobe int) int
	NProbe() int
	Count() int
	FraudCount() int
}

// NewDataset creates a Dataset with an empty index (for testing).
func NewDataset(_ []model.Reference) *Dataset {
	return &Dataset{index: new(knn.GridIndex)}
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

// GBDT returns the loaded GBDT pre-filter model, or nil if not loaded.
func (d *Dataset) GBDT() *gbdt.GBDT { return d.gbdt }

// Count returns total reference vectors.
func (d *Dataset) Count() int { return d.index.Count() }

// FraudCount returns fraud reference count.
func (d *Dataset) FraudCount() int { return d.index.FraudCount() }

// LegitCount returns legit reference count.
func (d *Dataset) LegitCount() int { return d.index.Count() - d.index.FraudCount() }

// LoadVectorizerOnly loads only the normalization constants and MCC risk scores
// needed for the vectorizer, without loading the KNN index.
// Use this when the predictor is a standalone model (GBDT) that does
// not need the KNN index at runtime.
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

	// Use an empty KNN index (never queried in production with GBDT)
	return &Dataset{
		norm:    norm,
		mccRisk: mccRisk,
		index:   new(knn.GridIndex),
	}, nil
}

// LoadDefault loads the dataset from path.
// Expects grid.bin to be present (built at Docker build time).
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

	gridPath := filepath.Join(path, "grid.bin")
	gridIdx, err := knn.LoadGrid(gridPath)
	if err != nil {
		return nil, fmt.Errorf("load %s: %w", gridPath, err)
	}

	ds := &Dataset{
		norm:    norm,
		mccRisk: mccRisk,
		index:   gridIdx,
	}

	// Optional: GBDT pre-filter model (gbdt.bin or model.bin in resources dir)
	for _, name := range []string{"gbdt.bin", "model.bin"} {
		gbdtPath := filepath.Join(path, name)
		if _, err := os.Stat(gbdtPath); err == nil {
			if g, err := gbdt.Load(gbdtPath); err == nil {
				ds.gbdt = g
				break
			}
		}
	}

	return ds, nil
}
