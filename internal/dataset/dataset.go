// Package dataset provides functionality for loading and managing the reference dataset.
package dataset

import (
	"os"
	"path/filepath"

	"github.com/fabianoflorentino/fraudctl/internal/gbdt"
	"github.com/fabianoflorentino/fraudctl/internal/knn"
	"github.com/fabianoflorentino/fraudctl/internal/model"
	"github.com/fabianoflorentino/fraudctl/internal/vectorizer"
)

// ============================================================================
// IVF KNN Performance Tuning Constants — 2-tier com EARLY EXIT
// ============================================================================
//
// Estratégia otimizada inspirada em joycegodinho/rinha-2026:
//   1. Quick probe: IVF_QUICK_PROBE clusters (saída RÁPIDA)
//   2. Verifica fraudCount:
//      - fraud ∈ {0,1,4,5} → resultado CLARO → retorna IMEDIATAMENTE (early exit)
//      - fraud ∈ {2,3}     → resultado AMBÍGUO → scan remaining clusters
//
// Por que isso funciona:
//   - Regra de decisão: >= 3/5 fraud neighbors → nega
//   - fraud=0,1 → claramente aprova (não pode chegar a 3 com vizinhos mais próximos)
//   - fraud=4,5 → claramente nega
//   - Apenas fraud=2,3 são limítrofes (adicionar vizinho pode inverter a decisão)
//
// Resultado: maioria das queries (~80-90%) saem cedo com apenas QUICK_PROBE clusters!
// ============================================================================

// IVF_NPROBE: total de clusters para casos ambíguos (quick + remaining)
// Com nlist=4096: nprobe=36 cobre 0.88% dos clusters.
const IVF_NPROBE = 36

// IVF_QUICK_PROBE: clusters para quick probe (early exit)
// Com nlist=4096: quickProbe=16 cobre 0.39% dos clusters.
const IVF_QUICK_PROBE = 16

// IVF_BOUNDARY_LO/HI: zona ambígua (apenas estes valores disparam re-score)
const IVF_BOUNDARY_LO = 2
const IVF_BOUNDARY_HI = 3

// IVF_RETRY_EXTRA: mantido para compatibilidade (não usado mais)
const IVF_RETRY_EXTRA = 0

// Dataset holds config and the KNN index needed to serve requests.
type Dataset struct {
	norm    model.NormalizationConstants
	mccRisk model.MCCRisk
	index   KNNIndex
	gbdt    *gbdt.GBDT
}

// KNNIndex is implemented by IVFIndex.
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
// Priority: ivf.bin (IVF approximate) > references.json.gz (fallback).
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

	// Priority 1: IVF index (primary - fast and accurate)
	ivfPath := filepath.Join(path, "ivf.bin")
	if _, err := os.Stat(ivfPath); err == nil {
		ivf, err := knn.LoadIVF(ivfPath)
		if err == nil {
			ivf.SetNProbe(IVF_NPROBE)
			ivf.SetRetry(IVF_QUICK_PROBE, IVF_BOUNDARY_LO, IVF_BOUNDARY_HI)
			idx = ivf
		}
	}

	// Priority 2: Brute force from gzip (fallback)
	if idx == nil {
		brute := knn.NewBruteIndex()
		gzPath := filepath.Join(path, "references.json.gz")
		if err := brute.BuildFromGzip(gzPath, 3_000_000); err != nil {
			return nil, err
		}
		idx = brute
	}

	ds := &Dataset{
		norm:    norm,
		mccRisk: mccRisk,
		index:   idx,
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
