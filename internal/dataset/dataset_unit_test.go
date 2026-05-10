package dataset

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/fabianoflorentino/fraudctl/internal/model"
)

func TestNewDataset_Empty(t *testing.T) {
	ds := NewDataset(nil)
	if ds == nil {
		t.Fatal("NewDataset returned nil")
	}
	if ds.Count() != 0 {
		t.Errorf("Count = %d, want 0", ds.Count())
	}
	if ds.FraudCount() != 0 {
		t.Errorf("FraudCount = %d, want 0", ds.FraudCount())
	}
	if ds.LegitCount() != 0 {
		t.Errorf("LegitCount = %d, want 0", ds.LegitCount())
	}
}

func TestNewDataset_WithRefs(t *testing.T) {
	refs := []model.Reference{
		{Vector: model.Vector14{0.1, 0.1, 0.1}, Label: "fraud"},
		{Vector: model.Vector14{0.2, 0.2, 0.2}, Label: "fraud"},
		{Vector: model.Vector14{0.9, 0.9, 0.9}, Label: "legit"},
	}
	ds := NewDataset(refs)
	if ds.Count() != 3 {
		t.Errorf("Count = %d, want 3", ds.Count())
	}
	if ds.FraudCount() != 2 {
		t.Errorf("FraudCount = %d, want 2", ds.FraudCount())
	}
	if ds.LegitCount() != 1 {
		t.Errorf("LegitCount = %d, want 1", ds.LegitCount())
	}
}

func TestDataset_SetConfig(t *testing.T) {
	ds := NewDataset(nil)
	norm := model.NormalizationConstants{
		MaxAmount:            5000,
		MaxInstallments:      6,
		AmountVsAvgRatio:     5,
		MaxMinutes:           720,
		MaxKm:                500,
		MaxTxCount24h:        10,
		MaxMerchantAvgAmount: 5000,
	}
	mccRisk := func() model.MCCRisk {
		var r model.MCCRisk
		for i := range r {
			r[i] = 0.5
		}
		r[5411] = 0.15
		return r
	}()

	ds.SetConfig(norm, mccRisk)

	if ds.norm.MaxAmount != 5000 {
		t.Errorf("norm.MaxAmount = %v, want 5000", ds.norm.MaxAmount)
	}
	if ds.mccRisk.Get("5411") != 0.15 {
		t.Errorf("mccRisk.Get(5411) = %v, want 0.15", ds.mccRisk.Get("5411"))
	}
}

func TestDataset_Vectorizer(t *testing.T) {
	ds := NewDataset(nil)
	ds.SetConfig(
		model.NormalizationConstants{MaxAmount: 10000, MaxInstallments: 12, AmountVsAvgRatio: 10, MaxMinutes: 1440, MaxKm: 1000, MaxTxCount24h: 20, MaxMerchantAvgAmount: 10000},
		func() model.MCCRisk {
			var r model.MCCRisk
			for i := range r {
				r[i] = 0.5
			}
			return r
		}(),
	)

	v := ds.Vectorizer()
	if v == nil {
		t.Fatal("Vectorizer returned nil")
	}

	req := &model.FraudScoreRequest{
		Transaction: model.TransactionData{
			Amount:       100,
			Installments: 1,
			RequestedAt:  "2026-03-11T10:00:00Z",
		},
		Customer: model.CustomerData{
			AvgAmount:  100,
			TxCount24h: 1,
		},
		Merchant: model.MerchantData{
			MCC: "5411",
		},
		Terminal: model.TerminalData{
			KmFromHome: 10,
		},
	}

	_ = v.Vectorize(req)
}

func TestDataset_KNN_Predict(t *testing.T) {
	refs := []model.Reference{
		{Vector: model.Vector14{0.1, 0.1, 0.1}, Label: "fraud"},
		{Vector: model.Vector14{0.2, 0.2, 0.2}, Label: "fraud"},
		{Vector: model.Vector14{0.9, 0.9, 0.9}, Label: "legit"},
	}
	ds := NewDataset(refs)

	knn := ds.KNN()
	if knn == nil {
		t.Fatal("KNN returned nil")
	}

	score := knn.Predict(model.Vector14{0.1, 0.1, 0.1}, 1)
	if score != 1.0 {
		t.Errorf("Predict = %v, want 1.0", score)
	}
	score = knn.Predict(model.Vector14{0.9, 0.9, 0.9}, 1)
	if score != 0.0 {
		t.Errorf("Predict = %v, want 0.0", score)
	}
}

func TestDataset_GBDT(t *testing.T) {
	ds := NewDataset(nil)
	gbdt := ds.GBDT()
	if gbdt != nil {
		t.Error("GBDT should be nil when not loaded")
	}
}

func TestDataset_Index(t *testing.T) {
	ds := NewDataset(nil)
	idx := ds.Index()
	if idx == nil {
		t.Fatal("Index returned nil")
	}
}

func TestLoadVectorizerOnly_InvalidPath(t *testing.T) {
	_, err := LoadVectorizerOnly("/nonexistent/path")
	if err == nil {
		t.Fatal("expected error for invalid path")
	}
}

func TestLoadVectorizerOnly_InvalidNormalization(t *testing.T) {
	tmpDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(tmpDir, "mcc_risk.json"), []byte(`{"5411": 0.15}`), 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	// No normalization.json → should fail
	_, err := LoadVectorizerOnly(tmpDir)
	if err == nil {
		t.Fatal("expected error when normalization.json is missing")
	}
}

func TestLoadVectorizerOnly_InvalidMCCRisk(t *testing.T) {
	tmpDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(tmpDir, "normalization.json"), []byte(`{"max_amount": 10000}`), 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	// No mcc_risk.json → should fail
	_, err := LoadVectorizerOnly(tmpDir)
	if err == nil {
		t.Fatal("expected error when mcc_risk.json is missing")
	}
}

func TestLoadVectorizerOnly_Success(t *testing.T) {
	tmpDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(tmpDir, "normalization.json"), []byte(`{"max_amount": 10000, "max_installments": 12, "amount_vs_avg_ratio": 10, "max_minutes": 1440, "max_km": 1000, "max_tx_count_24h": 20, "max_merchant_avg_amount": 10000}`), 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	if err := os.WriteFile(filepath.Join(tmpDir, "mcc_risk.json"), []byte(`{"5411": 0.15}`), 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	ds, err := LoadVectorizerOnly(tmpDir)
	if err != nil {
		t.Fatalf("LoadVectorizerOnly failed: %v", err)
	}
	if ds == nil {
		t.Fatal("LoadVectorizerOnly returned nil")
	}
	if ds.Count() != 0 {
		t.Errorf("Count = %d, want 0", ds.Count())
	}
	v := ds.Vectorizer()
	if v == nil {
		t.Fatal("Vectorizer returned nil")
	}
}

func TestLoadDefault_InvalidPath(t *testing.T) {
	_, err := LoadDefault("/nonexistent/path/that/does/not/exist")
	if err == nil {
		t.Fatal("expected error for invalid path")
	}
}

func TestLoader_LoadAll_InvalidPath(t *testing.T) {
	loader := NewLoader("/nonexistent")
	_, err := loader.LoadAll()
	if err == nil {
		t.Fatal("expected error for invalid path")
	}
}

func TestLoader_LoadMCCRisk_InvalidCode(t *testing.T) {
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "mcc_risk.json")
	if err := os.WriteFile(path, []byte(`{"12": 0.5}`), 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	loader := NewLoader("")
	risk, err := loader.LoadMCCRisk(path)
	if err != nil {
		t.Fatalf("LoadMCCRisk failed: %v", err)
	}

	if risk.Get("0012") != 0.5 {
		t.Errorf("MCCRisk.Get(0012) = %v, want 0.5", risk.Get("0012"))
	}
}

func TestLoader_LoadMCCRisk_DefaultValues(t *testing.T) {
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "mcc_risk.json")
	if err := os.WriteFile(path, []byte(`{"5411": 0.15}`), 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	loader := NewLoader("")
	risk, err := loader.LoadMCCRisk(path)
	if err != nil {
		t.Fatalf("LoadMCCRisk failed: %v", err)
	}

	if risk.Get("9999") != 0.5 {
		t.Errorf("MCCRisk.Get(9999) = %v, want 0.5 (default)", risk.Get("9999"))
	}
}
