package dataset

import (
	"testing"

	"github.com/fabianoflorentino/fraudctl/internal/model"
)

// TestDatasetVectorizerUsesConfiguredMCCRisk Verifies that the vectorizer correctly uses the configured MCC risk values.
func TestDatasetVectorizerUsesConfiguredMCCRisk(t *testing.T) {
	ds := NewDataset(nil)
	ds.SetConfig(model.NormalizationConstants{
		MaxAmount:            100,
		MaxInstallments:      10,
		AmountVsAvgRatio:     10,
		MaxMinutes:           1440,
		MaxKm:                100,
		MaxTxCount24h:        20,
		MaxMerchantAvgAmount: 1000,
	}, func() model.MCCRisk {
		var r model.MCCRisk
		for i := range r {
			r[i] = 0.5
		}
		r[5411] = 0.15
		return r
	}())

	v := ds.Vectorizer()
	req := &model.FraudScoreRequest{
		Transaction: model.TransactionData{
			Amount:       50,
			Installments: 1,
			RequestedAt:  "2026-03-11T10:00:00Z",
		},
		Customer: model.CustomerData{
			AvgAmount:      100,
			TxCount24h:     3,
			KnownMerchants: []string{"merchant-1"},
		},
		Merchant: model.MerchantData{
			ID:        "merchant-1",
			MCC:       "5411",
			AvgAmount: 120,
		},
		Terminal: model.TerminalData{
			IsOnline:    true,
			CardPresent: true,
			KmFromHome:  12,
		},
	}

	out := v.Vectorize(req)
	if len(out) != 14 {
		t.Fatalf("Vectorize() dimensions = %d, want 14", len(out))
	}
	if out[12] != 0.15 {
		t.Fatalf("Vectorize() mcc risk = %v, want 0.15", out[12])
	}
}

// TestDatasetKNN tests that KNN is accessible via Dataset.
func TestDatasetKNN(t *testing.T) {
	ds := NewDataset(nil)
	knn := ds.KNN()
	if knn == nil {
		t.Fatal("KNN() returned nil")
	}
	// Prediction is tested in knn/grid_test.go.
}

// TestLoadDefaultInvalidPath Verifies that LoadDefault returns an error for invalid paths.
func TestLoadDefaultInvalidPath(t *testing.T) {
	_, err := LoadDefault("/path/that/does/not/exist")
	if err == nil {
		t.Fatalf("LoadDefault() expected error for invalid path")
	}
}
