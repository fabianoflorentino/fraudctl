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
	}, model.MCCRisk{
		"5411": 0.15,
	})

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

// TestDatasetKNN Tests KNN prediction with reference data.
func TestDatasetKNN(t *testing.T) {
	refs := []model.Reference{
		{Vector: model.Vector14{0.1, 0.1, 0.1}, Label: "fraud"},
		{Vector: model.Vector14{0.2, 0.2, 0.2}, Label: "fraud"},
		{Vector: model.Vector14{0.3, 0.3, 0.3}, Label: "fraud"},
	}

	ds := NewDataset(refs)
	knn := ds.KNN()

	score := knn.Predict(model.Vector14{0.1, 0.1, 0.1}, 1)
	if score != 1.0 {
		t.Fatalf("Predict() score = %v, want 1.0", score)
	}
}

// TestLoadDefaultInvalidPath Verifies that LoadDefault returns an error for invalid paths.
func TestLoadDefaultInvalidPath(t *testing.T) {
	_, err := LoadDefault("/path/that/does/not/exist")
	if err == nil {
		t.Fatalf("LoadDefault() expected error for invalid path")
	}
}