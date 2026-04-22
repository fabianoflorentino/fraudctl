package dataset

import (
	"testing"

	"github.com/fabianoflorentino/fraudctl/internal/model"
)

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
	if len(out.Dimensions) != 14 {
		t.Fatalf("Vectorize() dimensions = %d, want 14", len(out.Dimensions))
	}
	if out.Dimensions[12] != 0.15 {
		t.Fatalf("Vectorize() mcc risk = %v, want 0.15", out.Dimensions[12])
	}
}

func TestDatasetKNN(t *testing.T) {
	refs := []model.Reference{
		{Vector: []float64{0.1, 0.1, 0.1}, Label: "fraud"},
		{Vector: []float64{0.2, 0.2, 0.2}, Label: "fraud"},
		{Vector: []float64{0.3, 0.3, 0.3}, Label: "fraud"},
	}

	ds := NewDataset(refs)
	knn := ds.KNN(1)

	score, approved := knn.Predict([]float64{0.1, 0.1, 0.1})
	if score != 1 {
		t.Fatalf("Predict() score = %v, want 1", score)
	}
	if approved {
		t.Fatalf("Predict() approved = true, want false")
	}
}

func TestLoadDefaultInvalidPath(t *testing.T) {
	_, err := LoadDefault("/path/that/does/not/exist")
	if err == nil {
		t.Fatalf("LoadDefault() expected error for invalid path")
	}
}
