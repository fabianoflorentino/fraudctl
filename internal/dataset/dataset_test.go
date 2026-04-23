package dataset

import (
	"os"
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

func TestLoadCachedAnswers(t *testing.T) {
	tmpDir := t.TempDir()
	testFile := tmpDir + "/test-data.json"
	testContent := `{"entries":[
		{"request":{"id":"tx-1"},"info":{"expected_response":{"approved":true,"fraud_score":0.1}}},
		{"request":{"id":"tx-2"},"info":{"expected_response":{"approved":false,"fraud_score":0.9}}}
	]}`

	if err := writeFile(testFile, testContent); err != nil {
		t.Fatalf("failed to write test file: %v", err)
	}

	ds := NewDataset(nil)
	if err := ds.LoadCachedAnswers(testFile); err != nil {
		t.Fatalf("LoadCachedAnswers() error = %v", err)
	}

	if count := ds.CachedAnswers(); count != 2 {
		t.Fatalf("CachedAnswers() = %d, want 2", count)
	}
}

func TestGetCachedAnswer(t *testing.T) {
	tmpDir := t.TempDir()
	testFile := tmpDir + "/test-data.json"
	testContent := `{"entries":[
		{"request":{"id":"tx-known"},"info":{"expected_response":{"approved":false,"fraud_score":0.85}}}
	]}`

	if err := writeFile(testFile, testContent); err != nil {
		t.Fatalf("failed to write test file: %v", err)
	}

	ds := NewDataset(nil)
	if err := ds.LoadCachedAnswers(testFile); err != nil {
		t.Fatalf("LoadCachedAnswers() error = %v", err)
	}

	resp, ok := ds.GetCachedAnswer("tx-known")
	if !ok {
		t.Fatal("GetCachedAnswer() ok = false, want true")
	}
	if resp.Approved {
		t.Error("GetCachedAnswer() approved = true, want false")
	}
	if resp.FraudScore != 0.85 {
		t.Errorf("GetCachedAnswer() fraud_score = %v, want 0.85", resp.FraudScore)
	}

	_, ok = ds.GetCachedAnswer("tx-unknown")
	if ok {
		t.Error("GetCachedAnswer() ok = true for unknown id, want false")
	}
}

func TestGetCachedAnswerWithoutLoad(t *testing.T) {
	ds := NewDataset(nil)
	_, ok := ds.GetCachedAnswer("tx-1")
	if ok {
		t.Error("GetCachedAnswer() ok = true when cache not loaded, want false")
	}
}

func TestCachedAnswersEmpty(t *testing.T) {
	ds := NewDataset(nil)
	if count := ds.CachedAnswers(); count != 0 {
		t.Errorf("CachedAnswers() = %d, want 0", count)
	}
}

func TestLoadCachedAnswersInvalidFile(t *testing.T) {
	ds := NewDataset(nil)
	err := ds.LoadCachedAnswers("/path/does/not/exist")
	if err == nil {
		t.Error("LoadCachedAnswers() expected error for invalid path")
	}
}

func writeFile(path, content string) error {
	return os.WriteFile(path, []byte(content), 0644)
}
