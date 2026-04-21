package vectorizer

import (
	"fraudctl/internal/model"
	"testing"
)

func BenchmarkVectorize(b *testing.B) {
	norm := model.NormalizationConstants{
		MaxAmount:            10000,
		MaxInstallments:      12,
		AmountVsAvgRatio:     10,
		MaxMinutes:           1440,
		MaxKm:                1000,
		MaxTxCount24h:        20,
		MaxMerchantAvgAmount: 10000,
	}
	mccRisk := model.MCCRisk{
		"5411": 0.15,
		"7995": 0.85,
	}

	v := New(norm, mccRisk)

	req := &model.FraudScoreRequest{
		ID: "tx-123",
		Transaction: model.TransactionData{
			Amount:       384.88,
			Installments: 3,
			RequestedAt:  "2026-03-11T20:23:35Z",
		},
		Customer: model.CustomerData{
			AvgAmount:      769.76,
			TxCount24h:     3,
			KnownMerchants: []string{"MERC-009", "MERC-001"},
		},
		Merchant: model.MerchantData{
			ID:        "MERC-001",
			MCC:       "5912",
			AvgAmount: 298.95,
		},
		Terminal: model.TerminalData{
			IsOnline:    false,
			CardPresent: true,
			KmFromHome:  13.7090520965,
		},
		LastTx: &model.LastTransactionData{
			Timestamp:     "2026-03-11T14:58:35Z",
			KmFromCurrent: 18.8626479774,
		},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = v.Vectorize(req)
	}
}

func BenchmarkVectorizeParallel(b *testing.B) {
	norm := model.NormalizationConstants{
		MaxAmount:            10000,
		MaxInstallments:      12,
		AmountVsAvgRatio:     10,
		MaxMinutes:           1440,
		MaxKm:                1000,
		MaxTxCount24h:        20,
		MaxMerchantAvgAmount: 10000,
	}
	mccRisk := model.MCCRisk{}

	v := New(norm, mccRisk)

	req := &model.FraudScoreRequest{
		Transaction: model.TransactionData{
			Amount:       100,
			Installments: 1,
			RequestedAt:  "2026-03-11T20:23:35Z",
		},
		Customer: model.CustomerData{
			AvgAmount:      100,
			TxCount24h:     1,
			KnownMerchants: []string{"MERC-001"},
		},
		Merchant: model.MerchantData{
			ID:        "MERC-001",
			MCC:       "5411",
			AvgAmount: 100,
		},
		Terminal: model.TerminalData{
			IsOnline:    false,
			CardPresent: true,
			KmFromHome:  10,
		},
	}

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			_ = v.Vectorize(req)
		}
	})
}

func BenchmarkGetVector(b *testing.B) {
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			vec := GetVector()
			PutVector(vec)
		}
	})
}
