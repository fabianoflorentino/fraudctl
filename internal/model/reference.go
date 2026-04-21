package model

type Reference struct {
	Vector []float64 `json:"vector"`
	Label  string    `json:"label"`
}

type ReferenceSlice []Reference

type NormalizationConstants struct {
	MaxAmount             float64 `json:"max_amount"`
	MaxInstallments       float64 `json:"max_installments"`
	AmountVsAvgRatio      float64 `json:"amount_vs_avg_ratio"`
	MaxMinutes            float64 `json:"max_minutes"`
	MaxKm                 float64 `json:"max_km"`
	MaxTxCount24h         float64 `json:"max_tx_count_24h"`
	MaxMerchantAvgAmount  float64 `json:"max_merchant_avg_amount"`
}

type MCCRisk map[string]float64

func (m MCCRisk) Get(mcc string) float64 {
	if val, ok := m[mcc]; ok {
		return val
	}
	return 0.5
}
