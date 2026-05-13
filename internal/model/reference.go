// Package model provides data structures for the fraud detection API.
//
// This package defines the request and response types used in the fraud-score
// endpoint, along with reference data structures for the KNN algorithm.
package model

// Vector14 is a fixed-size 14-dimensional vector using float32.
// Uses 56 bytes per vector vs 112 bytes for []float64.
type Vector14 [14]float32

// Reference represents a labeled reference vector from the training dataset.
// Each reference has a 14-dimensional normalized vector and a label.
//
// The label indicates whether the transaction was:
//   - "fraud" - fraudulent transaction
//   - "legit" - legitimate transaction
type Reference struct {
	Vector Vector14 `json:"vector"`
	Label  string   `json:"label"`
}

// NormalizationConstants contains the maximum values used for feature normalization.
// These values are loaded from normalization.json and used to scale raw transaction
// data into the [0.0, 1.0] range for the 14-dimensional vector.
//
// The constants are:
//   - MaxAmount: Maximum transaction amount (default: 10000)
//   - MaxInstallments: Maximum number of installments (default: 12)
//   - AmountVsAvgRatio: Maximum ratio of amount to average (default: 10)
//   - MaxMinutes: Maximum minutes since last transaction (default: 1440 = 24h)
//   - MaxKm: Maximum distance in kilometers (default: 1000)
//   - MaxTxCount24h: Maximum transactions in 24h (default: 20)
//   - MaxMerchantAvgAmount: Maximum merchant average amount (default: 10000)
type NormalizationConstants struct {
	MaxAmount            float64 `json:"max_amount"`
	MaxInstallments      float64 `json:"max_installments"`
	AmountVsAvgRatio     float64 `json:"amount_vs_avg_ratio"`
	MaxMinutes           float64 `json:"max_minutes"`
	MaxKm                float64 `json:"max_km"`
	MaxTxCount24h        float64 `json:"max_tx_count_24h"`
	MaxMerchantAvgAmount float64 `json:"max_merchant_avg_amount"`
}

// MCCRisk maps Merchant Category Codes (MCC) to fraud risk scores.
// MCC codes are standardized categories that identify the type of merchant.
//
// Risk scores range from 0.0 (low risk) to 1.0 (high risk).
// Common high-risk MCCs include:
//   - 7995: Betting/Casino (0.85)
//   - 7801: Government lotteries (0.80)
//   - 7802: Horse racing (0.75)
//
// Uses fixed array of 10000 elements (4-digit MCC codes 0000-9999)
// for O(1) access without hash map overhead.
type MCCRisk [10000]float64

// Get returns the risk score for a given MCC code.
// Converts 4-character MCC string to array index.
// Returns 0.5 (medium risk) if MCC is invalid or not found.
func (m MCCRisk) Get(mcc []byte) float64 {
	if len(mcc) != 4 {
		return 0.5
	}
	idx := int(mcc[0]-'0')*1000 + int(mcc[1]-'0')*100 + int(mcc[2]-'0')*10 + int(mcc[3]-'0')
	if idx < 0 || idx >= len(m) {
		return 0.5
	}
	return m[idx]
}
