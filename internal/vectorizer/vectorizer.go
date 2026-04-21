// Package vectorizer provides functionality to convert transaction requests
// into 14-dimensional normalized vectors for fraud detection using KNN.
//
// The vectorizer normalizes transaction data according to the Rinha 2026
// specification, creating a feature vector that can be used to find
// similar transactions in the reference dataset.
//
// # Vector Dimensions
//
// The 14 dimensions are:
//
//	 0: amount (normalized by max_amount)
//	 1: installments (normalized by max_installments)
//	 2: amount vs customer average (normalized by ratio)
//	 3: hour of day (0-23 mapped to 0-1)
//	 4: day of week (0-6 mapped to 0-1)
//	 5: minutes since last transaction (-1 if null)
//	 6: km from last transaction (-1 if null)
//	 7: km from home
//	 8: transaction count in 24h
//	 9: is online (1/0)
//	10: card present (1/0)
//	11: unknown merchant (1 if not in known merchants)
//	12: MCC risk score
//	 13: merchant average amount
//
// # Usage
//
//	vec := vectorizer.New(norm, mccRisk)
//	vector := vec.Vectorize(request)
package vectorizer

import (
	"fraudctl/internal/model"
	"time"
)

// Vectorizer normalizes transaction data into a 14-dimensional vector.
type Vectorizer struct {
	norm    model.NormalizationConstants
	mccRisk model.MCCRisk
}

// New creates a new Vectorizer with the given normalization constants
// and MCC risk mappings.
func New(norm model.NormalizationConstants, mccRisk model.MCCRisk) *Vectorizer {
	return &Vectorizer{
		norm:    norm,
		mccRisk: mccRisk,
	}
}

// Vectorize converts a FraudScoreRequest into a 14-dimensional normalized vector.
// Uses sync.Pool for memory efficiency.
//
// The returned vector contains normalized values in the range [0.0, 1.0],
// except for dimensions 5 and 6 (minutes_since_last_tx, km_from_last_tx)
// which use -1 as a sentinel value when LastTx is nil.
func (v *Vectorizer) Vectorize(req *model.FraudScoreRequest) Vector {
	vec := GetVector()
	defer PutVector(vec)

	vec[0] = clamp(float64(req.Transaction.Amount) / v.norm.MaxAmount)
	vec[1] = clamp(float64(req.Transaction.Installments) / v.norm.MaxInstallments)

	amountVsAvg := req.Transaction.Amount / req.Customer.AvgAmount
	vec[2] = clamp(amountVsAvg / v.norm.AmountVsAvgRatio)

	requestedAt, _ := req.Transaction.RequestedAtTime()
	hour := float64(requestedAt.UTC().Hour())
	dayOfWeek := float64(int(requestedAt.UTC().Weekday()))

	vec[3] = hour / 23.0
	vec[4] = dayOfWeek / 6.0

	if req.LastTx != nil {
		lastTxTime, _ := req.LastTx.TimestampTime()
		minutes := requestedAt.Sub(lastTxTime).Minutes()
		vec[5] = clamp(minutes / v.norm.MaxMinutes)
		vec[6] = clamp(req.LastTx.KmFromCurrent / v.norm.MaxKm)
	} else {
		vec[5] = -1
		vec[6] = -1
	}

	vec[7] = clamp(req.Terminal.KmFromHome / v.norm.MaxKm)
	vec[8] = clamp(float64(req.Customer.TxCount24h) / v.norm.MaxTxCount24h)

	if req.Terminal.IsOnline {
		vec[9] = 1
	} else {
		vec[9] = 0
	}

	if req.Terminal.CardPresent {
		vec[10] = 1
	} else {
		vec[10] = 0
	}

	knownMerchant := false
	for _, m := range req.Customer.KnownMerchants {
		if m == req.Merchant.ID {
			knownMerchant = true
			break
		}
	}
	if knownMerchant {
		vec[11] = 0
	} else {
		vec[11] = 1
	}

	vec[12] = v.mccRisk.Get(req.Merchant.MCC)
	vec[13] = clamp(req.Merchant.AvgAmount / v.norm.MaxMerchantAvgAmount)

	result := Vector{
		Dimensions: make([]float64, VectorSize),
	}
	copy(result.Dimensions, vec)

	return result
}

// Vector represents a 14-dimensional feature vector for fraud detection.
type Vector struct {
	Dimensions []float64
}

// Clone creates an independent copy of the vector.
func (v Vector) Clone() Vector {
	return Vector{
		Dimensions: append([]float64(nil), v.Dimensions...),
	}
}

// clamp restricts a value to the range [0.0, 1.0].
func clamp(val float64) float64 {
	if val < 0 {
		return 0
	}
	if val > 1 {
		return 1
	}
	return val
}

// clampFloat32 restricts a value to the range [0.0, 1.0] for float32.
func clampFloat32(val float32) float32 {
	if val < 0 {
		return 0
	}
	if val > 1 {
		return 1
	}
	return val
}

// ParseTimestamp parses an RFC3339 timestamp string.
func ParseTimestamp(s string) time.Time {
	t, _ := time.Parse(time.RFC3339, s)
	return t
}
