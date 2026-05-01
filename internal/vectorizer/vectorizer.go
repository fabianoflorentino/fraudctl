// Package vectorizer provides functionality to convert transaction requests
// into 14-dimensional normalized vectors for fraud detection using KNN.
//
// The vectorizer normalizes transaction data according to the Rinha 2026
// specification, creating a feature vector that can be used to find
// similar transactions in the reference dataset.
//
// # Vector Dimensions
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
	"time"

	"github.com/fabianoflorentino/fraudctl/internal/model"
)

// Vectorizer normalizes transaction data into a 14-dimensional vector.
type Vectorizer struct {
	norm              model.NormalizationConstants
	mccRisk           model.MCCRisk
	invMaxAmount      float32
	invMaxInstall     float32
	invAmountRatio    float32
	invMaxMinutes     float32
	invMaxKm          float32
	invMaxTxCount     float32
	invMaxMerchantAvg float32
}

// New creates a new Vectorizer with the given normalization constants
// and MCC risk mappings. Pre-computes inverse constants for faster division.
func New(norm model.NormalizationConstants, mccRisk model.MCCRisk) *Vectorizer {
	return &Vectorizer{
		norm:              norm,
		mccRisk:           mccRisk,
		invMaxAmount:      float32(1.0 / norm.MaxAmount),
		invMaxInstall:     float32(1.0 / norm.MaxInstallments),
		invAmountRatio:    float32(1.0 / norm.AmountVsAvgRatio),
		invMaxMinutes:     float32(1.0 / norm.MaxMinutes),
		invMaxKm:          float32(1.0 / norm.MaxKm),
		invMaxTxCount:     float32(1.0 / norm.MaxTxCount24h),
		invMaxMerchantAvg: float32(1.0 / norm.MaxMerchantAvgAmount),
	}
}

// Vectorize converts a FraudScoreRequest into a 14-dimensional normalized vector.
//
// The returned vector contains normalized values in the range [0.0, 1.0],
// except for dimensions 5 and 6 (minutes_since_last_tx, km_from_last_tx)
// which use -1 as a sentinel value when LastTx is nil.
func (v *Vectorizer) Vectorize(req *model.FraudScoreRequest) model.Vector14 {
	var vec model.Vector14

	vec[0] = clampFloat32(float32(req.Transaction.Amount) * v.invMaxAmount)
	vec[1] = clampFloat32(float32(req.Transaction.Installments) * v.invMaxInstall)

	var amountVsAvg float64
	if req.Customer.AvgAmount > 0 {
		amountVsAvg = req.Transaction.Amount / req.Customer.AvgAmount
	}
	vec[2] = clampFloat32(float32(amountVsAvg) * v.invAmountRatio)

	requestedAt, _ := req.Transaction.RequestedAtTime()
	hour := float32(requestedAt.UTC().Hour())
	dayOfWeek := float32(int(requestedAt.UTC().Weekday()))

	vec[3] = hour / 23.0
	vec[4] = dayOfWeek / 6.0

	if req.LastTx != nil {
		lastTxTime, _ := req.LastTx.TimestampTime()
		minutes := float32(requestedAt.Sub(lastTxTime).Minutes())
		vec[5] = clampFloat32(minutes * v.invMaxMinutes)
		vec[6] = clampFloat32(float32(req.LastTx.KmFromCurrent) * v.invMaxKm)
	} else {
		vec[5] = -1
		vec[6] = -1
	}

	vec[7] = clampFloat32(float32(req.Terminal.KmFromHome) * v.invMaxKm)
	vec[8] = clampFloat32(float32(req.Customer.TxCount24h) * v.invMaxTxCount)

	if req.Terminal.IsOnline {
		vec[9] = 1
	}

	if req.Terminal.CardPresent {
		vec[10] = 1
	}

	knownMerchant := false
	for _, m := range req.Customer.KnownMerchants {
		if m == req.Merchant.ID {
			knownMerchant = true
			break
		}
	}
	if !knownMerchant {
		vec[11] = 1
	}

	vec[12] = float32(v.mccRisk.Get(req.Merchant.MCC))
	vec[13] = clampFloat32(float32(req.Merchant.AvgAmount) * v.invMaxMerchantAvg)

	return vec
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
