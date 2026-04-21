package vectorizer

import (
	"fraudctl/internal/model"
	"time"
)

type Vectorizer struct {
	norm    model.NormalizationConstants
	mccRisk model.MCCRisk
}

func New(norm model.NormalizationConstants, mccRisk model.MCCRisk) *Vectorizer {
	return &Vectorizer{
		norm:    norm,
		mccRisk: mccRisk,
	}
}

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

type Vector struct {
	Dimensions []float64
}

func (v Vector) Clone() Vector {
	return Vector{
		Dimensions: append([]float64(nil), v.Dimensions...),
	}
}

func clamp(val float64) float64 {
	if val < 0 {
		return 0
	}
	if val > 1 {
		return 1
	}
	return val
}

func clampFloat32(val float32) float32 {
	if val < 0 {
		return 0
	}
	if val > 1 {
		return 1
	}
	return val
}

func ParseTimestamp(s string) time.Time {
	t, _ := time.Parse(time.RFC3339, s)
	return t
}
