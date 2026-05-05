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
	"math"

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
//
// All values are rounded to 4 decimal places to match the reference dataset.
func (v *Vectorizer) Vectorize(req *model.FraudScoreRequest) model.Vector14 {
	var vec model.Vector14

	vec[0] = round4(clampFloat32(float32(req.Transaction.Amount) * v.invMaxAmount))
	vec[1] = round4(clampFloat32(float32(req.Transaction.Installments) * v.invMaxInstall))

	var amountVsAvg float64
	if req.Customer.AvgAmount > 0 {
		amountVsAvg = req.Transaction.Amount / req.Customer.AvgAmount
	}
	vec[2] = round4(clampFloat32(float32(amountVsAvg) * v.invAmountRatio))

	hour, dayOfWeek := parseHourAndWeekday(req.Transaction.RequestedAt)

	vec[3] = round4(float32(hour) / 23.0)
	vec[4] = round4(float32(dayOfWeek) / 6.0)

	if req.LastTx != nil {
		reqSec := parseUnixSeconds(req.Transaction.RequestedAt)
		lastSec := parseUnixSeconds(req.LastTx.Timestamp)
		minutes := float32(reqSec-lastSec) / 60.0
		vec[5] = round4(clampFloat32(minutes * v.invMaxMinutes))
		vec[6] = round4(clampFloat32(float32(req.LastTx.KmFromCurrent) * v.invMaxKm))
	} else {
		vec[5] = -1
		vec[6] = -1
	}

	vec[7] = round4(clampFloat32(float32(req.Terminal.KmFromHome) * v.invMaxKm))
	vec[8] = round4(clampFloat32(float32(req.Customer.TxCount24h) * v.invMaxTxCount))

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
	vec[13] = round4(clampFloat32(float32(req.Merchant.AvgAmount) * v.invMaxMerchantAvg))

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

// round4 rounds a float32 to 4 decimal places, matching the reference dataset precision.
func round4(v float32) float32 {
	return float32(math.Round(float64(v)*10000) * 0.0001)
}

// parseHourAndWeekday extracts UTC hour (0-23) and weekday (0=Sun..6=Sat)
// from an RFC3339 string without allocating a time.Time.
// Format: "2006-01-02T15:04:05Z" or "2006-01-02T15:04:05+00:00"
// Returns (0, 0) on parse failure.
func parseHourAndWeekday(s string) (hour, weekday int) {
	// Need at minimum "YYYY-MM-DDTHH" = 13 chars.
	if len(s) < 13 || s[10] != 'T' {
		return 0, 0
	}
	// Parse hour directly from position 11-12.
	h := int(s[11]-'0')*10 + int(s[12]-'0')
	// Parse full date to compute weekday (Zeller's congruence).
	if len(s) < 10 {
		return h, 0
	}
	year := int(s[0]-'0')*1000 + int(s[1]-'0')*100 + int(s[2]-'0')*10 + int(s[3]-'0')
	month := int(s[5]-'0')*10 + int(s[6]-'0')
	day := int(s[8]-'0')*10 + int(s[9]-'0')
	// Tomohiko Sakamoto's algorithm for day-of-week (0=Sun).
	t := [12]int{0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4}
	if month < 3 {
		year--
	}
	wd := (year + year/4 - year/100 + year/400 + t[month-1] + day) % 7
	// Convert from 0=Sun to 0=Mon (matching the data generator).
	return h, (wd + 6) % 7
}

// parseUnixSeconds parses an RFC3339 string to seconds since Unix epoch.
// Zero-alloc. Returns 0 on failure.
func parseUnixSeconds(s string) int64 {
	if len(s) < 19 || s[10] != 'T' {
		return 0
	}
	year := int64(s[0]-'0')*1000 + int64(s[1]-'0')*100 + int64(s[2]-'0')*10 + int64(s[3]-'0')
	month := int64(s[5]-'0')*10 + int64(s[6]-'0')
	day := int64(s[8]-'0')*10 + int64(s[9]-'0')
	hour := int64(s[11]-'0')*10 + int64(s[12]-'0')
	min := int64(s[14]-'0')*10 + int64(s[15]-'0')
	sec := int64(s[17]-'0')*10 + int64(s[18]-'0')

	// Days from epoch using Julian Day Number approach.
	// Formula: days since 1970-01-01
	y, m := year, month
	if m <= 2 {
		y--
		m += 12
	}
	a := y / 100
	b := 2 - a + a/4
	jd := int64(365.25*float64(y+4716)) + int64(30.6001*float64(m+1)) + day + int64(b) - 1524
	days := jd - 2440588 // 2440588 = Julian Day for 1970-01-01

	return days*86400 + hour*3600 + min*60 + sec
}
