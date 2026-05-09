package vectorizer

import (
	"math"
	"testing"

	"github.com/fabianoflorentino/fraudctl/internal/model"
)

func TestNew(t *testing.T) {
	norm := model.NormalizationConstants{
		MaxAmount:            10000,
		MaxInstallments:      12,
		AmountVsAvgRatio:     10,
		MaxMinutes:           1440,
		MaxKm:                1000,
		MaxTxCount24h:        20,
		MaxMerchantAvgAmount: 10000,
	}
	mccRisk := func() model.MCCRisk {
		var r model.MCCRisk
		for i := range r {
			r[i] = 0.5
		}
		return r
	}()

	v := New(norm, mccRisk)
	if v == nil {
		t.Fatal("New returned nil")
	}
	if v.invMaxAmount != 1.0/10000 {
		t.Errorf("invMaxAmount = %v, want %v", v.invMaxAmount, 1.0/10000)
	}
	if v.invMaxInstall != 1.0/12 {
		t.Errorf("invMaxInstall = %v, want %v", v.invMaxInstall, 1.0/12)
	}
}

func TestNew_ZeroValues(t *testing.T) {
	norm := model.NormalizationConstants{}
	mccRisk := model.MCCRisk{}

	v := New(norm, mccRisk)
	if v == nil {
		t.Fatal("New returned nil")
	}
}

func TestVectorize_KnownMerchant(t *testing.T) {
	v := New(
		model.NormalizationConstants{MaxAmount: 10000, MaxInstallments: 12, AmountVsAvgRatio: 10, MaxMinutes: 1440, MaxKm: 1000, MaxTxCount24h: 20, MaxMerchantAvgAmount: 10000},
		func() model.MCCRisk {
			var r model.MCCRisk
			for i := range r {
				r[i] = 0.5
			}
			return r
		}(),
	)

	req := &model.FraudScoreRequest{
		Transaction: model.TransactionData{
			Amount:       100,
			Installments: 1,
			RequestedAt:  "2026-03-11T10:00:00Z",
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
			IsOnline:    true,
			CardPresent: true,
			KmFromHome:  10,
		},
		LastTx: &model.LastTransactionData{
			Timestamp:     "2026-03-11T09:00:00Z",
			KmFromCurrent: 5,
		},
	}

	vec := v.Vectorize(req)

	if vec[9] != 1 {
		t.Errorf("is_online = %v, want 1", vec[9])
	}
	if vec[10] != 1 {
		t.Errorf("card_present = %v, want 1", vec[10])
	}
	if vec[11] != 0 {
		t.Errorf("unknown_merchant = %v, want 0", vec[11])
	}
}

func TestVectorize_UnknownMerchant(t *testing.T) {
	v := New(
		model.NormalizationConstants{MaxAmount: 10000, MaxInstallments: 12, AmountVsAvgRatio: 10, MaxMinutes: 1440, MaxKm: 1000, MaxTxCount24h: 20, MaxMerchantAvgAmount: 10000},
		func() model.MCCRisk {
			var r model.MCCRisk
			for i := range r {
				r[i] = 0.5
			}
			return r
		}(),
	)

	req := &model.FraudScoreRequest{
		Transaction: model.TransactionData{
			Amount:       100,
			Installments: 1,
			RequestedAt:  "2026-03-11T10:00:00Z",
		},
		Customer: model.CustomerData{
			AvgAmount:      100,
			TxCount24h:     1,
			KnownMerchants: []string{"MERC-001"},
		},
		Merchant: model.MerchantData{
			ID:        "MERC-999",
			MCC:       "5411",
			AvgAmount: 100,
		},
		Terminal: model.TerminalData{
			IsOnline:    false,
			CardPresent: false,
			KmFromHome:  10,
		},
	}

	vec := v.Vectorize(req)

	if vec[9] != 0 {
		t.Errorf("is_online = %v, want 0", vec[9])
	}
	if vec[10] != 0 {
		t.Errorf("card_present = %v, want 0", vec[10])
	}
	if vec[11] != 1 {
		t.Errorf("unknown_merchant = %v, want 1", vec[11])
	}
}

func TestRound4(t *testing.T) {
	tests := []struct {
		v    float32
		want float32
	}{
		{0.12345, 0.1235},
		{0.12344, 0.1234},
		{1.0000, 1.0000},
		{0, 0},
		{0.99999, 1.0},
	}
	for _, tt := range tests {
		got := round4(tt.v)
		if math.Abs(float64(got-tt.want)) > 0.0001 {
			t.Errorf("round4(%v) = %v, want %v", tt.v, got, tt.want)
		}
	}
}

func TestParseUnixSeconds(t *testing.T) {
	tests := []struct {
		input string
		want  int64
	}{
		{"2026-03-11T20:23:35Z", 1773260615},
		{"1970-01-01T00:00:00Z", 0},
		{"2026-01-01T00:00:00Z", 1767225600},
		{"invalid", 0},
		{"", 0},
	}
	for _, tt := range tests {
		got := parseUnixSeconds(tt.input)
		if got != tt.want {
			t.Errorf("parseUnixSeconds(%q) = %d, want %d", tt.input, got, tt.want)
		}
	}
}

func TestParseUnixSeconds_InvalidFormat(t *testing.T) {
	tests := []string{
		"short",
		"no-t-separator",
		"2026-03-11 20:23:35Z",
	}
	for _, s := range tests {
		got := parseUnixSeconds(s)
		if got != 0 {
			t.Errorf("parseUnixSeconds(%q) = %d, want 0", s, got)
		}
	}
}

func TestParseHourAndWeekday_EdgeCases(t *testing.T) {
	tests := []struct {
		input       string
		wantHour    int
		wantWeekday int
	}{
		{"2026-01-01T00:00:00Z", 0, 3},
		{"2026-03-11T20:23:35Z", 20, 2},
		{"2026-12-31T23:59:59Z", 23, 3},
		{"2026-07-04T12:00:00Z", 12, 5},
	}
	for _, tt := range tests {
		h, wd := parseHourAndWeekday(tt.input)
		if h != tt.wantHour {
			t.Errorf("parseHourAndWeekday(%q) hour = %d, want %d", tt.input, h, tt.wantHour)
		}
		if wd != tt.wantWeekday {
			t.Errorf("parseHourAndWeekday(%q) weekday = %d, want %d", tt.input, wd, tt.wantWeekday)
		}
	}
}

func TestParseHourAndWeekday_Invalid(t *testing.T) {
	h, wd := parseHourAndWeekday("invalid")
	if h != 0 || wd != 0 {
		t.Errorf("parseHourAndWeekday(invalid) = %d,%d, want 0,0", h, wd)
	}
}

func TestVectorize_LastTxWithValues(t *testing.T) {
	v := New(
		model.NormalizationConstants{MaxAmount: 10000, MaxInstallments: 12, AmountVsAvgRatio: 10, MaxMinutes: 1440, MaxKm: 1000, MaxTxCount24h: 20, MaxMerchantAvgAmount: 10000},
		model.MCCRisk{},
	)

	req := &model.FraudScoreRequest{
		Transaction: model.TransactionData{
			Amount:       200,
			Installments: 2,
			RequestedAt:  "2026-03-11T12:00:00Z",
		},
		Customer: model.CustomerData{
			AvgAmount:      100,
			TxCount24h:     5,
			KnownMerchants: nil,
		},
		Merchant: model.MerchantData{
			ID:        "MERC-001",
			MCC:       "5411",
			AvgAmount: 150,
		},
		Terminal: model.TerminalData{
			KmFromHome: 50,
		},
		LastTx: &model.LastTransactionData{
			Timestamp:     "2026-03-11T10:00:00Z",
			KmFromCurrent: 30,
		},
	}

	vec := v.Vectorize(req)

	if vec[5] < 0 {
		t.Errorf("minutes_since = %v, want >= 0 with LastTx", vec[5])
	}
	if vec[6] < 0 {
		t.Errorf("km_from_last = %v, want >= 0 with LastTx", vec[6])
	}
}

func TestClampFloat32_EdgeCases(t *testing.T) {
	tests := []struct {
		name string
		val  float32
		want float32
	}{
		{"exact max", 1.0, 1.0},
		{"exact min", 0.0, 0.0},
		{"very negative", -1e10, 0},
		{"very positive", 1e10, 1},
	}
	for _, tt := range tests {
		got := clampFloat32(tt.val)
		if got != tt.want {
			t.Errorf("clampFloat32(%v) = %v, want %v", tt.val, got, tt.want)
		}
	}
}
