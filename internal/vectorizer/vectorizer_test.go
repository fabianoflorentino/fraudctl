package vectorizer

import (
	"math"
	"testing"

	"github.com/fabianoflorentino/fraudctl/internal/model"
)

// TestVectorizer_Vectorize Tests vector generation from fraud score requests with various configurations.
func TestVectorizer_Vectorize(t *testing.T) {
	tests := []struct {
		name     string
		req      *model.FraudScoreRequest
		norm     model.NormalizationConstants
		mccRisk  model.MCCRisk
		wantLen  int
		checkDim func(t *testing.T, got model.Vector14)
	}{
		{
			name: "basic transaction",
			req: &model.FraudScoreRequest{
				ID: "tx-123",
				Transaction: model.TransactionData{
					Amount:       500,
					Installments: 3,
					RequestedAt:  "2026-03-11T20:23:35Z",
				},
				Customer: model.CustomerData{
					AvgAmount:      500,
					TxCount24h:     2,
					KnownMerchants: []string{"MERC-001"},
				},
				Merchant: model.MerchantData{
					ID:        "MERC-001",
					MCC:       "5411",
					AvgAmount: 500,
				},
				Terminal: model.TerminalData{
					IsOnline:    false,
					CardPresent: true,
					KmFromHome:  10,
				},
			},
			norm:    model.NormalizationConstants{MaxAmount: 10000, MaxInstallments: 12, AmountVsAvgRatio: 10, MaxMinutes: 1440, MaxKm: 1000, MaxTxCount24h: 20, MaxMerchantAvgAmount: 10000},
			mccRisk: model.MCCRisk{"5411": 0.15},
			wantLen: 14,
			checkDim: func(t *testing.T, got model.Vector14) {
				if len(got) != 14 {
					t.Errorf("Vectorize() len = %d, want 14", len(got))
				}
			},
		},
		{
			name: "null last transaction",
			req: &model.FraudScoreRequest{
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
					IsOnline:    true,
					CardPresent: false,
					KmFromHome:  0,
				},
				LastTx: nil,
			},
			norm:    model.NormalizationConstants{MaxAmount: 10000, MaxInstallments: 12, AmountVsAvgRatio: 10, MaxMinutes: 1440, MaxKm: 1000, MaxTxCount24h: 20, MaxMerchantAvgAmount: 10000},
			mccRisk: model.MCCRisk{},
			wantLen: 14,
			checkDim: func(t *testing.T, got model.Vector14) {
				if got[5] != -1 || got[6] != -1 {
					t.Errorf("Vectorize() null last transaction dims 5,6 = %v,%v, want -1,-1", got[5], got[6])
				}
			},
		},
		{
			name: "zero avg_amount does not produce NaN",
			req: &model.FraudScoreRequest{
				Transaction: model.TransactionData{
					Amount:       200,
					Installments: 1,
					RequestedAt:  "2026-03-11T20:23:35Z",
				},
				Customer: model.CustomerData{
					AvgAmount:      0, // triggers division by zero without the guard
					TxCount24h:     1,
					KnownMerchants: []string{"MERC-001"},
				},
				Merchant: model.MerchantData{
					ID:        "MERC-001",
					MCC:       "5411",
					AvgAmount: 100,
				},
				Terminal: model.TerminalData{KmFromHome: 5},
			},
			norm:    model.NormalizationConstants{MaxAmount: 10000, MaxInstallments: 12, AmountVsAvgRatio: 10, MaxMinutes: 1440, MaxKm: 1000, MaxTxCount24h: 20, MaxMerchantAvgAmount: 10000},
			mccRisk: model.MCCRisk{},
			wantLen: 14,
			checkDim: func(t *testing.T, got model.Vector14) {
				if math.IsNaN(float64(got[2])) {
					t.Errorf("Vectorize() dim[2] = NaN when AvgAmount=0, want 0")
				}
				if got[2] != 0 {
					t.Errorf("Vectorize() dim[2] = %v when AvgAmount=0, want 0", got[2])
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			v := New(tt.norm, tt.mccRisk)
			got := v.Vectorize(tt.req)
			tt.checkDim(t, got)
		})
	}
}

// TestClampFloat32 Tests clamping of float32 values to the range [0, 1].
func TestClampFloat32(t *testing.T) {
	tests := []struct {
		name string
		val  float32
		want float32
	}{
		{"within range", 0.5, 0.5},
		{"below zero", -0.5, 0},
		{"above one", 1.5, 1},
		{"exactly zero", 0, 0},
		{"exactly one", 1, 1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := clampFloat32(tt.val); got != tt.want {
				t.Errorf("clampFloat32(%v) = %v, want %v", tt.val, got, tt.want)
			}
		})
	}
}

// TestParseHourAndWeekday tests the fast RFC3339 hour/weekday parser.
func TestParseHourAndWeekday(t *testing.T) {
	tests := []struct {
		name        string
		input       string
		wantHour    int
		wantWeekday int
		valid       bool
	}{
		{"valid UTC", "2026-03-11T20:23:35Z", 20, 3, true},  // 2026-03-11 is Wednesday=3
		{"midnight", "2026-01-01T00:00:00Z", 0, 4, true},    // 2026-01-01 is Thursday=4
		{"invalid format", "2026-03-11", 0, 0, false},
		{"empty string", "", 0, 0, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h, wd := parseHourAndWeekday(tt.input)
			if tt.valid && h != tt.wantHour {
				t.Errorf("parseHourAndWeekday(%q) hour = %d, want %d", tt.input, h, tt.wantHour)
			}
			if tt.valid && wd != tt.wantWeekday {
				t.Errorf("parseHourAndWeekday(%q) weekday = %d, want %d", tt.input, wd, tt.wantWeekday)
			}
		})
	}
}
