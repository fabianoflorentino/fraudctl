package vectorizer

import (
	"fraudctl/internal/model"
	"testing"
)

func TestVectorizer_Vectorize(t *testing.T) {
	tests := []struct {
		name         string
		req          *model.FraudScoreRequest
		norm         model.NormalizationConstants
		mccRisk      model.MCCRisk
		wantLen      int
		checkDim    func(t *testing.T, got Vector)
	}{
		{
			name: "basic transaction",
			req: &model.FraudScoreRequest{
				ID: "tx-123",
				Transaction: model.TransactionData{
					Amount:      500,
					Installments: 3,
					RequestedAt: "2026-03-11T20:23:35Z",
				},
				Customer: model.CustomerData{
					AvgAmount:      500,
					TxCount24h:    2,
					KnownMerchants: []string{"MERC-001"},
				},
				Merchant: model.MerchantData{
					ID:          "MERC-001",
					MCC:         "5411",
					AvgAmount:   500,
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
			checkDim: func(t *testing.T, got Vector) {
				if len(got.Dimensions) != 14 {
					t.Errorf("Vectorize() len = %d, want 14", len(got.Dimensions))
				}
			},
		},
		{
			name: "null last transaction",
			req: &model.FraudScoreRequest{
				Transaction: model.TransactionData{
					Amount:      100,
					Installments: 1,
					RequestedAt: "2026-03-11T20:23:35Z",
				},
				Customer: model.CustomerData{
					AvgAmount:    100,
					TxCount24h:   1,
					KnownMerchants: []string{"MERC-001"},
				},
				Merchant: model.MerchantData{
					ID:          "MERC-001",
					MCC:         "5411",
					AvgAmount:   100,
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
			checkDim: func(t *testing.T, got Vector) {
				if got.Dimensions[5] != -1 || got.Dimensions[6] != -1 {
					t.Errorf("Vectorize() null last transaction dims 5,6 = %v,%v, want -1,-1", got.Dimensions[5], got.Dimensions[6])
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

func TestClamp(t *testing.T) {
	tests := []struct {
		name  string
		val   float64
		want float64
	}{
		{"within range", 0.5, 0.5},
		{"below zero", -0.5, 0},
		{"above one", 1.5, 1},
		{"exactly zero", 0, 0},
		{"exactly one", 1, 1},
		{"negative infinity", -100, 0},
		{"positive infinity", 100, 1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := clamp(tt.val); got != tt.want {
				t.Errorf("clamp(%v) = %v, want %v", tt.val, got, tt.want)
			}
		})
	}
}

func TestPool_GetPut(t *testing.T) {
	vec := GetVector()
	if len(vec) != VectorSize {
		t.Errorf("GetVector() len = %d, want %d", len(vec), VectorSize)
	}

	PutVector(vec)
}

func TestPool_Concurrent(t *testing.T) {
	done := make(chan bool, 10)

	for i := 0; i < 10; i++ {
		go func() {
			for j := 0; j < 1000; j++ {
				vec := GetVector()
				PutVector(vec)
			}
			done <- true
		}()
	}

	for i := 0; i < 10; i++ {
		<-done
	}
}

func TestVectorToF32(t *testing.T) {
	vec := Vector{
		Dimensions: []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.11, 0.12, 0.13, 0.14},
	}

	f32 := VectorToF32(vec)

	for i := 0; i < VectorSize; i++ {
		got := float64(f32.Dimensions[i])
		want := vec.Dimensions[i]
		diff := got - want
		if diff < 0 {
			diff = -diff
		}
		if diff > 0.001 {
			t.Errorf("VectorToF32() dim[%d] = %v, want ~%v", i, got, want)
		}
	}
}

func TestVectorClone(t *testing.T) {
	original := Vector{
		Dimensions: []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.11, 0.12, 0.13, 0.14},
	}

	clone := original.Clone()

	for i := 0; i < len(original.Dimensions); i++ {
		if clone.Dimensions[i] != original.Dimensions[i] {
			t.Errorf("Clone() dim[%d] = %v, want %v", i, clone.Dimensions[i], original.Dimensions[i])
		}
	}

	clone.Dimensions[0] = 999
	if original.Dimensions[0] == clone.Dimensions[0] {
		t.Error("Clone() should be independent copy")
	}
}