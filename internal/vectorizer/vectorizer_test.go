package vectorizer

import (
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
		checkDim func(t *testing.T, got Vector)
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

// TestClamp Tests clamping of float64 values to the range [0, 1].
func TestClamp(t *testing.T) {
	tests := []struct {
		name string
		val  float64
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

// TestGetPutVectorF32 Tests retrieval and return of float32 vectors from the pool.
func TestGetPutVectorF32(t *testing.T) {
	v := GetVectorF32()
	if v == nil {
		t.Error("GetVectorF32() returned nil")
	}
	PutVectorF32(v)
}

// TestPool_GetPut Tests basic pool operations for float64 vectors.
func TestPool_GetPut(t *testing.T) {
	vec := GetVector()
	if len(vec) != VectorSize {
		t.Errorf("GetVector() len = %d, want %d", len(vec), VectorSize)
	}

	PutVector(vec)
}

// TestPool_Concurrent Tests thread safety of the vector pool under concurrent access.
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

// TestVectorToF32 Tests conversion of Vector (float64) to VectorF32 (float32).
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

// TestParseTimestamp Tests parsing of RFC3339 timestamp strings.
func TestParseTimestamp(t *testing.T) {
	tests := []struct {
		name  string
		input string
		valid bool
	}{
		{"valid RFC3339", "2026-03-11T20:23:35Z", true},
		{"valid with timezone", "2026-03-11T15:23:35-05:00", true},
		{"invalid format", "2026-03-11", false},
		{"empty string", "", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ParseTimestamp(tt.input)
			if tt.valid && got.IsZero() {
				t.Errorf("ParseTimestamp(%q) = zero, want valid time", tt.input)
			}
		})
	}
}

// TestVectorClone Tests that Clone creates an independent copy of the vector.
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