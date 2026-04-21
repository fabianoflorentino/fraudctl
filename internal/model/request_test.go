package model

import (
	"testing"
	"time"
)

func TestTransactionData_RequestedAtTime(t *testing.T) {
	tests := []struct {
		name    string
		reqAt   string
		want    time.Time
		wantErr bool
	}{
		{
			name:    "valid RFC3339",
			reqAt:   "2026-03-11T20:23:35Z",
			want:    time.Date(2026, 3, 11, 20, 23, 35, 0, time.UTC),
			wantErr: false,
		},
		{
			name:    "invalid format",
			reqAt:   "2026-03-11",
			want:    time.Time{},
			wantErr: true,
		},
		{
			name:    "empty string",
			reqAt:   "",
			want:    time.Time{},
			wantErr: true,
		},
		{
			name:    "with timezone offset",
			reqAt:   "2026-03-11T20:23:35+03:00",
			want:    time.Date(2026, 3, 11, 20, 23, 35, 0, time.FixedZone("", 3*60*60)),
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			td := &TransactionData{RequestedAt: tt.reqAt}
			got, err := td.RequestedAtTime()

			if (err != nil) != tt.wantErr {
				t.Errorf("RequestedAtTime() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr && !got.Equal(tt.want) {
				t.Errorf("RequestedAtTime() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestLastTransactionData_TimestampTime(t *testing.T) {
	tests := []struct {
		name    string
		ts      string
		want    time.Time
		wantErr bool
	}{
		{
			name:    "valid RFC3339",
			ts:      "2026-03-11T14:58:35Z",
			want:    time.Date(2026, 3, 11, 14, 58, 35, 0, time.UTC),
			wantErr: false,
		},
		{
			name:    "invalid format",
			ts:     "invalid",
			want:   time.Time{},
			wantErr: true,
		},
		{
			name:    "empty string",
			ts:     "",
			want:   time.Time{},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			lt := &LastTransactionData{Timestamp: tt.ts}
			got, err := lt.TimestampTime()

			if (err != nil) != tt.wantErr {
				t.Errorf("TimestampTime() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr && !got.Equal(tt.want) {
				t.Errorf("TimestampTime() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMCCRisk_Get(t *testing.T) {
	risk := MCCRisk{
		"5411": 0.15,
		"7995": 0.85,
		"5912": 0.20,
	}

	tests := []struct {
		name string
		mcc  string
		want float64
	}{
		{"known mcc", "5411", 0.15},
		{"known mcc high risk", "7995", 0.85},
		{"unknown mcc should return default", "1234", 0.5},
		{"empty mcc returns default", "", 0.5},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := risk.Get(tt.mcc); got != tt.want {
				t.Errorf("MCCRisk.Get(%q) = %v, want %v", tt.mcc, got, tt.want)
			}
		})
	}
}
