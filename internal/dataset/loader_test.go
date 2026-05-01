package dataset

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/fabianoflorentino/fraudctl/internal/model"
)

func TestLoader_LoadNormalization(t *testing.T) {
	tests := []struct {
		name    string
		path    string
		want    model.NormalizationConstants
		wantErr bool
		errMsg  string
	}{
		{
			name: "valid normalization file",
			path: "../../resources/normalization.json",
			want: model.NormalizationConstants{
				MaxAmount:            10000,
				MaxInstallments:      12,
				AmountVsAvgRatio:     10,
				MaxMinutes:           1440,
				MaxKm:                1000,
				MaxTxCount24h:        20,
				MaxMerchantAvgAmount: 10000,
			},
			wantErr: false,
		},
		{
			name:    "file not found",
			path:    "/nonexistent/path.json",
			want:    model.NormalizationConstants{},
			wantErr: true,
			errMsg:  "failed to read file",
		},
		{
			name:    "invalid JSON format",
			path:    "testdata/invalid.json",
			want:    model.NormalizationConstants{},
			wantErr: true,
			errMsg:  "failed to decode JSON",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			loader := NewLoader("")
			got, err := loader.LoadNormalization(tt.path)

			if (err != nil) != tt.wantErr {
				t.Errorf("LoadNormalization() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if tt.wantErr && err != nil {
				if tt.errMsg != "" && !contains(err.Error(), tt.errMsg) {
					t.Errorf("LoadNormalization() error message = %v, want contains %v", err, tt.errMsg)
				}
				return
			}

			if got.MaxAmount != tt.want.MaxAmount {
				t.Errorf("LoadNormalization() MaxAmount = %v, want %v", got.MaxAmount, tt.want.MaxAmount)
			}
			if got.MaxInstallments != tt.want.MaxInstallments {
				t.Errorf("LoadNormalization() MaxInstallments = %v, want %v", got.MaxInstallments, tt.want.MaxInstallments)
			}
			if got.AmountVsAvgRatio != tt.want.AmountVsAvgRatio {
				t.Errorf("LoadNormalization() AmountVsAvgRatio = %v, want %v", got.AmountVsAvgRatio, tt.want.AmountVsAvgRatio)
			}
			if got.MaxMinutes != tt.want.MaxMinutes {
				t.Errorf("LoadNormalization() MaxMinutes = %v, want %v", got.MaxMinutes, tt.want.MaxMinutes)
			}
			if got.MaxKm != tt.want.MaxKm {
				t.Errorf("LoadNormalization() MaxKm = %v, want %v", got.MaxKm, tt.want.MaxKm)
			}
			if got.MaxTxCount24h != tt.want.MaxTxCount24h {
				t.Errorf("LoadNormalization() MaxTxCount24h = %v, want %v", got.MaxTxCount24h, tt.want.MaxTxCount24h)
			}
			if got.MaxMerchantAvgAmount != tt.want.MaxMerchantAvgAmount {
				t.Errorf("LoadNormalization() MaxMerchantAvgAmount = %v, want %v", got.MaxMerchantAvgAmount, tt.want.MaxMerchantAvgAmount)
			}
		})
	}
}

func TestLoader_LoadMCCRisk(t *testing.T) {
	tests := []struct {
		name    string
		path    string
		wantErr bool
		check   func(t *testing.T, got model.MCCRisk)
	}{
		{
			name:    "valid mcc risk file",
			path:    "../../resources/mcc_risk.json",
			wantErr: false,
			check: func(t *testing.T, got model.MCCRisk) {
				if got.Get("5411") != 0.15 {
					t.Errorf("MCCRisk.Get(5411) = %v, want 0.15", got.Get("5411"))
				}
				if got.Get("7995") != 0.85 {
					t.Errorf("MCCRisk.Get(7995) = %v, want 0.85", got.Get("7995"))
				}
				if got.Get("UNKNOWN") != 0.5 {
					t.Errorf("MCCRisk.Get(UNKNOWN) = %v, want 0.5 (default)", got.Get("UNKNOWN"))
				}
			},
		},
		{
			name:    "file not found",
			path:    "/nonexistent/mcc.json",
			wantErr: true,
			check:   nil,
		},
		{
			name:    "invalid JSON",
			path:    "testdata/invalid.json",
			wantErr: true,
			check:   nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			loader := NewLoader("")
			got, err := loader.LoadMCCRisk(tt.path)

			if (err != nil) != tt.wantErr {
				t.Errorf("LoadMCCRisk() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr && tt.check != nil {
				tt.check(t, got)
			}
		})
	}
}

func TestLoader_LoadReferences(t *testing.T) {
	tests := []struct {
		name    string
		path    string
		wantErr bool
		check   func(t *testing.T, got int)
	}{
		{
			name:    "valid gz file",
			path:    "../../resources/references.json.gz",
			wantErr: false,
			check: func(t *testing.T, got int) {
				if got != 3000000 {
					t.Errorf("LoadReferences() count = %v, want 3000000", got)
				}
			},
		},
		{
			name:    "file not found",
			path:    "/nonexistent/references.json.gz",
			wantErr: true,
			check:   nil,
		},
		{
			name:    "invalid gzip",
			path:    "testdata/invalid.gz",
			wantErr: true,
			check:   nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			loader := NewLoader("")
			got, err := loader.LoadReferences(tt.path)

			if (err != nil) != tt.wantErr {
				t.Errorf("LoadReferences() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr && tt.check != nil {
				tt.check(t, len(got))
			}
		})
	}
}

func TestLoader_LoadAll(t *testing.T) {
	tests := []struct {
		name    string
		path    string
		wantErr bool
		check   func(t *testing.T, ds *Dataset)
	}{
		{
			name:    "full load",
			path:    "../../resources",
			wantErr: false,
			check: func(t *testing.T, ds *Dataset) {
				if ds.Count() != 3000000 {
					t.Errorf("Dataset.Count() = %v, want 3000000", ds.Count())
				}
				if ds.FraudCount() == 0 {
					t.Error("Dataset.FraudCount() should be > 0")
				}
				if ds.LegitCount() == 0 {
					t.Error("Dataset.LegitCount() should be > 0")
				}
				if ds.norm.MaxAmount != 10000 {
					t.Errorf("Dataset.norm.MaxAmount = %v, want 10000", ds.norm.MaxAmount)
				}
				if ds.mccRisk.Get("5411") != 0.15 {
					t.Errorf("Dataset.mccRisk.Get(5411) = %v, want 0.15", ds.mccRisk.Get("5411"))
				}
			},
		},
		{
			name:    "invalid path",
			path:    "/nonexistent",
			wantErr: true,
			check:   nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			loader := NewLoader(tt.path)
			ds, err := loader.LoadAll()

			if (err != nil) != tt.wantErr {
				t.Errorf("LoadAll() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr && tt.check != nil {
				tt.check(t, ds)
			}
		})
	}
}

func TestLoader_WithBasePath(t *testing.T) {
	tests := []struct {
		name     string
		basePath string
		resource string
		want     string
	}{
		{
			name:     "default path",
			basePath: "./resources/resources",
			resource: "",
			want:     "./resources/resources/normalization.json",
		},
		{
			name:     "custom path",
			basePath: "/data/fraudctl",
			resource: "",
			want:     "/data/fraudctl/normalization.json",
		},
		{
			name:     "empty base path",
			basePath: "",
			resource: "/absolute/path.json",
			want:     "/absolute/path.json",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			loader := NewLoader(tt.basePath)
			_, _ = loader.LoadNormalization(tt.resource)
		})
	}
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > 0 && containsHelper(s, substr))
}

func containsHelper(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

func TestCreateTestData(t *testing.T) {
	tmpDir := t.TempDir()

	normPath := filepath.Join(tmpDir, "normalization.json")
	normData := `{"max_amount": 5000, "max_installments": 6}`
	if err := os.WriteFile(normPath, []byte(normData), 0644); err != nil {
		t.Fatalf("Failed to create test file: %v", err)
	}

	loader := NewLoader("")
	norm, err := loader.LoadNormalization(normPath)
	if err != nil {
		t.Fatalf("LoadNormalization() error = %v", err)
	}

	if norm.MaxAmount != 5000 {
		t.Errorf("MaxAmount = %v, want 5000", norm.MaxAmount)
	}
	if norm.MaxInstallments != 6 {
		t.Errorf("MaxInstallments = %v, want 6", norm.MaxInstallments)
	}
}
