package vectorizer

import (
	"testing"

	gojson "github.com/goccy/go-json"

	"github.com/fabianoflorentino/fraudctl/internal/model"
)

const testJSON1 = `{
	"id": "tx-123",
	"transaction": {"amount": 100.0, "installments": 1, "requested_at": "2026-03-11T14:30:00Z"},
	"customer": {"avg_amount": 100.0, "tx_count_24h": 5, "known_merchants": ["merch-a"]},
	"merchant": {"id": "merch-a", "mcc": "5812", "avg_amount": 150.0},
	"terminal": {"is_online": true, "card_present": false, "km_from_home": 42.5}
}`

const testJSON2 = `{
	"id": "tx-456",
	"transaction": {"amount": 500.0, "installments": 3, "requested_at": "2026-03-15T20:00:00Z"},
	"customer": {"avg_amount": 200.0, "tx_count_24h": 2, "known_merchants": ["m1", "m2"]},
	"merchant": {"id": "unknown", "mcc": "7995", "avg_amount": 1000.0},
	"terminal": {"is_online": false, "card_present": true, "km_from_home": 100.0},
	"last_transaction": {"timestamp": "2026-03-15T19:30:00Z", "km_from_current": 50.0}
}`

var testNorm = model.NormalizationConstants{
	MaxAmount:            10000,
	MaxInstallments:      12,
	AmountVsAvgRatio:     10,
	MaxMinutes:           1440,
	MaxKm:                1000,
	MaxTxCount24h:        20,
	MaxMerchantAvgAmount: 10000,
}

var testMCCRisk = model.MCCRisk{}

func init() {
	for i := range testMCCRisk {
		testMCCRisk[i] = 0.5
	}
	testMCCRisk[5812] = 0.3
	testMCCRisk[7995] = 0.85
}

func TestVectorizeJSON_MatchesOriginal(t *testing.T) {
	vec := New(testNorm, testMCCRisk)

	testCases := []string{testJSON1, testJSON2}

	for i, jsonStr := range testCases {
		var req model.FraudScoreRequest
		if err := gojson.Unmarshal([]byte(jsonStr), &req); err != nil {
			t.Fatalf("Case %d: gojson.Unmarshal failed: %v", i, err)
		}

		expected := vec.Vectorize(&req)
		actual, err := vec.VectorizeJSON([]byte(jsonStr))

		if err != nil {
			t.Fatalf("Case %d: VectorizeJSON failed: %v", i, err)
		}

		for d := 0; d < 14; d++ {
			if expected[d] != actual[d] {
				t.Errorf("Case %d: dim[%d] mismatch: expected %v, got %v", i, d, expected[d], actual[d])
			}
		}
	}
}

func TestVectorizeJSON_InvalidJSON(t *testing.T) {
	vec := New(testNorm, testMCCRisk)

	testCases := []struct {
		name    string
		json    string
		wantErr bool
	}{
		{"empty string", "", true},
		{"whitespace only", "   ", true},
		{"invalid json", "not json", true},
		{"array instead of object", "[1, 2, 3]", true},
		{"missing opening brace", `"transaction": {}}`, true},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := vec.VectorizeJSON([]byte(tc.json))
			if tc.wantErr && err == nil {
				t.Error("expected error, got nil")
			}
		})
	}
}

func TestVectorizeJSON_MissingFields(t *testing.T) {
	vec := New(testNorm, testMCCRisk)

	_, err := vec.VectorizeJSON([]byte(`{"id": "tx-123"}`))
	if err != nil {
		t.Errorf("unexpected error for missing fields: %v", err)
	}
}

func TestVectorizeJSON_NestedObjectSearch(t *testing.T) {
	vec := New(testNorm, testMCCRisk)

	jsonWithNested := `{
		"transaction": {"amount": 100, "installments": 1, "requested_at": "2026-03-11T10:00:00Z"},
		"customer": {"avg_amount": 100, "tx_count_24h": 1, "known_merchants": [], "nested": {"deep": 1}},
		"merchant": {"id": "m1", "mcc": "5411", "avg_amount": 50},
		"terminal": {"is_online": false, "card_present": true, "km_from_home": 10}
	}`

	_, err := vec.VectorizeJSON([]byte(jsonWithNested))
	if err != nil {
		t.Errorf("unexpected error with nested objects: %v", err)
	}
}

func TestVectorizeJSON_BoolParsing(t *testing.T) {
	vec := New(testNorm, testMCCRisk)

	testCases := []struct {
		name        string
		isOnline    bool
		cardPresent bool
	}{
		{"both true", true, true},
		{"both false", false, false},
		{"mixed 1", true, false},
		{"mixed 2", false, true},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			jsonStr := `{
				"transaction": {"amount": 100, "installments": 1, "requested_at": "2026-03-11T10:00:00Z"},
				"customer": {"avg_amount": 100, "tx_count_24h": 1, "known_merchants": []},
				"merchant": {"id": "m1", "mcc": "5411", "avg_amount": 50},
				"terminal": {"is_online": ` + boolToString(tc.isOnline) + `, "card_present": ` + boolToString(tc.cardPresent) + `, "km_from_home": 10}
			}`

			_, err := vec.VectorizeJSON([]byte(jsonStr))
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
		})
	}
}

func TestVectorizeJSON_EmptyKnownMerchants(t *testing.T) {
	vec := New(testNorm, testMCCRisk)

	jsonStr := `{
		"transaction": {"amount": 100, "installments": 1, "requested_at": "2026-03-11T10:00:00Z"},
		"customer": {"avg_amount": 100, "tx_count_24h": 1, "known_merchants": []},
		"merchant": {"id": "m1", "mcc": "5411", "avg_amount": 50},
		"terminal": {"is_online": false, "card_present": true, "km_from_home": 10}
	}`

	_, err := vec.VectorizeJSON([]byte(jsonStr))
	if err != nil {
		t.Errorf("unexpected error with empty known_merchants: %v", err)
	}
}

func TestVectorizeJSON_NumberParsing(t *testing.T) {
	vec := New(testNorm, testMCCRisk)

	jsonStr := `{
		"transaction": {"amount": 9999.99, "installments": 12, "requested_at": "2026-03-11T10:00:00Z"},
		"customer": {"avg_amount": 5000.5, "tx_count_24h": 20, "known_merchants": []},
		"merchant": {"id": "m1", "mcc": "5411", "avg_amount": 9999.99},
		"terminal": {"is_online": false, "card_present": true, "km_from_home": 999.999}
	}`

	_, err := vec.VectorizeJSON([]byte(jsonStr))
	if err != nil {
		t.Errorf("unexpected error with large numbers: %v", err)
	}
}

func boolToString(b bool) string {
	if b {
		return "true"
	}
	return "false"
}

func BenchmarkVectorizeJSON(b *testing.B) {
	vec := New(testNorm, testMCCRisk)
	data := []byte(testJSON2)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = vec.VectorizeJSON(data)
	}
}

func BenchmarkOriginalVectorize(b *testing.B) {
	vec := New(testNorm, testMCCRisk)
	data := []byte(testJSON2)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var req model.FraudScoreRequest
		_ = gojson.Unmarshal(data, &req)
		_ = vec.Vectorize(&req)
	}
}
