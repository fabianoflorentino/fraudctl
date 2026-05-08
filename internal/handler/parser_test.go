package handler

import (
	"testing"

	gojson "github.com/goccy/go-json"

	"github.com/fabianoflorentino/fraudctl/internal/model"
)

func abs(v float64) float64 {
	if v < 0 {
		return -v
	}
	return v
}

const testPayload = `{"id":"tx-123","transaction":{"amount":100.50,"installments":2,"requested_at":"2026-03-11T20:23:35Z"},"customer":{"avg_amount":200.0,"tx_count_24h":5,"known_merchants":["merchant-1","merchant-2"]},"merchant":{"id":"merchant-1","mcc":"5411","avg_amount":50.0},"terminal":{"is_online":true,"card_present":false,"km_from_home":10.5},"last_transaction":{"timestamp":"2026-03-11T19:00:00Z","km_from_current":5.2}}`

const testPayloadNoLastTx = `{"id":"tx-456","transaction":{"amount":50.0,"installments":1,"requested_at":"2026-03-11T10:00:00Z"},"customer":{"avg_amount":100.0,"tx_count_24h":2,"known_merchants":[]},"merchant":{"id":"unknown-merchant","mcc":"1234","avg_amount":25.0},"terminal":{"is_online":false,"card_present":true,"km_from_home":0.5}}`

const testPayloadLastTxNull = `{"id":"tx-789","transaction":{"amount":50.0,"installments":1,"requested_at":"2026-03-11T10:00:00Z"},"customer":{"avg_amount":100.0,"tx_count_24h":2,"known_merchants":[]},"merchant":{"id":"unknown-merchant","mcc":"1234","avg_amount":25.0},"terminal":{"is_online":false,"card_present":true,"km_from_home":0.5},"last_transaction":null}`

func TestParseFraudRequest(t *testing.T) {
	r := parseFraudRequest([]byte(testPayload))

	if r.Amount != 100.50 {
		t.Errorf("Amount = %v, want 100.50", r.Amount)
	}
	if r.Installments != 2 {
		t.Errorf("Installments = %v, want 2", r.Installments)
	}
	if string(r.RequestedAt) != "2026-03-11T20:23:35Z" {
		t.Errorf("RequestedAt = %q, want 2026-03-11T20:23:35Z", string(r.RequestedAt))
	}
	if r.AvgAmount != 200.0 {
		t.Errorf("AvgAmount = %v, want 200.0", r.AvgAmount)
	}
	if r.TxCount24h != 5 {
		t.Errorf("TxCount24h = %v, want 5", r.TxCount24h)
	}
	if string(r.MerchantID) != "merchant-1" {
		t.Errorf("MerchantID = %q, want merchant-1", string(r.MerchantID))
	}
	if string(r.MerchantMCC) != "5411" {
		t.Errorf("MerchantMCC = %q, want 5411", string(r.MerchantMCC))
	}
	if r.MerchantAvg != 50.0 {
		t.Errorf("MerchantAvg = %v, want 50.0", r.MerchantAvg)
	}
	if r.IsOnline != true {
		t.Errorf("IsOnline = %v, want true", r.IsOnline)
	}
	if r.CardPresent != false {
		t.Errorf("CardPresent = %v, want false", r.CardPresent)
	}
	if r.KmFromHome != 10.5 {
		t.Errorf("KmFromHome = %v, want 10.5", r.KmFromHome)
	}
	if !r.HasLastTx {
		t.Errorf("HasLastTx = false, want true")
	}
	if string(r.LastTimestamp) != "2026-03-11T19:00:00Z" {
		t.Errorf("LastTimestamp = %q, want 2026-03-11T19:00:00Z", string(r.LastTimestamp))
	}
	if r.LastKmFromCur != 5.2 {
		t.Errorf("LastKmFromCur = %v, want 5.2", r.LastKmFromCur)
	}
	if !r.KnownMerchant {
		t.Errorf("KnownMerchant = false, want true (merchant-1 is in known_merchants)")
	}
}

func TestParseFraudRequest_NoLastTx(t *testing.T) {
	r := parseFraudRequest([]byte(testPayloadNoLastTx))

	if r.Amount != 50.0 {
		t.Errorf("Amount = %v, want 50.0", r.Amount)
	}
	if r.Installments != 1 {
		t.Errorf("Installments = %v, want 1", r.Installments)
	}
	if r.AvgAmount != 100.0 {
		t.Errorf("AvgAmount = %v, want 100.0", r.AvgAmount)
	}
	if string(r.MerchantID) != "unknown-merchant" {
		t.Errorf("MerchantID = %q, want unknown-merchant", string(r.MerchantID))
	}
	if r.IsOnline != false {
		t.Errorf("IsOnline = %v, want false", r.IsOnline)
	}
	if r.CardPresent != true {
		t.Errorf("CardPresent = %v, want true", r.CardPresent)
	}
	if r.HasLastTx {
		t.Errorf("HasLastTx = true, want false")
	}
	if r.KnownMerchant {
		t.Errorf("KnownMerchant = true, want false (unknown-merchant not in empty list)")
	}
}

func TestParseFraudRequest_LastTxNull(t *testing.T) {
	r := parseFraudRequest([]byte(testPayloadLastTxNull))

	if r.Amount != 50.0 {
		t.Errorf("Amount = %v, want 50.0", r.Amount)
	}
	if r.HasLastTx {
		t.Errorf("HasLastTx = true, want false (last_transaction is null)")
	}
}

func TestParseFraudRequest_VsGojson(t *testing.T) {
	var gojsonReq model.FraudScoreRequest
	if err := gojson.Unmarshal([]byte(testPayload), &gojsonReq); err != nil {
		t.Fatalf("gojson unmarshal: %v", err)
	}

	r := parseFraudRequest([]byte(testPayload))

	if abs(float64(r.Amount)-gojsonReq.Transaction.Amount) > 0.001 {
		t.Errorf("Amount: got %v, want %v", r.Amount, gojsonReq.Transaction.Amount)
	}
	if int(r.Installments) != gojsonReq.Transaction.Installments {
		t.Errorf("Installments: got %v, want %v", r.Installments, gojsonReq.Transaction.Installments)
	}
	if string(r.RequestedAt) != gojsonReq.Transaction.RequestedAt {
		t.Errorf("RequestedAt: got %q, want %q", string(r.RequestedAt), gojsonReq.Transaction.RequestedAt)
	}
	if abs(float64(r.AvgAmount)-gojsonReq.Customer.AvgAmount) > 0.001 {
		t.Errorf("AvgAmount: got %v, want %v", r.AvgAmount, gojsonReq.Customer.AvgAmount)
	}
	if int(r.TxCount24h) != gojsonReq.Customer.TxCount24h {
		t.Errorf("TxCount24h: got %v, want %v", r.TxCount24h, gojsonReq.Customer.TxCount24h)
	}
	if string(r.MerchantID) != gojsonReq.Merchant.ID {
		t.Errorf("MerchantID: got %q, want %q", string(r.MerchantID), gojsonReq.Merchant.ID)
	}
	if string(r.MerchantMCC) != gojsonReq.Merchant.MCC {
		t.Errorf("MerchantMCC: got %q, want %q", string(r.MerchantMCC), gojsonReq.Merchant.MCC)
	}
	if abs(float64(r.MerchantAvg)-gojsonReq.Merchant.AvgAmount) > 0.001 {
		t.Errorf("MerchantAvg: got %v, want %v", r.MerchantAvg, gojsonReq.Merchant.AvgAmount)
	}
	if r.IsOnline != gojsonReq.Terminal.IsOnline {
		t.Errorf("IsOnline: got %v, want %v", r.IsOnline, gojsonReq.Terminal.IsOnline)
	}
	if r.CardPresent != gojsonReq.Terminal.CardPresent {
		t.Errorf("CardPresent: got %v, want %v", r.CardPresent, gojsonReq.Terminal.CardPresent)
	}
	if abs(float64(r.KmFromHome)-gojsonReq.Terminal.KmFromHome) > 0.001 {
		t.Errorf("KmFromHome: got %v, want %v", r.KmFromHome, gojsonReq.Terminal.KmFromHome)
	}

	// Check last_transaction
	if gojsonReq.LastTx != nil {
		if !r.HasLastTx {
			t.Errorf("HasLastTx = false, want true")
		}
		if string(r.LastTimestamp) != gojsonReq.LastTx.Timestamp {
			t.Errorf("LastTimestamp: got %q, want %q", string(r.LastTimestamp), gojsonReq.LastTx.Timestamp)
		}
			got := float64(r.LastKmFromCur)
		want := gojsonReq.LastTx.KmFromCurrent
		diff := got - want
		if diff < 0 {
			diff = -diff
		}
		if diff > 0.001 {
			t.Errorf("LastKmFromCur: got %v, want %v", got, want)
		}
	} else if r.HasLastTx {
		t.Errorf("HasLastTx = true, but gojson has LastTx = nil")
	}

	// Check known_merchant
	known := false
	for _, m := range gojsonReq.Customer.KnownMerchants {
		if m == gojsonReq.Merchant.ID {
			known = true
			break
		}
	}
	if r.KnownMerchant != known {
		t.Errorf("KnownMerchant: got %v, want %v", r.KnownMerchant, known)
	}
}
