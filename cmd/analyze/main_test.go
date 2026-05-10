package main

import (
	"encoding/json"
	"os"
	"testing"

	"github.com/fabianoflorentino/fraudctl/internal/model"
)

func TestEntryJSON(t *testing.T) {
	data := `{"request": {"id": "tx-1", "transaction": {"amount": 100, "installments": 1, "requested_at": "2026-03-11T10:00:00Z"}, "customer": {"avg_amount": 100, "tx_count_24h": 1}, "merchant": {"id": "MERC-001", "mcc": "5411", "avg_amount": 100}, "terminal": {"is_online": false, "card_present": true, "km_from_home": 10}}, "expected_approved": true}`

	var entry Entry
	if err := json.Unmarshal([]byte(data), &entry); err != nil {
		t.Fatalf("Unmarshal Entry failed: %v", err)
	}

	if entry.Request.ID != "tx-1" {
		t.Errorf("ID = %q, want tx-1", entry.Request.ID)
	}
	if !entry.ExpectedApproved {
		t.Error("ExpectedApproved should be true")
	}
	if entry.Request.Transaction.Amount != 100 {
		t.Errorf("Amount = %v, want 100", entry.Request.Transaction.Amount)
	}
}

func TestTestDataJSON(t *testing.T) {
	data := `{"entries": [
		{"request": {"id": "tx-1", "transaction": {"amount": 100, "installments": 1, "requested_at": "2026-03-11T10:00:00Z"}, "customer": {"avg_amount": 100, "tx_count_24h": 1}, "merchant": {"id": "MERC-001", "mcc": "5411", "avg_amount": 100}, "terminal": {"is_online": false, "card_present": true, "km_from_home": 10}}, "expected_approved": true},
		{"request": {"id": "tx-2", "transaction": {"amount": 500, "installments": 3, "requested_at": "2026-03-11T12:00:00Z"}, "customer": {"avg_amount": 50, "tx_count_24h": 5}, "merchant": {"id": "MERC-002", "mcc": "7995", "avg_amount": 200}, "terminal": {"is_online": true, "card_present": false, "km_from_home": 100}}, "expected_approved": false}
	]}`

	var td TestData
	if err := json.Unmarshal([]byte(data), &td); err != nil {
		t.Fatalf("Unmarshal TestData failed: %v", err)
	}

	if len(td.Entries) != 2 {
		t.Errorf("entries = %d, want 2", len(td.Entries))
	}
}

func TestEntry_RequestedAtTime(t *testing.T) {
	entry := Entry{
		Request: model.FraudScoreRequest{
			Transaction: model.TransactionData{
				Amount:       100,
				Installments: 1,
				RequestedAt:  "2026-03-11T10:00:00Z",
			},
		},
		ExpectedApproved: true,
	}

	ts, err := entry.Request.Transaction.RequestedAtTime()
	if err != nil {
		t.Fatalf("RequestedAtTime failed: %v", err)
	}
	if ts.Year() != 2026 {
		t.Errorf("year = %d, want 2026", ts.Year())
	}
	if ts.Month() != 3 {
		t.Errorf("month = %d, want 3", ts.Month())
	}
}

func TestEntry_RequestedAtTime_Invalid(t *testing.T) {
	entry := Entry{
		Request: model.FraudScoreRequest{
			Transaction: model.TransactionData{
				RequestedAt: "invalid-date",
			},
		},
	}

	_, err := entry.Request.Transaction.RequestedAtTime()
	if err == nil {
		t.Fatal("expected error for invalid date")
	}
}

func TestWriteAndReadTestData(t *testing.T) {
	td := TestData{
		Entries: []Entry{
			{
				Request: model.FraudScoreRequest{
					ID: "tx-1",
					Transaction: model.TransactionData{
						Amount:       100,
						Installments: 1,
						RequestedAt:  "2026-03-11T10:00:00Z",
					},
					Customer: model.CustomerData{
						AvgAmount:  100,
						TxCount24h: 1,
					},
					Merchant: model.MerchantData{
						ID:  "MERC-001",
						MCC: "5411",
					},
					Terminal: model.TerminalData{
						KmFromHome: 10,
					},
				},
				ExpectedApproved: true,
			},
		},
	}

	tmpFile := t.TempDir() + "/testdata.json"
	data, err := json.Marshal(td)
	if err != nil {
		t.Fatalf("Marshal failed: %v", err)
	}
	if err := os.WriteFile(tmpFile, data, 0644); err != nil {
		t.Fatalf("WriteFile failed: %v", err)
	}

	readBack, err := os.ReadFile(tmpFile)
	if err != nil {
		t.Fatalf("ReadFile failed: %v", err)
	}

	var decoded TestData
	if err := json.Unmarshal(readBack, &decoded); err != nil {
		t.Fatalf("Unmarshal failed: %v", err)
	}

	if len(decoded.Entries) != 1 {
		t.Errorf("entries = %d, want 1", len(decoded.Entries))
	}
	if decoded.Entries[0].Request.ID != "tx-1" {
		t.Errorf("ID = %q, want tx-1", decoded.Entries[0].Request.ID)
	}
}
