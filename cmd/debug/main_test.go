package main

import (
	"encoding/json"
	"os"
	"testing"
)

func TestDebugTestDataJSON(t *testing.T) {
	data := `{"entries": [
		{"request": {"id": "tx-1"}, "expected_approved": true}
	]}`

	var td struct {
		Entries []struct {
			Request          json.RawMessage `json:"request"`
			ExpectedApproved bool            `json:"expected_approved"`
		} `json:"entries"`
	}

	if err := json.Unmarshal([]byte(data), &td); err != nil {
		t.Fatalf("Unmarshal failed: %v", err)
	}

	if len(td.Entries) != 1 {
		t.Errorf("entries = %d, want 1", len(td.Entries))
	}

	if !td.Entries[0].ExpectedApproved {
		t.Error("ExpectedApproved should be true")
	}

	if td.Entries[0].Request == nil {
		t.Error("Request should not be nil")
	}
}

func TestDebugTestDataParse(t *testing.T) {
	data := `{"entries": [
		{"request": {"id": "tx-1", "amount": 100}, "expected_approved": true},
		{"request": {"id": "tx-2", "amount": 500}, "expected_approved": false}
	]}`

	var td struct {
		Entries []struct {
			Request          json.RawMessage `json:"request"`
			ExpectedApproved bool            `json:"expected_approved"`
		} `json:"entries"`
	}

	if err := json.Unmarshal([]byte(data), &td); err != nil {
		t.Fatalf("Unmarshal failed: %v", err)
	}

	if len(td.Entries) != 2 {
		t.Errorf("entries = %d, want 2", len(td.Entries))
	}
}

func TestDebug_FileNotFound(t *testing.T) {
	_, err := os.Open("/nonexistent/test-data.json")
	if err == nil {
		t.Fatal("expected error for nonexistent file")
	}
}

func TestDebug_InvalidJSON(t *testing.T) {
	tmpFile := t.TempDir() + "/invalid.json"
	os.WriteFile(tmpFile, []byte("{invalid}"), 0644)

	data, err := os.ReadFile(tmpFile)
	if err != nil {
		t.Fatalf("ReadFile failed: %v", err)
	}

	var td struct {
		Entries []struct {
			Request          json.RawMessage `json:"request"`
			ExpectedApproved bool            `json:"expected_approved"`
		} `json:"entries"`
	}

	if err := json.Unmarshal(data, &td); err == nil {
		t.Fatal("expected unmarshal error for invalid JSON")
	}
}

func TestDebug_EmptyFile(t *testing.T) {
	tmpFile := t.TempDir() + "/empty.json"
	os.WriteFile(tmpFile, []byte("{}"), 0644)

	data, err := os.ReadFile(tmpFile)
	if err != nil {
		t.Fatalf("ReadFile failed: %v", err)
	}

	var td struct {
		Entries []struct {
			Request          json.RawMessage `json:"request"`
			ExpectedApproved bool            `json:"expected_approved"`
		} `json:"entries"`
	}

	if err := json.Unmarshal(data, &td); err != nil {
		t.Fatalf("Unmarshal failed: %v", err)
	}
	if td.Entries != nil && len(td.Entries) != 0 {
		t.Errorf("entries = %d, want 0", len(td.Entries))
	}
}
