// Package model provides data structures for the fraud detection API.
//
// This package defines the request and response types used in the fraud-score
// endpoint, following the schema defined in the Rinha 2026 documentation.
//
// # Request/Response Types
//
// The API uses JSON for request and response payloads. All timestamp fields
// must be in RFC3339 format (e.g., "2026-03-11T20:23:35Z").
//
// # Example
//
//	req := &model.FraudScoreRequest{
//	    ID: "tx-123",
//	    Transaction: model.TransactionData{
//	        Amount:      500.00,
//	        Installments: 3,
//	        RequestedAt:  "2026-03-11T20:23:35Z",
//	    },
//	    // ...
//	}
package model

import "time"

// FraudScoreRequest represents a transaction request for fraud detection.
// It contains all the information needed to evaluate a transaction.
//
// Fields:
//   - ID: Unique transaction identifier
//   - Transaction: Transaction details (amount, installments, timestamp)
//   - Customer: Cardholder information (average spending, transaction count)
//   - Merchant: Merchant details (ID, MCC, average ticket)
//   - Terminal: Terminal information (online/offline, card present, location)
//   - LastTx: Previous transaction data (optional, may be nil)
type FraudScoreRequest struct {
	ID          string               `json:"id"`
	Transaction TransactionData      `json:"transaction"`
	Customer    CustomerData         `json:"customer"`
	Merchant    MerchantData         `json:"merchant"`
	Terminal    TerminalData         `json:"terminal"`
	LastTx      *LastTransactionData `json:"last_transaction"`
}

// TransactionData contains the transaction details.
type TransactionData struct {
	Amount       float64 `json:"amount"`
	Installments int     `json:"installments"`
	RequestedAt  string  `json:"requested_at"`
}

// CustomerData contains cardholder information for fraud analysis.
type CustomerData struct {
	AvgAmount      float64  `json:"avg_amount"`
	TxCount24h     int      `json:"tx_count_24h"`
	KnownMerchants []string `json:"known_merchants"`
}

// MerchantData contains merchant identification and category.
type MerchantData struct {
	ID        string  `json:"id"`
	MCC       string  `json:"mcc"`
	AvgAmount float64 `json:"avg_amount"`
}

// TerminalData contains terminal/capture device information.
type TerminalData struct {
	IsOnline    bool    `json:"is_online"`
	CardPresent bool    `json:"card_present"`
	KmFromHome  float64 `json:"km_from_home"`
}

// LastTransactionData contains the previous transaction for velocity checks.
// This field is optional and may be nil.
type LastTransactionData struct {
	Timestamp     string  `json:"timestamp"`
	KmFromCurrent float64 `json:"km_from_current"`
}

// FraudScoreResponse represents the API response with fraud detection result.
//
// The fraud_score is a value between 0.0 and 1.0, where:
//   - 0.0 means all K nearest neighbors are legitimate
//   - 1.0 means all K nearest neighbors are fraudulent
//
// The approved field indicates whether the transaction should be accepted:
//   - true: Transaction approved (fraud_score < 0.6)
//   - false: Transaction denied (fraud_score >= 0.6)
type FraudScoreResponse struct {
	Approved   bool    `json:"approved"`
	FraudScore float64 `json:"fraud_score"`
}

// RequestedAtTime parses the transaction timestamp from RFC3339 format.
// Returns an error if the timestamp is invalid.
func (t *TransactionData) RequestedAtTime() (time.Time, error) {
	return time.Parse(time.RFC3339, t.RequestedAt)
}

// TimestampTime parses the last transaction timestamp from RFC3339 format.
// Returns an error if the timestamp is invalid.
func (l *LastTransactionData) TimestampTime() (time.Time, error) {
	return time.Parse(time.RFC3339, l.Timestamp)
}
