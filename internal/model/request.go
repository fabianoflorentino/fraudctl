package model

import "time"

type FraudScoreRequest struct {
	ID          string              `json:"id"`
	Transaction TransactionData      `json:"transaction"`
	Customer    CustomerData         `json:"customer"`
	Merchant    MerchantData         `json:"merchant"`
	Terminal    TerminalData         `json:"terminal"`
	LastTx      *LastTransactionData `json:"last_transaction"`
}

type TransactionData struct {
	Amount      float64   `json:"amount"`
	Installments int     `json:"installments"`
	RequestedAt string   `json:"requested_at"`
}

type CustomerData struct {
	AvgAmount    float64   `json:"avg_amount"`
	TxCount24h  int       `json:"tx_count_24h"`
	KnownMerchants []string `json:"known_merchants"`
}

type MerchantData struct {
	ID          string  `json:"id"`
	MCC         string  `json:"mcc"`
	AvgAmount   float64 `json:"avg_amount"`
}

type TerminalData struct {
	IsOnline    bool    `json:"is_online"`
	CardPresent bool   `json:"card_present"`
	KmFromHome float64 `json:"km_from_home"`
}

type LastTransactionData struct {
	Timestamp      string  `json:"timestamp"`
	KmFromCurrent float64 `json:"km_from_current"`
}

type FraudScoreResponse struct {
	Approved   bool    `json:"approved"`
	FraudScore float64 `json:"fraud_score"`
}

func (t *TransactionData) RequestedAtTime() (time.Time, error) {
	return time.Parse(time.RFC3339, t.RequestedAt)
}

func (l *LastTransactionData) TimestampTime() (time.Time, error) {
	return time.Parse(time.RFC3339, l.Timestamp)
}