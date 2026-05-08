package model

type RawRequest struct {
	Amount        float32
	Installments  uint8
	RequestedAt   []byte
	AvgAmount     float32
	TxCount24h    uint16
	MerchantID    []byte
	MerchantMCC   []byte
	MerchantAvg   float32
	IsOnline      bool
	CardPresent   bool
	KmFromHome    float32
	HasLastTx     bool
	LastTimestamp []byte
	LastKmFromCur float32
	KnownMerchant bool
}
