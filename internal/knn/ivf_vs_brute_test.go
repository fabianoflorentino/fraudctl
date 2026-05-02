package knn

import (
	"encoding/json"
	"os"
	"testing"

	"github.com/fabianoflorentino/fraudctl/internal/model"
)

type testEntry struct {
	Request struct {
		ID              string `json:"id"`
		Transaction     struct {
			Amount      float64 `json:"amount"`
			Installments int    `json:"installments"`
			RequestedAt string  `json:"requested_at"`
		} `json:"transaction"`
		Customer struct {
			AvgAmount      float64  `json:"avg_amount"`
			TxCount24h     int      `json:"tx_count_24h"`
			KnownMerchants []string `json:"known_merchants"`
		} `json:"customer"`
		Merchant struct {
			ID         string  `json:"id"`
			MCC        string  `json:"mcc"`
			AvgAmount  float64 `json:"avg_amount"`
		} `json:"merchant"`
		Terminal struct {
			IsOnline    bool    `json:"is_online"`
			CardPresent bool    `json:"card_present"`
			KmFromHome  float64 `json:"km_from_home"`
		} `json:"terminal"`
		LastTransaction *struct {
			Timestamp    string  `json:"timestamp"`
			KmFromCurrent float64 `json:"km_from_current"`
		} `json:"last_transaction"`
	} `json:"request"`
	ExpectedApproved bool `json:"expected_approved"`
}

func TestIVFVsBrute(t *testing.T) {
	ivfPath := "../../resources/ivf.bin"
	refsGz := "../../resources/references.json.gz"

	ivf, err := LoadIVF(ivfPath)
	if err != nil {
		t.Skipf("ivf.bin not found: %v", err)
	}
	ivf.SetNProbe(24)

	brute := NewBruteIndex()
	if err := brute.BuildFromGzip(refsGz, 3_000_000); err != nil {
		t.Skipf("references not found: %v", err)
	}

	f, err := os.Open("../../test-official-data.json")
	if err != nil {
		t.Skipf("test data not found: %v", err)
	}
	defer f.Close()

	var testData struct {
		Entries []testEntry `json:"entries"`
	}
	if err := json.NewDecoder(f).Decode(&testData); err != nil {
		t.Fatalf("decode test data: %v", err)
	}

	var mismatch int
	limit := 100
	if len(testData.Entries) < limit {
		limit = len(testData.Entries)
	}

	for i := 0; i < limit; i++ {
		entry := testData.Entries[i]

		req := model.FraudScoreRequest{
			ID: entry.Request.ID,
			Transaction: model.TransactionData{
				Amount:      entry.Request.Transaction.Amount,
				Installments: entry.Request.Transaction.Installments,
				RequestedAt: entry.Request.Transaction.RequestedAt,
			},
			Customer: model.CustomerData{
				AvgAmount:      entry.Request.Customer.AvgAmount,
				TxCount24h:     entry.Request.Customer.TxCount24h,
				KnownMerchants: entry.Request.Customer.KnownMerchants,
			},
			Merchant: model.MerchantData{
				ID:        entry.Request.Merchant.ID,
				MCC:       entry.Request.Merchant.MCC,
				AvgAmount: entry.Request.Merchant.AvgAmount,
			},
			Terminal: model.TerminalData{
				IsOnline:    entry.Request.Terminal.IsOnline,
				CardPresent: entry.Request.Terminal.CardPresent,
				KmFromHome:  entry.Request.Terminal.KmFromHome,
			},
		}
		if entry.Request.LastTransaction != nil {
			req.LastTx = &model.LastTransactionData{
				Timestamp:     entry.Request.LastTransaction.Timestamp,
				KmFromCurrent: entry.Request.LastTransaction.KmFromCurrent,
			}
		}

		vec := vectorizeTest(&req)

		ivfScore := ivf.Predict(vec, K)
		bruteScore := brute.Predict(vec, K)

		ivfApproved := ivfScore < 0.5
		bruteApproved := bruteScore < 0.5

		if ivfApproved != bruteApproved {
			mismatch++
			t.Logf("mismatch #%d: ivf=%.4f brute=%.4f expected_approved=%v",
				i, ivfScore, bruteScore, entry.ExpectedApproved)
		}
	}

	t.Logf("First %d entries: IVF vs Brute mismatches=%d", limit, mismatch)

	if mismatch > 10 {
		t.Errorf("Too many mismatches between IVF and brute force: %d", mismatch)
	}
}

func vectorizeTest(req *model.FraudScoreRequest) model.Vector14 {
	var vec model.Vector14
	vec[0] = clampF(float32(req.Transaction.Amount / 10000.0))
	vec[1] = clampF(float32(req.Transaction.Installments) / 12.0)
	if req.Customer.AvgAmount > 0 {
		vec[2] = clampF(float32(req.Transaction.Amount / req.Customer.AvgAmount / 10.0))
	}
	_ = vec[3]
	_ = vec[4]
	vec[5] = -1
	vec[6] = -1
	if req.LastTx != nil {
		vec[5] = clampF(float32(req.LastTx.KmFromCurrent / 1000.0))
		vec[6] = clampF(float32(req.LastTx.KmFromCurrent / 1000.0))
	}
	vec[7] = clampF(float32(req.Terminal.KmFromHome / 1000.0))
	vec[8] = clampF(float32(req.Customer.TxCount24h) / 20.0)
	if req.Terminal.IsOnline {
		vec[9] = 1
	}
	if req.Terminal.CardPresent {
		vec[10] = 1
	}
	known := false
	for _, m := range req.Customer.KnownMerchants {
		if m == req.Merchant.ID {
			known = true
			break
		}
	}
	if !known {
		vec[11] = 1
	}
	vec[12] = getMccRiskTest(req.Merchant.MCC)
	vec[13] = clampF(float32(req.Merchant.AvgAmount / 10000.0))
	return vec
}

func clampF(v float32) float32 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

func getMccRiskTest(mcc string) float32 {
	mccRisks := map[string]float32{
		"5411": 0.15, "5812": 0.30, "5912": 0.20, "5944": 0.45,
		"7801": 0.80, "7802": 0.75, "7995": 0.85, "4511": 0.35,
		"5311": 0.25, "5999": 0.50,
	}
	if r, ok := mccRisks[mcc]; ok {
		return r
	}
	return 0.0
}
