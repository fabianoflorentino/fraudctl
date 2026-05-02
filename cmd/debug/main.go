package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
)

func main() {
	testDataPath := os.Args[1]
	apiURL := "http://localhost:9999/fraud-score"

	f, err := os.Open(testDataPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "open: %v\n", err)
		os.Exit(1)
	}
	defer f.Close()

	data, err := io.ReadAll(f)
	if err != nil {
		fmt.Fprintf(os.Stderr, "read: %v\n", err)
		os.Exit(1)
	}

	var testData struct {
		Entries []struct {
			Request          json.RawMessage `json:"request"`
			ExpectedApproved bool            `json:"expected_approved"`
		} `json:"entries"`
	}
	if err := json.Unmarshal(data, &testData); err != nil {
		fmt.Fprintf(os.Stderr, "decode: %v\n", err)
		os.Exit(1)
	}

	scoreTotal := map[float64]int{}
	scoreApprovedCount := map[float64]int{}
	scoreExpectedApproved := map[float64]int{}
	scoreExpectedDenied := map[float64]int{}

	for i := 0; i < len(testData.Entries); i++ {
		entry := testData.Entries[i]
		expectedApproved := entry.ExpectedApproved

		resp, err := http.Post(apiURL, "application/json", bytes.NewReader(entry.Request))
		if err != nil {
			continue
		}

		var result struct {
			Approved   bool    `json:"approved"`
			FraudScore float64 `json:"fraud_score"`
		}
		json.NewDecoder(resp.Body).Decode(&result)
		resp.Body.Close()

		s := result.FraudScore
		scoreTotal[s]++
		if result.Approved {
			scoreApprovedCount[s]++
		}
		if expectedApproved {
			scoreExpectedApproved[s]++
		} else {
			scoreExpectedDenied[s]++
		}
	}

	fmt.Printf("Score Distribution (API threshold=0.5, approved if score < 0.5):\n")
	scores := []float64{0, 0.2, 0.4, 0.6, 0.8, 1.0}
	for _, s := range scores {
		total := scoreTotal[s]
		_ = scoreApprovedCount[s]
		expApproved := scoreExpectedApproved[s]
		expDenied := scoreExpectedDenied[s]
		// With threshold 0.5: 0, 0.2, 0.4 are approved; 0.6, 0.8, 1.0 are denied
		apiApproved := s < 0.5
		var fp, fn int
		if apiApproved {
			fn = expDenied // fraud that got approved
		} else {
			fp = expApproved // legit that got denied
		}
		fmt.Printf("  score=%.1f: total=%d api_approved=%v expected_approved=%d expected_denied=%d FP=%d FN=%d\n",
			s, total, apiApproved, expApproved, expDenied, fp, fn)
	}
}
