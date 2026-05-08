package main

import (
	"fmt"
	"testing"

	gojson "github.com/goccy/go-json"

	"github.com/fabianoflorentino/fraudctl/internal/dataset"
	"github.com/fabianoflorentino/fraudctl/internal/model"
)

func BenchmarkFullHandler(b *testing.B) {
	payload := []byte(`{"id":"tx-1","transaction":{"amount":100.50,"installments":1,"requested_at":"2026-01-15T10:30:00Z"},"customer":{"avg_amount":50.0,"tx_count_24h":10,"known_merchants":["amazon","google"]},"merchant":{"id":"MERC-001","mcc":"1234","avg_amount":25.0},"terminal":{"is_online":true,"card_present":false,"km_from_home":5.2},"last_transaction":{"timestamp":"2026-01-15T09:30:00Z","km_from_current":3.1}}`)

	ds, err := dataset.LoadDefault("./../../resources")
	if err != nil {
		b.Fatal(err)
	}
	knn := ds.KNN()
	vec := ds.Vectorizer()

	b.ResetTimer()
	var req model.FraudScoreRequest
	for i := 0; i < b.N; i++ {
		req = model.FraudScoreRequest{}
		if err := gojson.Unmarshal(payload, &req); err != nil {
			b.Fatal(err)
		}
		query := vec.Vectorize(&req)
		nprobe := knn.NProbe()
		knn.PredictRaw(query, nprobe)
	}
	_ = fmt.Sprintf("%d", 0)
}
