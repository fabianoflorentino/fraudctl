package main

import (
	"fmt"
	"testing"

	gojson "github.com/goccy/go-json"

	"github.com/fabianoflorentino/fraudctl/internal/dataset"
	"github.com/fabianoflorentino/fraudctl/internal/model"
)

func BenchmarkHandler(b *testing.B) {
	payload := []byte(`{"id":"tx-1","merchant":"amazon","mcc":1234,"amount":100.50,"timestamp":"2026-01-15T10:30:00Z","terminal":{"id":"term-1","km_from_home":5.2,"is_online":true,"card_present":false},"customer":{"id":"cust-1","avg_amount":50.0,"tx_count_24h":10,"known_merchants":["amazon","google"]},"last_tx":{"timestamp":"2026-01-15T09:30:00Z","km_from_current":3.1}}`)

	ds, err := dataset.LoadVectorizerOnly("./../../resources")
	if err != nil {
		b.Fatal(err)
	}
	vec := ds.Vectorizer()

	b.ResetTimer()
	var req model.FraudScoreRequest
	for i := 0; i < b.N; i++ {
		req = model.FraudScoreRequest{}
		if err := gojson.Unmarshal(payload, &req); err != nil {
			b.Fatal(err)
		}
		vec.Vectorize(&req)
	}
}

func BenchmarkIVFSearch(b *testing.B) {
	ds, err := dataset.LoadDefault("./../../resources")
	if err != nil {
		b.Fatal(err)
	}
	knn := ds.KNN()
	vec := ds.Vectorizer()

	payload := []byte(`{"id":"tx-1","merchant":"amazon","mcc":1234,"amount":100.50,"timestamp":"2026-01-15T10:30:00Z","terminal":{"id":"term-1","km_from_home":5.2,"is_online":true,"card_present":false},"customer":{"id":"cust-1","avg_amount":50.0,"tx_count_24h":10,"known_merchants":["amazon","google"]},"last_tx":{"timestamp":"2026-01-15T09:30:00Z","km_from_current":3.1}}`)

	var req model.FraudScoreRequest
	gojson.Unmarshal(payload, &req)
	query := vec.Vectorize(&req)
	nprobe := knn.NProbe()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		knn.PredictRaw(query, nprobe)
	}
	_ = fmt.Sprintf("%d", nprobe)
}
