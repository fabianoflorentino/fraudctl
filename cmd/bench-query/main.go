package main

import (
	"fmt"
	"math/rand"
	"time"
	"github.com/fabianoflorentino/fraudctl/internal/knn"
	"github.com/fabianoflorentino/fraudctl/internal/model"
)

func main() {
	fmt.Println("Loading IVF index...")
	t0 := time.Now()
	idx, err := knn.LoadIVF("resources/ivf.bin")
	if err != nil { panic(err) }
	fmt.Printf("Loaded in %v. Count=%d FraudCount=%d\n", time.Since(t0), idx.Count(), idx.FraudCount())

	rng := rand.New(rand.NewSource(1))
	var q model.Vector14
	for i := range q { q[i] = rng.Float32() }

	for i := 0; i < 100; i++ { idx.Predict(q, 5) }

	N := 10000
	t1 := time.Now()
	for i := 0; i < N; i++ {
		for j := range q { q[j] = rng.Float32() }
		idx.Predict(q, 5)
	}
	elapsed := time.Since(t1)
	fmt.Printf("%d queries: total=%v  per-query=%v\n", N, elapsed, elapsed/time.Duration(N))
}
