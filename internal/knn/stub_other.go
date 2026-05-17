//go:build !amd64

package knn

func scanClusterAVX2(_ []int16, _ []byte, _, _ int, _ [DIM]int16, _ *topK5) {}

func accumulateDotProductsAVX2(_ []float32, _ int, _ [DIM]float32, _ []float32) {}
