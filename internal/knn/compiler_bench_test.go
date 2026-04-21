package knn

import "testing"

func BenchmarkCompilerTest(b *testing.B) {
	for i := 0; b.Loop(); i++ {
		_ = float64(i) * 0.5
	}
}
