package knn

import (
	"github.com/fabianoflorentino/fraudctl/internal/model"
)

const (
	// K is the number of nearest neighbors to find.
	K = 5
)

// BruteForce implements KNN search using optimized brute-force.
// Thread-safe for concurrent Predict calls.
type BruteForce struct {
	vectors []model.Vector14
	labels  []bool
	n       int
}

// NewBruteForce creates a new brute-force KNN predictor.
func NewBruteForce() *BruteForce {
	return &BruteForce{}
}

// Build loads reference vectors into the predictor.
func (bf *BruteForce) Build(vectors []model.Vector14, labels []bool) {
	bf.vectors = vectors
	bf.labels = labels
	bf.n = len(vectors)
}

// Predict finds the k nearest neighbors and returns the fraud probability.
// Thread-safe for concurrent calls. Zero allocations.
func (bf *BruteForce) Predict(query model.Vector14, k int) float64 {
	n := bf.n
	if n == 0 {
		return 0.0
	}

	// Top-k slots
	var dists [5]float32
	var frauds [5]bool
	count := 0

	for i := 0; i < n; i++ {
		v := bf.vectors[i]
		isFraud := bf.labels[i]

		// Compute distance with early exit
		var dist float32
		if count < k {
			// No need for early exit when filling initial slots
			d0 := query[0] - v[0]
			dist = d0 * d0
			d1 := query[1] - v[1]
			dist += d1 * d1
			d2 := query[2] - v[2]
			dist += d2 * d2
			d3 := query[3] - v[3]
			dist += d3 * d3
			d4 := query[4] - v[4]
			dist += d4 * d4
			d5 := query[5] - v[5]
			dist += d5 * d5
			d6 := query[6] - v[6]
			dist += d6 * d6
			d7 := query[7] - v[7]
			dist += d7 * d7
			d8 := query[8] - v[8]
			dist += d8 * d8
			d9 := query[9] - v[9]
			dist += d9 * d9
			d10 := query[10] - v[10]
			dist += d10 * d10
			d11 := query[11] - v[11]
			dist += d11 * d11
			d12 := query[12] - v[12]
			dist += d12 * d12
			d13 := query[13] - v[13]
			dist += d13 * d13

			dists[count] = dist
			frauds[count] = isFraud
			count++
			continue
		}

		// Find max distance slot
		maxIdx := 0
		maxDist := dists[0]
		if dists[1] > maxDist {
			maxDist = dists[1]
			maxIdx = 1
		}
		if dists[2] > maxDist {
			maxDist = dists[2]
			maxIdx = 2
		}
		if dists[3] > maxDist {
			maxDist = dists[3]
			maxIdx = 3
		}
		if dists[4] > maxDist {
			maxIdx = 4
		}

		// Early exit: compute distance incrementally and bail if exceeds maxDist
		d0 := query[0] - v[0]
		dist = d0 * d0
		if dist >= maxDist {
			continue
		}
		d1 := query[1] - v[1]
		dist += d1 * d1
		if dist >= maxDist {
			continue
		}
		d2 := query[2] - v[2]
		dist += d2 * d2
		if dist >= maxDist {
			continue
		}
		d3 := query[3] - v[3]
		dist += d3 * d3
		if dist >= maxDist {
			continue
		}
		d4 := query[4] - v[4]
		dist += d4 * d4
		if dist >= maxDist {
			continue
		}
		d5 := query[5] - v[5]
		dist += d5 * d5
		if dist >= maxDist {
			continue
		}
		d6 := query[6] - v[6]
		dist += d6 * d6
		if dist >= maxDist {
			continue
		}
		d7 := query[7] - v[7]
		dist += d7 * d7
		if dist >= maxDist {
			continue
		}
		d8 := query[8] - v[8]
		dist += d8 * d8
		if dist >= maxDist {
			continue
		}
		d9 := query[9] - v[9]
		dist += d9 * d9
		if dist >= maxDist {
			continue
		}
		d10 := query[10] - v[10]
		dist += d10 * d10
		if dist >= maxDist {
			continue
		}
		d11 := query[11] - v[11]
		dist += d11 * d11
		if dist >= maxDist {
			continue
		}
		d12 := query[12] - v[12]
		dist += d12 * d12
		if dist >= maxDist {
			continue
		}
		d13 := query[13] - v[13]
		dist += d13 * d13

		// Replace if closer
		if dist < maxDist {
			dists[maxIdx] = dist
			frauds[maxIdx] = isFraud
		}
	}

	if count == 0 {
		return 0.0
	}

	fraudCount := 0
	for i := 0; i < count; i++ {
		if frauds[i] {
			fraudCount++
		}
	}

	return float64(fraudCount) / float64(count)
}

// Count returns the number of vectors in the predictor.
func (bf *BruteForce) Count() int {
	return bf.n
}
