// Package knn provides K-Nearest Neighbors prediction for fraud detection.
//
// This implementation is optimized for minimal latency:
// - Zero allocations on the hot path
// - Sequential memory access (cache-friendly)
// - No goroutines (goroutine overhead)
// - Uses sync.Pool for result buffers
package knn

import (
	"fraudctl/internal/vectorizer"
	"sync"
)

// K is the number of nearest neighbors to consider.
const K = 5

// Dimensions is the number of vector dimensions.
const Dimensions = 14

// Reference represents a labeled reference vector for KNN.
type Reference struct {
	Vector  vectorizer.Vector
	IsFraud bool
}

// Dataset holds the reference vectors and provides prediction.
type Dataset struct {
	references []Reference
	vectors    [][]float64
	fraudFlags []bool
	pool       *vectorPool
}

// vectorPool manages pre-allocated buffers to avoid allocations.
type vectorPool struct {
	pool sync.Pool
}

// newVectorPool creates a pool for neighbor result slices.
func newVectorPool() *vectorPool {
	return &vectorPool{
		pool: sync.Pool{
			New: func() any {
				// Pre-allocate with length K and some capacity
				return make([]neighbor, 0, K)
			},
		},
	}
}

// neighbor represents a single nearest neighbor result.
type neighbor struct {
	Index    int
	Distance float64
	IsFraud  bool
}

// NewDataset creates a new KNN Dataset from references.
func NewDataset(references []Reference, _ int) *Dataset {
	vectors := make([][]float64, len(references))
	fraudFlags := make([]bool, len(references))

	for i, ref := range references {
		vectors[i] = ref.Vector.Dimensions
		fraudFlags[i] = ref.IsFraud
	}

	return &Dataset{
		references: references,
		vectors:    vectors,
		fraudFlags: fraudFlags,
		pool:       newVectorPool(),
	}
}

// Predict performs KNN prediction on the query vector.
// Optimized for minimal latency with zero allocations.
func (d *Dataset) Predict(query []float64) (fraudScore float64, approved bool) {
	// Get buffer from pool (zero allocation)
	buf := d.pool.get()

	// Ensure buffer has capacity for K neighbors
	if cap(buf) < K {
		buf = make([]neighbor, K)
	} else {
		// Reset slice to have length K
		buf = buf[:K]
	}

	// Put back in pool when done
	defer d.pool.put(buf)

	// Find K nearest neighbors using linear scan with heap
	k := 0
	for i := range d.vectors {
		dist := euclideanDistanceSquared(query, d.vectors[i])
		if k < K {
			buf[k] = neighbor{
				Index:    i,
				Distance: dist,
				IsFraud:  d.fraudFlags[i],
			}
			k++
			continue
		}

		// Find max distance in current heap
		maxDist := buf[0].Distance
		maxIdx := 0
		for j := 1; j < K; j++ {
			if buf[j].Distance > maxDist {
				maxDist = buf[j].Distance
				maxIdx = j
			}
		}

		// Replace if closer
		if dist < maxDist {
			buf[maxIdx] = neighbor{
				Index:    i,
				Distance: dist,
				IsFraud:  d.fraudFlags[i],
			}
		}
	}

	// Count fraud neighbors
	fraudCount := 0
	for i := 0; i < k; i++ {
		if buf[i].IsFraud {
			fraudCount++
		}
	}

	// Handle case with fewer than K references
	if k < K {
		fraudScore = float64(fraudCount) / float64(k)
	} else {
		fraudScore = float64(fraudCount) / float64(K)
	}
	approved = fraudScore < 0.6

	return fraudScore, approved
}

// euclideanDistanceSquared computes squared Euclidean distance.
// Using squared distance avoids sqrt, which is faster.
func euclideanDistanceSquared(a, b []float64) float64 {
	var sum float64
	for i := 0; i < len(a); i++ {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return sum
}

// get retrieves a buffer from the pool.
func (p *vectorPool) get() []neighbor {
	return p.pool.Get().([]neighbor)[:0]
}

// put returns a buffer to the pool.
//
//nolint:staticcheck // SA6002: slice is intentional for pool reuse
func (p *vectorPool) put(buf []neighbor) {
	p.pool.Put(buf)
}
