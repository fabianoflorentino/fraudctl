// Package knn provides K-Nearest Neighbors prediction for fraud detection.
//
// The package implements a simple brute-force KNN algorithm using Euclidean distance.
// It finds the K closest reference transactions and uses majority voting
// to determine if a transaction is fraudulent.
//
// # Algorithm
//
// Given a query vector (14-dimensional), the algorithm:
//  1. Computes Euclidean distance to all reference vectors
//  2. Finds the K nearest neighbors
//  3. Counts fraud vs legit neighbors
//  4. Computes fraud_score = fraud_count / K
//  5. Returns approved = (fraud_score < 0.6)
//
// # Performance
//
// For 100k reference vectors:
//   - Single worker: ~5ms
//   - Multi-worker: slower due to goroutine overhead
//
// # Usage
//
//	refs := []knn.Reference{
//	    {Vector: vectorizer.Vector{Dimensions: []float64{0.1, ...}}, IsFraud: true},
//	    // ...
//	}
//	k := knn.NewDataset(refs, 1)
//	score, approved := k.Predict(query)
package knn

import (
	"fraudctl/internal/vectorizer"
	"sync"
)

// K is the number of nearest neighbors to consider.
const K = 5

// Reference represents a labeled reference vector for KNN.
type Reference struct {
	Vector vectorizer.Vector
	IsFraud bool
}

// Dataset holds the reference vectors and provides prediction.
type Dataset struct {
	references []Reference
	vectors    [][]float64
	fraudFlags []bool
	numWorkers int
}

// NewDataset creates a new KNN Dataset from references.
// The numWorkers parameter specifies the number of goroutines
// for parallel distance computation. Use 1 for optimal performance
// (goroutine overhead makes more workers slower).
func NewDataset(references []Reference, numWorkers int) *Dataset {
	if numWorkers <= 0 {
		numWorkers = 1
	}

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
		numWorkers: numWorkers,
	}
}

// Predict performs KNN prediction on the query vector.
// Returns the fraud score (0.0 to 1.0) and whether the
// transaction is approved.
//
// The fraud score is the fraction of neighbors marked as fraud.
// Transactions with fraud_score >= 0.6 are denied.
func (d *Dataset) Predict(query []float64) (fraudScore float64, approved bool) {
	neighbors := d.findKNearest(query)

	fraudCount := 0
	for _, n := range neighbors {
		if n.IsFraud {
			fraudCount++
		}
	}

	fraudScore = float64(fraudCount) / float64(K)
	approved = fraudScore < 0.6

	return fraudScore, approved
}

// neighbor represents a single nearest neighbor result.
type neighbor struct {
	Index    int
	Distance float64
	IsFraud  bool
}

// distanceResult is used internally for collecting parallel results.
type distanceResult struct {
	index    int
	distance float64
	isFraud bool
}

// findKNearest finds the K nearest neighbors using parallel brute-force.
func (d *Dataset) findKNearest(query []float64) []neighbor {
	chunkSize := (len(d.vectors) + d.numWorkers - 1) / d.numWorkers

	results := make(chan distanceResult, len(d.vectors))
	var wg sync.WaitGroup

	for w := 0; w < d.numWorkers; w++ {
		wg.Add(1)
		start := w * chunkSize
		end := start + chunkSize
		if end > len(d.vectors) {
			end = len(d.vectors)
		}

		go func(start, end int) {
			defer wg.Done()
			for i := start; i < end; i++ {
				dist := euclideanDistance(query, d.vectors[i])
				results <- distanceResult{
					index:    i,
					distance: dist,
					isFraud:  d.fraudFlags[i],
				}
			}
		}(start, end)
	}

	go func() {
		wg.Wait()
		close(results)
	}()

	allResults := make([]distanceResult, 0, len(d.vectors))
	for r := range results {
		allResults = append(allResults, r)
	}

	return topK(allResults, K)
}

// euclideanDistance computes the squared Euclidean distance between two vectors.
// Using squared distance (without sqrt) is faster and sufficient for KNN comparison.
func euclideanDistance(a, b []float64) float64 {
	var sum float64
	for i := 0; i < len(a); i++ {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return sum
}

// topK returns the K closest neighbors from all results.
func topK(results []distanceResult, k int) []neighbor {
	heap := make([]neighbor, k)
	for i := 0; i < k && i < len(results); i++ {
		heap[i] = neighbor{
			Index:    results[i].index,
			Distance: results[i].distance,
			IsFraud:  results[i].isFraud,
		}
	}

	maxIdx := 0
	for i := 1; i < k && i < len(results); i++ {
		if heap[i].Distance > heap[maxIdx].Distance {
			maxIdx = i
		}
	}

	for i := k; i < len(results); i++ {
		if results[i].distance < heap[maxIdx].Distance {
			heap[maxIdx] = neighbor{
				Index:    results[i].index,
				Distance: results[i].distance,
				IsFraud:  results[i].isFraud,
			}
			for j := 0; j < k; j++ {
				if heap[j].Distance > heap[maxIdx].Distance {
					maxIdx = j
				}
			}
		}
	}

	return heap[:k]
}