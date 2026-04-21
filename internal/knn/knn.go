package knn

import (
	"fraudctl/internal/vectorizer"
	"sync"
)

const K = 5

type Reference struct {
	Vector vectorizer.Vector
	IsFraud bool
}

type Dataset struct {
	references []Reference
	vectors    [][]float64
	fraudFlags []bool
	numWorkers int
}

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

type neighbor struct {
	Index    int
	Distance float64
	IsFraud  bool
}

type distanceResult struct {
	index    int
	distance float64
	isFraud bool
}

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

func euclideanDistance(a, b []float64) float64 {
	var sum float64
	for i := 0; i < len(a); i++ {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return sum
}

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
