// Package knn — pure-Go IVF search (no CGo).
//
// Layout of idx.blocks (SoA, block of 8 vectors):
//
//	block b, dimension d, slot s  →  blocks[(b*DIM + d)*8 + s]
//
// Quantisation: float32 ∈ [-1,1]  →  int16 × 10000
// Padding slots carry int16Pad (32767 = INT16_MAX), excluded from top-K.
//
// Algorithm per query:
//  1. Find top-nprobe centroids by squared-float32 distance.
//  2. For each probe cluster scan its SoA blocks.
//     - For each block, compute squared int32 distance for all 8 slots in
//       parallel (auto-vectorisable inner loop), then update top-K heap.
//  3. If fraud count ∈ [boundaryLo, boundaryHi] and retryExtra > 0,
//     extend the probe list incrementally (heap preserved).
//  4. Return raw fraud count (0..K).
package knn

import (
	"math"

	"github.com/fabianoflorentino/fraudctl/internal/model"
)

// topK is a fixed-size max-heap tracking the K nearest neighbours.
// dist[i] == math.MaxInt32 means slot i is empty (not yet filled).
type topK struct {
	dist  [K]int32
	label [K]uint8
	worst int // index of the maximum dist in the heap
}

func newTopK() topK {
	var h topK
	for i := range h.dist {
		h.dist[i] = math.MaxInt32
	}
	return h
}

// worstDist returns the current largest distance in the heap.
func (h *topK) worstDist() int32 { return h.dist[h.worst] }

// tryInsert inserts (d, lbl) if d < worst distance, then updates worst.
func (h *topK) tryInsert(d int32, lbl uint8) {
	if d >= h.dist[h.worst] {
		return
	}
	h.dist[h.worst] = d
	h.label[h.worst] = lbl
	// find new worst (K=5, tiny loop)
	wi, wv := 0, h.dist[0]
	for i := 1; i < K; i++ {
		if h.dist[i] > wv {
			wv = h.dist[i]
			wi = i
		}
	}
	h.worst = wi
}

// fraudCount returns how many of the K neighbours are labelled fraud (1).
func (h *topK) fraudCount() int {
	n := 0
	for i := 0; i < K; i++ {
		if h.label[i] == 1 {
			n++
		}
	}
	return n
}

// selectProbes fills out[:nprobe] with the indices of the nprobe nearest
// centroids to query (float32 squared-distance, AoS layout).
// centroids layout: centroids[ci*DIM + d]
func selectProbes(centroids []float32, nlist int, query [DIM]float32, nprobe int, out []int) {
	if nprobe > nlist {
		nprobe = nlist
	}

	probeDist := make([]float32, nprobe)
	for i := range probeDist {
		probeDist[i] = math.MaxFloat32
	}
	worst := 0

	for ci := 0; ci < nlist; ci++ {
		base := ci * DIM
		var d float32
		for dim := 0; dim < DIM; dim++ {
			diff := query[dim] - centroids[base+dim]
			d += diff * diff
		}
		if d < probeDist[worst] {
			out[worst] = ci
			probeDist[worst] = d
			wv := probeDist[0]
			wi := 0
			for i := 1; i < nprobe; i++ {
				if probeDist[i] > wv {
					wv = probeDist[i]
					wi = i
				}
			}
			worst = wi
		}
	}
}

// scanCluster iterates over the SoA blocks belonging to cluster [startBlock, endBlock)
// and updates the top-K heap h.
//
// For each block, we accumulate squared int32 distances for all 8 slots
// simultaneously across all 14 dimensions. The inner loop over DIM=14 with a
// fixed stride of 8 is auto-vectorisable by the Go compiler (SSE2/AVX2).
//
// blocks layout: blocks[(blockIdx*DIM + dim)*8 + slot]
// labels layout: labels[blockIdx*8 + slot]
func scanCluster(blocks []int16, labels []byte, startBlock, endBlock int, q [DIM]int32, h *topK) {
	for bi := startBlock; bi < endBlock; bi++ {
		blockBase := bi * DIM * 8
		labelBase := bi * 8

		// Accumulate squared distance for each of the 8 slots.
		var dist [8]int32

		for d := 0; d < DIM; d++ {
			qd := q[d]
			base := blockBase + d*8
			// This inner loop is over a contiguous int16 slice of length 8,
			// enabling the compiler to auto-vectorise with SIMD.
			for s := 0; s < 8; s++ {
				diff := int32(blocks[base+s]) - qd
				dist[s] += diff * diff
			}
		}

		worst := h.worstDist()
		for s := 0; s < 8; s++ {
			// Padding slots have int16Pad in every dimension, giving a very
			// large distance. We filter them out by checking the raw value of
			// dim 0 for this slot.
			if blocks[blockBase+s] == int16Pad {
				continue
			}
			if dist[s] < worst {
				h.tryInsert(dist[s], labels[labelBase+s])
				worst = h.worstDist()
			}
		}
	}
}

// quantizeQuery converts a float32 query vector to int32 scaled by int16Scale.
// Using int32 to avoid overflow when computing differences (max diff = 20000).
func quantizeQuery(query model.Vector14) [DIM]int32 {
	var q [DIM]int32
	for d := 0; d < DIM; d++ {
		v := query[d]
		if v > 1.0 {
			v = 1.0
		} else if v < -1.0 {
			v = -1.0
		}
		q[d] = int32(math.Round(float64(v) * int16Scale))
	}
	return q
}

// PredictRaw returns the raw fraud neighbour count (0..K) using the pure-Go
// IVF search with incremental boundary retry (no CGo).
func (idx *IVFIndex) PredictRaw(query model.Vector14, nprobe int) int {
	if len(idx.blocks) == 0 {
		return 0
	}

	var qf [DIM]float32
	for d := 0; d < DIM; d++ {
		qf[d] = query[d]
	}
	qi := quantizeQuery(query)

	totalProbes := nprobe + idx.retryExtra
	if totalProbes > idx.nlist {
		totalProbes = idx.nlist
	}
	if nprobe > idx.nlist {
		nprobe = idx.nlist
	}

	// Pre-select all candidate centroids up to totalProbes.
	probes := make([]int, totalProbes)
	selectProbes(idx.centroids, idx.nlist, qf, totalProbes, probes)

	h := newTopK()

	// Phase 1: scan first nprobe clusters.
	for pi := 0; pi < nprobe; pi++ {
		ci := probes[pi]
		start := int(idx.offsets[ci])
		end := int(idx.offsets[ci+1])
		if start == end {
			continue
		}
		scanCluster(idx.blocks, idx.labels, start, end, qi, &h)
	}

	fraud := h.fraudCount()

	// Phase 2: boundary retry — scan extra clusters if result is ambiguous.
	if idx.retryExtra > 0 && fraud >= idx.boundaryLo && fraud <= idx.boundaryHi {
		for pi := nprobe; pi < totalProbes; pi++ {
			ci := probes[pi]
			start := int(idx.offsets[ci])
			end := int(idx.offsets[ci+1])
			if start == end {
				continue
			}
			scanCluster(idx.blocks, idx.labels, start, end, qi, &h)
		}
		fraud = h.fraudCount()
	}

	return fraud
}
