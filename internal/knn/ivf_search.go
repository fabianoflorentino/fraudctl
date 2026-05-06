// Package knn — pure-Go IVF search (no CGo).
//
// Format v4 layout (AoS, bit-packed labels):
//
//	vectors[i*DIM + d]  — quantized int16, AoS order within each cluster
//	labels bit-packed   — labels[i/8] bit (i%8) == 1 → fraud
//
// Algorithm per query:
//  1. Sorted-insertion top-nprobe centroid selection.
//  2. For each probe cluster, iterate vectors sequentially.
//     Stage 1: compute dims 0-7, early-exit if dist ≥ worst-K.
//     Stage 2: compute dims 8-13, update top-K heap.
//  3. Boundary retry if fraud count in [lo, hi].
//  4. Return raw fraud count (0..K).
package knn

import (
	"math"

	"github.com/fabianoflorentino/fraudctl/internal/model"
)

// topK5 is a fixed-size sorted array tracking the K=5 nearest neighbours.
// Kept sorted ascending by dist so topDist[K-1] is always the worst.
type topK5 struct {
	dist  [K]uint64
	idx   [K]int
	count int
}

func newTopK5() topK5 {
	var h topK5
	for i := range h.dist {
		h.dist[i] = math.MaxUint64
	}
	return h
}

func (h *topK5) worstDist() uint64 { return h.dist[K-1] }

// tryInsert does a sorted insertion if dist < worst.
func (h *topK5) tryInsert(dist uint64, vecIdx int) {
	if dist >= h.dist[K-1] {
		return
	}
	// Shift right from the back to find insertion point.
	pos := K - 1
	for pos > 0 && dist < h.dist[pos-1] {
		h.dist[pos] = h.dist[pos-1]
		h.idx[pos] = h.idx[pos-1]
		pos--
	}
	h.dist[pos] = dist
	h.idx[pos] = vecIdx
	if h.count < K {
		h.count++
	}
}

// fraudCount returns how many of the K neighbours are labelled fraud.
// labels is bit-packed: labels[i/8] bit (i%8).
func (h *topK5) fraudCount(labels []byte) int {
	n := 0
	for i := 0; i < K; i++ {
		if h.dist[i] == math.MaxUint64 {
			break
		}
		vi := h.idx[i]
		if (labels[vi>>3] & (1 << uint(vi&7))) != 0 {
			n++
		}
	}
	return n
}

// maxProbes is the maximum total probes (nprobe + retryExtra) we ever use.
// Kept as a fixed constant so selectProbes can use a stack-allocated array.
const maxProbes = 32

// selectProbes fills out[:nprobe] sorted by ascending distance to query.
// Uses sorted insertion — nprobe is tiny (≤maxProbes), so O(nprobe) per centroid is fine.
// out must have capacity ≥ nprobe.
func selectProbes(centroids []float32, nlist int, query [DIM]float32, nprobe int, out []int) {
	if nprobe > nlist {
		nprobe = nlist
	}

	var probeDist [maxProbes]float32
	for i := 0; i < nprobe; i++ {
		probeDist[i] = math.MaxFloat32
	}

	for ci := 0; ci < nlist; ci++ {
		base := ci * DIM
		var d float32
		for dim := 0; dim < DIM; dim++ {
			diff := query[dim] - centroids[base+dim]
			d += diff * diff
		}
		if d < probeDist[nprobe-1] {
			pos := nprobe - 1
			for pos > 0 && d < probeDist[pos-1] {
				probeDist[pos] = probeDist[pos-1]
				out[pos] = out[pos-1]
				pos--
			}
			probeDist[pos] = d
			out[pos] = ci
		}
	}
}

// scanCluster scans vectors [start, end) in AoS layout and updates h.
//
// vectors layout: vectors[(globalVecIdx)*DIM + d]
// labels  layout: bit-packed, labels[globalVecIdx/8] bit (globalVecIdx%8)
//
// Two-stage early exit (inspired by xgboost-go reference):
//   Stage 1: dims 0–7  → if dist ≥ worst, skip vector entirely.
//   Stage 2: dims 8–13 → complete the distance.
func scanCluster(vectors []int16, labels []byte, start, end int, q [DIM]int16, h *topK5) {
	worst := h.worstDist()

	for i := start; i < end; i++ {
		base := i * DIM

		// Stage 1: dims 0-7
		v0 := int32(vectors[base+0]) - int32(q[0])
		v1 := int32(vectors[base+1]) - int32(q[1])
		v2 := int32(vectors[base+2]) - int32(q[2])
		v3 := int32(vectors[base+3]) - int32(q[3])
		v4 := int32(vectors[base+4]) - int32(q[4])
		v5 := int32(vectors[base+5]) - int32(q[5])
		v6 := int32(vectors[base+6]) - int32(q[6])
		v7 := int32(vectors[base+7]) - int32(q[7])
		dist := uint64(v0*v0) + uint64(v1*v1) + uint64(v2*v2) + uint64(v3*v3) +
			uint64(v4*v4) + uint64(v5*v5) + uint64(v6*v6) + uint64(v7*v7)

		if dist >= worst {
			continue
		}

		// Stage 2: dims 8-13
		v8 := int32(vectors[base+8]) - int32(q[8])
		v9 := int32(vectors[base+9]) - int32(q[9])
		v10 := int32(vectors[base+10]) - int32(q[10])
		v11 := int32(vectors[base+11]) - int32(q[11])
		v12 := int32(vectors[base+12]) - int32(q[12])
		v13 := int32(vectors[base+13]) - int32(q[13])
		dist += uint64(v8*v8) + uint64(v9*v9) + uint64(v10*v10) +
			uint64(v11*v11) + uint64(v12*v12) + uint64(v13*v13)

		h.tryInsert(dist, i)
		worst = h.worstDist()
	}
}

// quantizeQuery converts a float32 query to int16 (same scale as stored vectors).
func quantizeQuery(query model.Vector14) [DIM]int16 {
	var q [DIM]int16
	for d := 0; d < DIM; d++ {
		v := query[d]
		if v > 1.0 {
			q[d] = int16Scale
		} else if v < -1.0 {
			q[d] = -int16Scale
		} else if v < 0 {
			q[d] = int16(v*int16Scale - 0.5)
		} else {
			q[d] = int16(v*int16Scale + 0.5)
		}
	}
	return q
}

// PredictRaw returns the raw fraud neighbour count (0..K) using pure-Go IVF
// search with incremental boundary retry.
func (idx *IVFIndex) PredictRaw(query model.Vector14, nprobe int) int {
	if len(idx.vectors) == 0 {
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

	var probesBuf [maxProbes]int // stack-allocated, zero heap
	probes := probesBuf[:]
	selectProbes(idx.centroids, idx.nlist, qf, totalProbes, probes)

	h := newTopK5()

	for pi := 0; pi < nprobe; pi++ {
		ci := probes[pi]
		start := int(idx.offsets[ci])
		end := int(idx.offsets[ci+1])
		if start == end {
			continue
		}
		scanCluster(idx.vectors, idx.labels, start, end, qi, &h)
	}

	fraud := h.fraudCount(idx.labels)

	if idx.retryExtra > 0 && fraud >= idx.boundaryLo && fraud <= idx.boundaryHi {
		for pi := nprobe; pi < totalProbes; pi++ {
			ci := probes[pi]
			start := int(idx.offsets[ci])
			end := int(idx.offsets[ci+1])
			if start == end {
				continue
			}
			scanCluster(idx.vectors, idx.labels, start, end, qi, &h)
		}
		fraud = h.fraudCount(idx.labels)
	}

	return fraud
}
