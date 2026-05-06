// Package knn — pure-Go IVF search (no CGo).
//
// Format v5 layout:
//
//	centroids: SoA — centroids[d*nlist + ci] (dim-major, cache-friendly for selectProbes)
//	bbox_min:  AoI int16 — bboxMin[ci*DIM+d]
//	bbox_max:  AoI int16 — bboxMax[ci*DIM+d]
//	vectors:   AoS blocks of 8 — vectors[blockIdx*DIM*8 + d*8 + lane]
//	labels:    bit-packed — labels[i/8] bit (i%8) == 1 → fraud
//
// Algorithm per query:
//  1. selectProbes: SoA centroid scan — processes all nlist centroids dim-by-dim.
//  2. Two-pass adaptive (correct):
//     Pass 1: select fastNProbe clusters → scan.  If fraud ∉ {2,3} → done.
//     Pass 2: select fullNProbe clusters → scan only the NEW ones (fastNProbe..fullNProbe).
//  3. scanCluster: blocks of 8 vectors.
//     Stage 1: dims 0-7  → if ALL 8 exceed worst, skip block.
//     Stage 2: dims 8-13 → complete distance, update top-K.
//  4. bbox pruning: before scanCluster, check bounding box lower-bound vs worst_d.
package knn

import (
	"math"

	"github.com/fabianoflorentino/fraudctl/internal/model"
)

const (
	fastNProbe = 8
	fullNProbe = 24
	blockSize  = 8 // vectors per scan block
)

// topK5 is a fixed-size sorted array tracking the K=5 nearest neighbours.
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

func (h *topK5) tryInsert(dist uint64, vecIdx int) {
	if dist >= h.dist[K-1] {
		return
	}
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

const maxProbes = 32

// bboxMayImprove returns true if the cluster's bounding box could contain a
// vector closer than worstDist to the query. Fully unrolled in high-variance
// dim order (5,6,2,0,7,8,11,12,9,10,1,13,3,4) for early-exit efficiency.
func bboxMayImprove(bboxMin, bboxMax []int16, ci int, q [DIM]int16, worstDist uint64) bool {
	base := ci * DIM
	var d uint64
	var diff int32

	diff = 0
	if q[5] < bboxMin[base+5] {
		diff = int32(bboxMin[base+5]) - int32(q[5])
	} else if q[5] > bboxMax[base+5] {
		diff = int32(q[5]) - int32(bboxMax[base+5])
	}
	d += uint64(diff * diff)
	if d >= worstDist {
		return false
	}

	diff = 0
	if q[6] < bboxMin[base+6] {
		diff = int32(bboxMin[base+6]) - int32(q[6])
	} else if q[6] > bboxMax[base+6] {
		diff = int32(q[6]) - int32(bboxMax[base+6])
	}
	d += uint64(diff * diff)
	if d >= worstDist {
		return false
	}

	diff = 0
	if q[2] < bboxMin[base+2] {
		diff = int32(bboxMin[base+2]) - int32(q[2])
	} else if q[2] > bboxMax[base+2] {
		diff = int32(q[2]) - int32(bboxMax[base+2])
	}
	d += uint64(diff * diff)
	if d >= worstDist {
		return false
	}

	diff = 0
	if q[0] < bboxMin[base+0] {
		diff = int32(bboxMin[base+0]) - int32(q[0])
	} else if q[0] > bboxMax[base+0] {
		diff = int32(q[0]) - int32(bboxMax[base+0])
	}
	d += uint64(diff * diff)
	if d >= worstDist {
		return false
	}

	diff = 0
	if q[7] < bboxMin[base+7] {
		diff = int32(bboxMin[base+7]) - int32(q[7])
	} else if q[7] > bboxMax[base+7] {
		diff = int32(q[7]) - int32(bboxMax[base+7])
	}
	d += uint64(diff * diff)
	if d >= worstDist {
		return false
	}

	diff = 0
	if q[8] < bboxMin[base+8] {
		diff = int32(bboxMin[base+8]) - int32(q[8])
	} else if q[8] > bboxMax[base+8] {
		diff = int32(q[8]) - int32(bboxMax[base+8])
	}
	d += uint64(diff * diff)
	if d >= worstDist {
		return false
	}

	diff = 0
	if q[11] < bboxMin[base+11] {
		diff = int32(bboxMin[base+11]) - int32(q[11])
	} else if q[11] > bboxMax[base+11] {
		diff = int32(q[11]) - int32(bboxMax[base+11])
	}
	d += uint64(diff * diff)
	if d >= worstDist {
		return false
	}

	diff = 0
	if q[12] < bboxMin[base+12] {
		diff = int32(bboxMin[base+12]) - int32(q[12])
	} else if q[12] > bboxMax[base+12] {
		diff = int32(q[12]) - int32(bboxMax[base+12])
	}
	d += uint64(diff * diff)
	if d >= worstDist {
		return false
	}

	diff = 0
	if q[9] < bboxMin[base+9] {
		diff = int32(bboxMin[base+9]) - int32(q[9])
	} else if q[9] > bboxMax[base+9] {
		diff = int32(q[9]) - int32(bboxMax[base+9])
	}
	d += uint64(diff * diff)
	if d >= worstDist {
		return false
	}

	diff = 0
	if q[10] < bboxMin[base+10] {
		diff = int32(bboxMin[base+10]) - int32(q[10])
	} else if q[10] > bboxMax[base+10] {
		diff = int32(q[10]) - int32(bboxMax[base+10])
	}
	d += uint64(diff * diff)
	if d >= worstDist {
		return false
	}

	diff = 0
	if q[1] < bboxMin[base+1] {
		diff = int32(bboxMin[base+1]) - int32(q[1])
	} else if q[1] > bboxMax[base+1] {
		diff = int32(q[1]) - int32(bboxMax[base+1])
	}
	d += uint64(diff * diff)
	if d >= worstDist {
		return false
	}

	diff = 0
	if q[13] < bboxMin[base+13] {
		diff = int32(bboxMin[base+13]) - int32(q[13])
	} else if q[13] > bboxMax[base+13] {
		diff = int32(q[13]) - int32(bboxMax[base+13])
	}
	d += uint64(diff * diff)
	if d >= worstDist {
		return false
	}

	diff = 0
	if q[3] < bboxMin[base+3] {
		diff = int32(bboxMin[base+3]) - int32(q[3])
	} else if q[3] > bboxMax[base+3] {
		diff = int32(q[3]) - int32(bboxMax[base+3])
	}
	d += uint64(diff * diff)
	if d >= worstDist {
		return false
	}

	diff = 0
	if q[4] < bboxMin[base+4] {
		diff = int32(bboxMin[base+4]) - int32(q[4])
	} else if q[4] > bboxMax[base+4] {
		diff = int32(q[4]) - int32(bboxMax[base+4])
	}
	d += uint64(diff * diff)
	if d >= worstDist {
		return false
	}

	return true
}

// selectProbes uses SoA centroid layout: centroids[d*nlist + ci].
// Processing one dimension at a time is cache-friendly (stride-1 reads).
func selectProbes(centroids []float32, nlist int, query [DIM]float32, nprobe int, out []int) {
	if nprobe > nlist {
		nprobe = nlist
	}

	var probeDist [maxProbes]float32
	for i := 0; i < nprobe; i++ {
		probeDist[i] = math.MaxFloat32
	}

	// Accumulate squared distance dim by dim (SoA: centroids[d*nlist + ci]).
	var acc [4096]float32 // stack — nlist ≤ 4096
	for ci := 0; ci < nlist; ci++ {
		acc[ci] = 0
	}
	for d := 0; d < DIM; d++ {
		qd := query[d]
		base := d * nlist
		for ci := 0; ci < nlist; ci++ {
			diff := qd - centroids[base+ci]
			acc[ci] += diff * diff
		}
	}

	// Pick top-nprobe by sorted insertion.
	for ci := 0; ci < nlist; ci++ {
		d := acc[ci]
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

// scanCluster scans vectors in blocks of 8 (AoS within block).
//
// Dimension processing order mirrors the C reference (highest variance first):
//
//	Stage 1: dims 5,6,2,0,7,1,3,4  (first 8 of C order — maximize early-exit)
//	Stage 2: dims 8,11,12,9,10,13  (remaining 6)
//
// Two-stage block pruning:
//
//	Stage 1: if ALL 8 lanes exceed worst, skip block.
//	Stage 2: complete distance, update top-K.
func scanCluster(vectors []int16, labels []byte, start, end int, q [DIM]int16, h *topK5) {
	worst := h.worstDist()
	nVecs := end - start

	// Process full blocks of 8.
	nBlocks := nVecs / blockSize
	for b := 0; b < nBlocks; b++ {
		blockBase := (start + b*blockSize) * DIM
		globalBase := start + b*blockSize

		// Stage 1: dims 5,6,2,0,7,1,3,4 for all 8 lanes.
		var dist1 [blockSize]uint64
		anyBelow := false
		for lane := 0; lane < blockSize; lane++ {
			base := blockBase + lane*DIM
			v5 := int32(vectors[base+5]) - int32(q[5])
			v6 := int32(vectors[base+6]) - int32(q[6])
			v2 := int32(vectors[base+2]) - int32(q[2])
			v0 := int32(vectors[base+0]) - int32(q[0])
			v7 := int32(vectors[base+7]) - int32(q[7])
			v1 := int32(vectors[base+1]) - int32(q[1])
			v3 := int32(vectors[base+3]) - int32(q[3])
			v4 := int32(vectors[base+4]) - int32(q[4])
			d := uint64(v5*v5) + uint64(v6*v6) + uint64(v2*v2) + uint64(v0*v0) +
				uint64(v7*v7) + uint64(v1*v1) + uint64(v3*v3) + uint64(v4*v4)
			dist1[lane] = d
			if d < worst {
				anyBelow = true
			}
		}
		if !anyBelow {
			continue
		}

		// Stage 2: dims 8,11,12,9,10,13 for lanes that passed stage 1.
		for lane := 0; lane < blockSize; lane++ {
			if dist1[lane] >= worst {
				continue
			}
			base := blockBase + lane*DIM
			v8 := int32(vectors[base+8]) - int32(q[8])
			v11 := int32(vectors[base+11]) - int32(q[11])
			v12 := int32(vectors[base+12]) - int32(q[12])
			v9 := int32(vectors[base+9]) - int32(q[9])
			v10 := int32(vectors[base+10]) - int32(q[10])
			v13 := int32(vectors[base+13]) - int32(q[13])
			dist := dist1[lane] + uint64(v8*v8) + uint64(v11*v11) + uint64(v12*v12) +
				uint64(v9*v9) + uint64(v10*v10) + uint64(v13*v13)
			h.tryInsert(dist, globalBase+lane)
			worst = h.worstDist()
		}
	}

	// Tail: remaining vectors that don't fill a full block.
	// Scalar loop with per-dim early-exit in C reference order: 5,6,2,0,7,8,11,12,9,10,1,13,3,4
	tailStart := start + nBlocks*blockSize
	for i := tailStart; i < end; i++ {
		base := i * DIM
		v5 := int32(vectors[base+5]) - int32(q[5])
		v6 := int32(vectors[base+6]) - int32(q[6])
		dist := uint64(v5*v5) + uint64(v6*v6)
		if dist >= worst {
			continue
		}
		v2 := int32(vectors[base+2]) - int32(q[2])
		dist += uint64(v2 * v2)
		if dist >= worst {
			continue
		}
		v0 := int32(vectors[base+0]) - int32(q[0])
		dist += uint64(v0 * v0)
		if dist >= worst {
			continue
		}
		v7 := int32(vectors[base+7]) - int32(q[7])
		dist += uint64(v7 * v7)
		if dist >= worst {
			continue
		}
		v8 := int32(vectors[base+8]) - int32(q[8])
		dist += uint64(v8 * v8)
		if dist >= worst {
			continue
		}
		v11 := int32(vectors[base+11]) - int32(q[11])
		dist += uint64(v11 * v11)
		if dist >= worst {
			continue
		}
		v12 := int32(vectors[base+12]) - int32(q[12])
		dist += uint64(v12 * v12)
		if dist >= worst {
			continue
		}
		v9 := int32(vectors[base+9]) - int32(q[9])
		dist += uint64(v9 * v9)
		if dist >= worst {
			continue
		}
		v10 := int32(vectors[base+10]) - int32(q[10])
		dist += uint64(v10 * v10)
		if dist >= worst {
			continue
		}
		v1 := int32(vectors[base+1]) - int32(q[1])
		dist += uint64(v1 * v1)
		if dist >= worst {
			continue
		}
		v13 := int32(vectors[base+13]) - int32(q[13])
		dist += uint64(v13 * v13)
		if dist >= worst {
			continue
		}
		v3 := int32(vectors[base+3]) - int32(q[3])
		dist += uint64(v3 * v3)
		if dist >= worst {
			continue
		}
		v4 := int32(vectors[base+4]) - int32(q[4])
		dist += uint64(v4 * v4)
		h.tryInsert(dist, i)
		worst = h.worstDist()
	}
}

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

// PredictRaw uses two-pass adaptive IVF with bbox pruning:
//
//	selectProbes once with fullNProbe=24.
//	Pass 1: scan fastNProbe=8 clusters (bbox pruning when worstDist is known).
//	Pass 2: only if fraud ∈ {2,3} → scan remaining clusters with bbox pruning.
func (idx *IVFIndex) PredictRaw(query model.Vector14, _ int) int {
	if len(idx.vectors) == 0 {
		return 0
	}

	var qf [DIM]float32
	for d := 0; d < DIM; d++ {
		qf[d] = query[d]
	}
	qi := quantizeQuery(query)

	// Select all probes once — reused for both passes.
	full := fullNProbe
	if full > idx.nlist {
		full = idx.nlist
	}
	fast := fastNProbe
	if fast > full {
		fast = full
	}

	var probesBuf [maxProbes]int
	selectProbes(idx.centroids, idx.nlist, qf, full, probesBuf[:full])

	h := newTopK5()
	hasBbox := len(idx.bboxMin) > 0

	// Pass 1: fast probes.
	for pi := 0; pi < fast; pi++ {
		ci := probesBuf[pi]
		start := int(idx.offsets[ci])
		end := int(idx.offsets[ci+1])
		if start >= end {
			continue
		}
		// bbox pruning only once we have a real worst distance.
		if hasBbox && h.count == K {
			if !bboxMayImprove(idx.bboxMin, idx.bboxMax, ci, qi, h.worstDist()) {
				continue
			}
		}
		scanCluster(idx.vectors, idx.labels, start, end, qi, &h)
	}

	fraud := h.fraudCount(idx.labels)

	// Pass 2: only if result is ambiguous (boundary zone 2 or 3 out of 5).
	if fraud == 2 || fraud == 3 {
		for pi := fast; pi < full; pi++ {
			ci := probesBuf[pi]
			start := int(idx.offsets[ci])
			end := int(idx.offsets[ci+1])
			if start >= end {
				continue
			}
			if hasBbox && h.count == K {
				if !bboxMayImprove(idx.bboxMin, idx.bboxMax, ci, qi, h.worstDist()) {
					continue
				}
			}
			scanCluster(idx.vectors, idx.labels, start, end, qi, &h)
		}
		fraud = h.fraudCount(idx.labels)
	}

	return fraud
}
