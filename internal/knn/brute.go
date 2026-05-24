// Package knn implements k-nearest-neighbor search over 14-dimensional vectors.
package knn

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"syscall"
	"unsafe"

	"github.com/fabianoflorentino/fraudctl/internal/model"
)

// IVFIndex holds the IVF index in format v5:
//
//	centroids: SoA — centroids[d*nlist + ci] (transposed on load for cache-friendly selectProbes)
//	vectors:   AoS int16, layout vectors[i*DIM + d]
//	labels:    bit-packed, labels[i/8] bit (i%8) == 1 → fraud
//	offsets:   offsets[ci]..offsets[ci+1] are the global vector indices for cluster ci
//	bboxMin:   AoI int16 — bboxMin[ci*DIM+d] = min quantized value in dim d for cluster ci
//	bboxMax:   AoI int16 — bboxMax[ci*DIM+d] = max quantized value in dim d for cluster ci
type IVFIndex struct {
	centroids     []float32 // SoA: [DIM * nlist], indexed as centroids[d*nlist + ci]
	centroidNorms []float32 // pre-computed sum(c²) per centroid for dot-to-L2 conversion
	vectors       []int16   // AoS: [N * DIM]
	labels        []byte    // bit-packed: [ceil(N/8)]
	offsets       []uint32
	bboxMin       []int16   // AoI: [nlist * DIM]
	bboxMax       []int16   // AoI: [nlist * DIM]
	radii         []float32 // per-cluster radius for triangle-inequality pruning
	nlist         int
	nprobe        int
	quickProbe    int
	boundaryLo    int
	boundaryHi    int
}

// SetNProbe sets the number of clusters to probe during IVF search.
func (idx *IVFIndex) SetNProbe(n int) {
	if n < 1 {
		n = 1
	}
	idx.nprobe = n
}

// NProbe returns the current number of clusters to probe.
func (idx *IVFIndex) NProbe() int { return idx.nprobe }

// SetRetry configures 2-tier early exit strategy.
//
//	quick: number of clusters for quick probe (early exit if clear)
//	lo, hi: ambiguous zone (fraud count in [lo, hi] triggers full scan)
func (idx *IVFIndex) SetRetry(quick, lo, hi int) {
	idx.quickProbe = quick
	idx.boundaryLo = lo
	idx.boundaryHi = hi
}

// NewIVFIndex creates an empty IVFIndex.
func NewIVFIndex() *IVFIndex { return &IVFIndex{} }

// LoadIVF loads an IVF index in format v5 or v6 (v6 adds per-cluster radii).
func LoadIVF(path string) (*IVFIndex, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open ivf: %w", err)
	}
	defer func() { _ = f.Close() }()

	var magic, version, nlist, dim uint32
	if err := binary.Read(f, binary.LittleEndian, &magic); err != nil || magic != ivfMagic {
		return nil, fmt.Errorf("invalid ivf magic (got 0x%08x)", magic)
	}
	if err := binary.Read(f, binary.LittleEndian, &version); err != nil {
		return nil, fmt.Errorf("read version: %w", err)
	}
	if version != 5 && version != 6 {
		return nil, fmt.Errorf("unsupported ivf version %d", version)
	}
	if err := binary.Read(f, binary.LittleEndian, &nlist); err != nil {
		return nil, fmt.Errorf("read nlist: %w", err)
	}
	if err := binary.Read(f, binary.LittleEndian, &dim); err != nil || dim != 14 {
		return nil, fmt.Errorf("expected dim=14, got %d", dim)
	}

	var n uint32
	if err := binary.Read(f, binary.LittleEndian, &n); err != nil {
		return nil, fmt.Errorf("read n: %w", err)
	}

	rawCentroids := make([]float32, nlist*dim)
	if err := binary.Read(f, binary.LittleEndian, rawCentroids); err != nil {
		return nil, fmt.Errorf("read centroids: %w", err)
	}
	centroids := make([]float32, nlist*dim)
	for ci := 0; ci < int(nlist); ci++ {
		for d := 0; d < int(dim); d++ {
			centroids[d*int(nlist)+ci] = rawCentroids[ci*int(dim)+d]
		}
	}

	offsets := make([]uint32, nlist+1)
	if err := binary.Read(f, binary.LittleEndian, offsets); err != nil {
		return nil, fmt.Errorf("read offsets: %w", err)
	}

	bboxMin := make([]int16, int(nlist)*DIM)
	if err := binary.Read(f, binary.LittleEndian, bboxMin); err != nil {
		return nil, fmt.Errorf("read bbox_min: %w", err)
	}
	bboxMax := make([]int16, int(nlist)*DIM)
	if err := binary.Read(f, binary.LittleEndian, bboxMax); err != nil {
		return nil, fmt.Errorf("read bbox_max: %w", err)
	}

	var radii []float32
	if version >= 6 {
		radii = make([]float32, nlist)
		if err := binary.Read(f, binary.LittleEndian, radii); err != nil {
			return nil, fmt.Errorf("read radii: %w", err)
		}
	}

	vectors := make([]int16, int(n)*DIM)
	if err := binary.Read(f, binary.LittleEndian, vectors); err != nil {
		return nil, fmt.Errorf("read vectors: %w", err)
	}

	labelBytes := (int(n) + 7) / 8
	labels := make([]byte, labelBytes)
	if _, err := io.ReadFull(f, labels); err != nil {
		return nil, fmt.Errorf("read labels: %w", err)
	}

	centroidNorms := make([]float32, nlist)
	for ci := 0; ci < int(nlist); ci++ {
		var sum float32
		for d := 0; d < int(dim); d++ {
			v := centroids[d*int(nlist)+ci]
			sum += v * v
		}
		centroidNorms[ci] = sum
	}

	idx := &IVFIndex{
		nlist:         int(nlist),
		nprobe:        64,
		centroids:     centroids,
		centroidNorms: centroidNorms,
		vectors:       vectors,
		labels:        labels,
		offsets:       offsets,
		bboxMin:       bboxMin,
		bboxMax:       bboxMax,
		radii:         radii,
	}

	// Pre-touch all pages to avoid page faults during queries.
	for i := 0; i < len(vectors); i += 4096 {
		_ = vectors[i]
	}
	for i := 0; i < len(labels); i += 4096 {
		_ = labels[i]
	}

	if len(vectors) > 0 {
		vb := unsafe.Slice((*byte)(unsafe.Pointer(&vectors[0])), len(vectors)*2)
		_ = syscall.Madvise(vb, syscall.MADV_WILLNEED)
	}

	return idx, nil
}

// Predict returns the fraud score (fraudCount/K) for the given query vector
// using the current nprobe setting.
func (idx *IVFIndex) Predict(query model.Vector14, k int) float64 {
	return float64(idx.PredictRaw(query, idx.nprobe)) / float64(K)
}

// Count returns approximate total number of vectors (last offset value × 1,
// since offsets are in vector units in v4).
func (idx *IVFIndex) Count() int {
	return int(idx.offsets[idx.nlist])
}

// FraudCount returns the total number of fraud-labeled vectors in the index.
func (idx *IVFIndex) FraudCount() int {
	n := 0
	total := idx.Count()
	for i := 0; i < total; i++ {
		if (idx.labels[i>>3] & (1 << uint(i&7))) != 0 {
			n++
		}
	}
	return n
}

// DebugNList returns the number of clusters in the index (for testing).
func (idx *IVFIndex) DebugNList() int { return idx.nlist }

// DebugOffsets returns the cluster offset array (for testing).
func (idx *IVFIndex) DebugOffsets() []uint32 { return idx.offsets }

// DebugCentroids returns the centroid array in SoA layout (for testing).
func (idx *IVFIndex) DebugCentroids() []float32 { return idx.centroids }

// ensureCentroidNorms lazily computes centroid norms if not already set.
// Production path always has norms (set in LoadIVF); this handles tests.
func (idx *IVFIndex) ensureCentroidNorms() {
	if len(idx.centroidNorms) > 0 {
		return
	}
	norms := make([]float32, idx.nlist)
	for ci := 0; ci < idx.nlist; ci++ {
		var s float32
		for d := 0; d < DIM; d++ {
			v := idx.centroids[d*idx.nlist+ci]
			s += v * v
		}
		norms[ci] = s
	}
	idx.centroidNorms = norms
}
