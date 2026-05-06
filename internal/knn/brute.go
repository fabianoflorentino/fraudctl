// Package knn implements k-nearest-neighbor search over 14-dimensional vectors.
package knn

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/fabianoflorentino/fraudctl/internal/model"
)

// IVFIndex holds the IVF index in format v5:
//   centroids: SoA — centroids[d*nlist + ci] (transposed on load for cache-friendly selectProbes)
//   vectors:   AoS int16, layout vectors[i*DIM + d]
//   labels:    bit-packed, labels[i/8] bit (i%8) == 1 → fraud
//   offsets:   offsets[ci]..offsets[ci+1] are the global vector indices for cluster ci
//   bboxMin:   AoI int16 — bboxMin[ci*DIM+d] = min quantized value in dim d for cluster ci
//   bboxMax:   AoI int16 — bboxMax[ci*DIM+d] = max quantized value in dim d for cluster ci
type IVFIndex struct {
	centroids  []float32 // SoA: [DIM * nlist], indexed as centroids[d*nlist + ci]
	vectors    []int16   // AoS: [N * DIM]
	labels     []byte    // bit-packed: [ceil(N/8)]
	offsets    []uint32
	bboxMin    []int16 // AoI: [nlist * DIM]
	bboxMax    []int16 // AoI: [nlist * DIM]
	nlist      int
	nprobe     int
	retryExtra int
	boundaryLo int
	boundaryHi int
}

func (idx *IVFIndex) SetNProbe(n int) {
	if n < 1 {
		n = 1
	}
	idx.nprobe = n
}

func (idx *IVFIndex) NProbe() int { return idx.nprobe }

// SetRetry configures the incremental boundary retry.
func (idx *IVFIndex) SetRetry(retryExtra, lo, hi int) {
	idx.retryExtra = retryExtra
	idx.boundaryLo = lo
	idx.boundaryHi = hi
}

func NewIVFIndex() *IVFIndex { return &IVFIndex{} }

// LoadIVF loads an IVF index in format v5 (AoS, bit-packed labels, bbox per cluster).
func LoadIVF(path string) (*IVFIndex, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open ivf: %w", err)
	}
	defer f.Close()

	var magic, version, nlist, dim uint32
	if err := binary.Read(f, binary.LittleEndian, &magic); err != nil || magic != ivfMagic {
		return nil, fmt.Errorf("invalid ivf magic (got 0x%08x)", magic)
	}
	if err := binary.Read(f, binary.LittleEndian, &version); err != nil || version != 5 {
		return nil, fmt.Errorf("unsupported ivf version %d (expected 5)", version)
	}
	if err := binary.Read(f, binary.LittleEndian, &nlist); err != nil {
		return nil, fmt.Errorf("read nlist: %w", err)
	}
	if err := binary.Read(f, binary.LittleEndian, &dim); err != nil || dim != 14 {
		return nil, fmt.Errorf("expected dim=14, got %d", dim)
	}

	// Total number of vectors
	var n uint32
	if err := binary.Read(f, binary.LittleEndian, &n); err != nil {
		return nil, fmt.Errorf("read n: %w", err)
	}

	// Centroids: nlist * DIM float32 (on-disk: AoI row-major, ci-major)
	// Transpose to SoA (dim-major) for cache-friendly selectProbes.
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

	// Offsets: (nlist+1) uint32
	offsets := make([]uint32, nlist+1)
	if err := binary.Read(f, binary.LittleEndian, offsets); err != nil {
		return nil, fmt.Errorf("read offsets: %w", err)
	}

	// BBox: nlist * DIM int16 each
	bboxMin := make([]int16, int(nlist)*DIM)
	if err := binary.Read(f, binary.LittleEndian, bboxMin); err != nil {
		return nil, fmt.Errorf("read bbox_min: %w", err)
	}
	bboxMax := make([]int16, int(nlist)*DIM)
	if err := binary.Read(f, binary.LittleEndian, bboxMax); err != nil {
		return nil, fmt.Errorf("read bbox_max: %w", err)
	}

	// Vectors: n * DIM int16 (AoS)
	vectors := make([]int16, int(n)*DIM)
	if err := binary.Read(f, binary.LittleEndian, vectors); err != nil {
		return nil, fmt.Errorf("read vectors: %w", err)
	}

	// Labels: ceil(n/8) bytes, bit-packed
	labelBytes := (int(n) + 7) / 8
	labels := make([]byte, labelBytes)
	if _, err := io.ReadFull(f, labels); err != nil {
		return nil, fmt.Errorf("read labels: %w", err)
	}

	idx := &IVFIndex{
		nlist:     int(nlist),
		nprobe:    16,
		centroids: centroids,
		vectors:   vectors,
		labels:    labels,
		offsets:   offsets,
		bboxMin:   bboxMin,
		bboxMax:   bboxMax,
	}

	// Pre-touch all pages to avoid page faults during queries.
	for i := 0; i < len(vectors); i += 4096 {
		_ = vectors[i]
	}
	for i := 0; i < len(labels); i += 4096 {
		_ = labels[i]
	}

	// Keep pages hot to prevent swapping under memory pressure.
	go func() {
		ticker := time.NewTicker(10 * time.Second)
		defer ticker.Stop()
		for range ticker.C {
			for i := 0; i < len(vectors); i += 2048 {
				_ = vectors[i]
			}
		}
	}()

	return idx, nil
}

func (idx *IVFIndex) Predict(query model.Vector14, k int) float64 {
	return float64(idx.PredictRaw(query, idx.nprobe)) / float64(K)
}

// Count returns approximate total number of vectors (last offset value × 1,
// since offsets are in vector units in v4).
func (idx *IVFIndex) Count() int {
	return int(idx.offsets[idx.nlist])
}

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

func (idx *IVFIndex) DebugNList() int        { return idx.nlist }
func (idx *IVFIndex) DebugOffsets() []uint32 { return idx.offsets }
func (idx *IVFIndex) DebugCentroids() []float32 { return idx.centroids }
