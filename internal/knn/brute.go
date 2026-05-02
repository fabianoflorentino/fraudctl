// Package knn implements k-nearest-neighbor search over 14-dimensional vectors.
package knn

// #cgo CFLAGS: -march=x86-64-v3 -O3 -flto
// #cgo LDFLAGS: -lm
// #include "simd_knn.h"
// #include <stdlib.h>
import "C"
import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"unsafe"

	"github.com/fabianoflorentino/fraudctl/internal/model"
)

type IVFIndex struct {
	centroids *C.float
	blocks    *C.int16_t
	labels    *C.uchar
	offsets   []uint32
	nlist     int
	nprobe    int
	nvec      int
}

func (idx *IVFIndex) SetNProbe(n int) {
	if n < 1 {
		n = 1
	}
	idx.nprobe = n
}

func NewIVFIndex() *IVFIndex { return &IVFIndex{} }

func LoadIVF(path string) (*IVFIndex, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open ivf: %w", err)
	}
	defer f.Close()

	var magic, version, nlist, dim uint32
	if err := binary.Read(f, binary.LittleEndian, &magic); err != nil || magic != ivfMagic {
		return nil, fmt.Errorf("invalid ivf magic")
	}
	if err := binary.Read(f, binary.LittleEndian, &version); err != nil || version != 3 {
		return nil, fmt.Errorf("unsupported ivf version %d (expected 3)", version)
	}
	binary.Read(f, binary.LittleEndian, &nlist)
	binary.Read(f, binary.LittleEndian, &dim)
	if dim != 14 {
		return nil, fmt.Errorf("expected dim=14, got %d", dim)
	}

	centroids := make([]float32, nlist*dim)
	if err := binary.Read(f, binary.LittleEndian, centroids); err != nil {
		return nil, fmt.Errorf("read centroids: %w", err)
	}

	type clusterMeta struct {
		n          uint32
		nBlocks    uint32
		fileOffset int64
	}
	metas := make([]clusterMeta, nlist)
	totalBlocks := uint64(0)
	for i := uint32(0); i < nlist; i++ {
		var n uint32
		if err := binary.Read(f, binary.LittleEndian, &n); err != nil {
			return nil, fmt.Errorf("read cluster size %d: %w", i, err)
		}
		nBlocks := (n + 7) / 8
		off, _ := f.Seek(0, io.SeekCurrent)
		metas[i] = clusterMeta{n: n, nBlocks: nBlocks, fileOffset: off}
		totalBlocks += uint64(nBlocks)
		skip := int64(nBlocks)*int64(dim)*8*2 + int64(nBlocks)*8
		if _, err := f.Seek(skip, io.SeekCurrent); err != nil {
			return nil, fmt.Errorf("seek cluster %d: %w", i, err)
		}
	}

	offsets := make([]uint32, nlist+1)
	var blockPos uint64

	totalVecBytes := totalBlocks * uint64(dim) * 8 * 2
	totalLabelBytes := totalBlocks * 8

	// Allocate off-heap memory to avoid GC scanning
	cBlocks := (*C.int16_t)(C.malloc(C.size_t(totalVecBytes)))
	if cBlocks == nil {
		return nil, fmt.Errorf("malloc blocks failed")
	}
	cLabels := (*C.uchar)(C.malloc(C.size_t(totalLabelBytes)))
	if cLabels == nil {
		C.free(unsafe.Pointer(cBlocks))
		return nil, fmt.Errorf("malloc labels failed")
	}

	bufBlocks := unsafe.Slice((*byte)(unsafe.Pointer(cBlocks)), totalVecBytes)
	bufLabels := unsafe.Slice((*byte)(unsafe.Pointer(cLabels)), totalLabelBytes)

	for i := uint32(0); i < nlist; i++ {
		nb := uint64(metas[i].nBlocks)
		vecBytes := nb * uint64(dim) * 8 * 2
		labelBytes := nb * 8

		if _, err := f.Seek(metas[i].fileOffset, io.SeekStart); err != nil {
			return nil, fmt.Errorf("seek cluster %d data: %w", i, err)
		}
		if _, err := io.ReadFull(f, bufBlocks[blockPos*uint64(dim)*8*2:blockPos*uint64(dim)*8*2+vecBytes]); err != nil {
			return nil, fmt.Errorf("read cluster %d vecs: %w", i, err)
		}
		if _, err := io.ReadFull(f, bufLabels[blockPos*8:blockPos*8+labelBytes]); err != nil {
			return nil, fmt.Errorf("read cluster %d labels: %w", i, err)
		}

		offsets[i] = uint32(blockPos)
		blockPos += nb
	}
	offsets[nlist] = uint32(blockPos)

	// Pre-touch: read all pages to avoid page faults during queries
	for i := 0; i < int(totalVecBytes); i += 4096 {
		_ = bufBlocks[i]
	}
	for i := 0; i < int(totalLabelBytes); i += 4096 {
		_ = bufLabels[i]
	}

	// Copy centroids to off-heap
	centBytes := uintptr(nlist*dim) * 4
	cCentroids := (*C.float)(C.malloc(C.size_t(centBytes)))
	if cCentroids == nil {
		C.free(unsafe.Pointer(cBlocks))
		C.free(unsafe.Pointer(cLabels))
		return nil, fmt.Errorf("malloc centroids failed")
	}
	copy(unsafe.Slice((*float32)(unsafe.Pointer(cCentroids)), nlist*dim), centroids)

	idx := &IVFIndex{
		nlist:     int(nlist),
		nprobe:    24,
		centroids: cCentroids,
		blocks:    cBlocks,
		labels:    cLabels,
		offsets:   offsets,
		nvec:      int(blockPos) * 8,
	}

	return idx, nil
}

func (idx *IVFIndex) Predict(query model.Vector14, k int) float64 {
	if idx.blocks == nil {
		return 0
	}

	var qi [14]int16
	for d := 0; d < 14; d++ {
		if query[d] < -1.0 {
			qi[d] = -int16Scale
		} else if query[d] > 1.0 {
			qi[d] = int16Scale
		} else {
			qi[d] = int16(math.Round(float64(query[d] * int16Scale)))
		}
	}

	var fraudCount C.int
	var qiC [14]C.int16_t
	for d := 0; d < 14; d++ {
		qiC[d] = C.int16_t(qi[d])
	}

	C.knn_fraud_count_avx2(
		idx.blocks,
		idx.labels,
		(*C.uint32_t)(unsafe.Pointer(&idx.offsets[0])),
		idx.centroids,
		C.int(idx.nlist),
		C.int(idx.nprobe),
		(*C.int16_t)(unsafe.Pointer(&qiC[0])),
		&fraudCount,
	)

	if k == 0 {
		k = K
	}
	return float64(fraudCount) / float64(k)
}

func (idx *IVFIndex) Count() int { return idx.nvec }

func (idx *IVFIndex) FraudCount() int {
	if idx.labels == nil {
		return 0
	}
	n := 0
	labels := unsafe.Slice((*byte)(unsafe.Pointer(idx.labels)), idx.nvec)
	for i := 0; i < idx.nvec; i++ {
		n += int(labels[i])
	}
	return n
}

func (idx *IVFIndex) Close() {
	if idx.blocks != nil {
		C.free(unsafe.Pointer(idx.blocks))
		idx.blocks = nil
	}
	if idx.labels != nil {
		C.free(unsafe.Pointer(idx.labels))
		idx.labels = nil
	}
	if idx.centroids != nil {
		C.free(unsafe.Pointer(idx.centroids))
		idx.centroids = nil
	}
}
