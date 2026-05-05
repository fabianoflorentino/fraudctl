// Package knn implements k-nearest-neighbor search over 14-dimensional vectors.
package knn

// #cgo CFLAGS: -march=x86-64-v3 -O3 -flto
// #cgo LDFLAGS: -lm
// #include "simd_knn.h"
import "C"
import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"time"
	"unsafe"

	"github.com/fabianoflorentino/fraudctl/internal/model"
)

type IVFIndex struct {
	centroids []float32
	blocks    []int16
	labels    []byte
	offsets   []uint32
	nlist     int
	nprobe    int
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
	allVecBytes := make([]byte, totalVecBytes)
	allLabels := make([]byte, totalLabelBytes)

	for i := uint32(0); i < nlist; i++ {
		nb := uint64(metas[i].nBlocks)
		vecBytes := nb * uint64(dim) * 8 * 2
		labelBytes := nb * 8

		if _, err := f.Seek(metas[i].fileOffset, io.SeekStart); err != nil {
			return nil, fmt.Errorf("seek cluster %d data: %w", i, err)
		}
		if _, err := io.ReadFull(f, allVecBytes[blockPos*uint64(dim)*8*2:blockPos*uint64(dim)*8*2+vecBytes]); err != nil {
			return nil, fmt.Errorf("read cluster %d vecs: %w", i, err)
		}
		if _, err := io.ReadFull(f, allLabels[blockPos*8:blockPos*8+labelBytes]); err != nil {
			return nil, fmt.Errorf("read cluster %d labels: %w", i, err)
		}

		offsets[i] = uint32(blockPos)

		blockPos += nb
	}
	offsets[nlist] = uint32(blockPos)

	blocks := unsafe.Slice((*int16)(unsafe.Pointer(&allVecBytes[0])), len(allVecBytes)/2)

	idx := &IVFIndex{
		nlist:     int(nlist),
		nprobe:    4,
		centroids: centroids,
		blocks:    blocks,
		labels:    allLabels,
		offsets:   offsets,
	}

	// Pre-touch all pages to avoid page faults during queries.
	for i := 0; i < len(blocks); i += 4096 {
		_ = blocks[i]
	}
	for i := 0; i < len(allLabels); i += 4096 {
		_ = allLabels[i]
	}

	// Keep pages hot: with swappiness=100 on the host, the kernel aggressively
	// swaps out inactive pages even when RAM is available. Touch every page every
	// 10s to prevent the index from being evicted to swap.
	go func() {
		ticker := time.NewTicker(10 * time.Second)
		defer ticker.Stop()
		for range ticker.C {
			for i := 0; i < len(blocks); i += 2048 {
				_ = blocks[i]
			}
		}
	}()

	return idx, nil
}

func (idx *IVFIndex) Predict(query model.Vector14, k int) float64 {
	if len(idx.blocks) == 0 {
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
		(*C.int16_t)(unsafe.Pointer(&idx.blocks[0])),
		(*C.uint8_t)(unsafe.Pointer(&idx.labels[0])),
		(*C.uint32_t)(unsafe.Pointer(&idx.offsets[0])),
		(*C.float)(unsafe.Pointer(&idx.centroids[0])),
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

func (idx *IVFIndex) Count() int {
	return int(idx.offsets[idx.nlist]) * 8
}

func (idx *IVFIndex) FraudCount() int {
	n := 0
	for _, lb := range idx.labels {
		if lb == 1 {
			n++
		}
	}
	return n
}

func (idx *IVFIndex) DebugBlocks() []int16 {
	return idx.blocks
}

func (idx *IVFIndex) DebugNList() int {
	return idx.nlist
}

func (idx *IVFIndex) DebugOffsets() []uint32 {
	return idx.offsets
}

func (idx *IVFIndex) DebugCentroids() []float32 {
	return idx.centroids
}
