package knn

// #cgo CFLAGS: -march=x86-64-v3 -O3 -flto
// #cgo LDFLAGS: -lm
// #include "simd_brute.h"
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

const bruteLoadMagic uint32 = 0x42525554

type BruteAVX2Index struct {
	data   []int16 // SoA: [DIM][N]
	labels []byte
	N      int
}

func NewBruteAVX2Index() *BruteAVX2Index { return &BruteAVX2Index{} }

func LoadBruteAVX2(path string) (*BruteAVX2Index, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var magic, version, N, dim uint32
	if err := binary.Read(f, binary.LittleEndian, &magic); err != nil || magic != bruteLoadMagic {
		return nil, fmt.Errorf("invalid brute magic")
	}
	if err := binary.Read(f, binary.LittleEndian, &version); err != nil || version != 1 {
		return nil, fmt.Errorf("unsupported brute version")
	}
	binary.Read(f, binary.LittleEndian, &N)
	binary.Read(f, binary.LittleEndian, &dim)
	if dim != DIM {
		return nil, fmt.Errorf("expected dim=%d, got %d", DIM, dim)
	}

	soa := make([]int16, int(N)*DIM)
	if err := binary.Read(f, binary.LittleEndian, soa); err != nil {
		return nil, err
	}

	labels := make([]byte, N)
	if _, err := io.ReadFull(f, labels); err != nil {
		return nil, err
	}

	return &BruteAVX2Index{data: soa, labels: labels, N: int(N)}, nil
}

func (idx *BruteAVX2Index) Predict(query model.Vector14, k int) float64 {
	if idx.N == 0 || idx.data == nil {
		return 0
	}
	if k == 0 {
		k = K
	}

	var qi [DIM]int16
	for d := 0; d < DIM; d++ {
		if query[d] < -1.0 {
			qi[d] = -int16Scale
		} else if query[d] > 1.0 {
			qi[d] = int16Scale
		} else {
			qi[d] = int16(math.Round(float64(query[d] * int16Scale)))
		}
	}

	var fraudCount C.int
	var qiC [DIM]C.int16_t
	for d := 0; d < DIM; d++ {
		qiC[d] = C.int16_t(qi[d])
	}

	C.brute_fraud_count_avx2(
		(*C.int16_t)(unsafe.Pointer(&idx.data[0])),
		(*C.uint8_t)(unsafe.Pointer(&idx.labels[0])),
		C.int(idx.N),
		(*C.int16_t)(unsafe.Pointer(&qiC[0])),
		C.int(k),
		&fraudCount,
	)

	return float64(fraudCount) / float64(k)
}

func (idx *BruteAVX2Index) Count() int     { return idx.N }
func (idx *BruteAVX2Index) FraudCount() int {
	n := 0
	for _, b := range idx.labels {
		n += int(b)
	}
	return n
}

func ExistsBrute(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}
