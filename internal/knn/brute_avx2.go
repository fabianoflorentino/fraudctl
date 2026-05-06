package knn

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"

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

// Predict performs pure-Go brute-force KNN on SoA int16 layout.
// data layout: data[d*N + i] = quantized dim d of vector i.
func (idx *BruteAVX2Index) Predict(query model.Vector14, k int) float64 {
	if idx.N == 0 || idx.data == nil {
		return 0
	}
	if k == 0 {
		k = K
	}

	var qi [DIM]int32
	for d := 0; d < DIM; d++ {
		qi[d] = int32(quantizeFloat32(query[d]))
	}

	// max-heap of size k tracking worst distance
	hDist := make([]int32, k)
	hLbl := make([]byte, k)
	for i := range hDist {
		hDist[i] = math.MaxInt32
	}
	worst := 0

	for i := 0; i < idx.N; i++ {
		var dist int32
		for d := 0; d < DIM; d++ {
			v := int32(idx.data[d*idx.N+i])
			diff := v - qi[d]
			dist += diff * diff
		}
		if dist < hDist[worst] {
			hDist[worst] = dist
			hLbl[worst] = idx.labels[i]
			// find new worst
			wv, wi := hDist[0], 0
			for j := 1; j < k; j++ {
				if hDist[j] > wv {
					wv = hDist[j]
					wi = j
				}
			}
			worst = wi
		}
	}

	fraudCount := 0
	for i := 0; i < k; i++ {
		if hDist[i] != math.MaxInt32 && hLbl[i] == 1 {
			fraudCount++
		}
	}
	filled := 0
	for i := 0; i < k; i++ {
		if hDist[i] != math.MaxInt32 {
			filled++
		}
	}
	if filled == 0 {
		return 0
	}
	return float64(fraudCount) / float64(filled)
}

func (idx *BruteAVX2Index) Count() int  { return idx.N }
func (idx *BruteAVX2Index) NProbe() int { return 0 }
func (idx *BruteAVX2Index) PredictRaw(query model.Vector14, _ int) int {
	score := idx.Predict(query, K)
	return int(math.Round(score * float64(K)))
}
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
