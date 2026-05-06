package knn

import (
	"compress/gzip"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sync"
)

const K = 5
const DIM = 14
const int16Scale = 10000
const int16Pad = math.MaxInt16
const ivfMagic uint32 = 0x49564649
const bruteMagic uint32 = 0x42525554

func quantizeFloat32(v float32) int16 {
	if v > 1.0 {
		return int16Scale
	}
	if v < -1.0 {
		return -int16Scale
	}
	return int16(math.Round(float64(v * int16Scale)))
}

// BuildIVF builds an IVF index in format v5 (SoA blocks, bit-packed labels).
//
// File layout:
//
//	magic       uint32  0x49564649
//	version     uint32  5
//	nlist       uint32
//	dim         uint32  14
//	n           uint32  total vectors
//	centroids   [nlist*DIM]float32
//	offsets     [nlist+1]uint32   — vector indices (not block indices)
//	blockOff    [nlist+1]uint32   — block indices per cluster
//	vectors     [nBlocks*DIM*blockSize]int16  — SoA blocks; block[b][d][lane]
//	            padding: last block has unused lanes filled with int16Pad
//	labels      [ceil(n/8)]byte   — bit-packed: bit i%8 of byte i/8
//
// SoA block layout: block b contains up to blockSize vectors.
// Within a block: offset of dimension d, lane l = (b*DIM*blockSize + d*blockSize + l) * 2
func BuildIVF(refsGz, outPath string, nlist, iterations int) error {
	fmt.Printf("BuildIVF: loading %s ...\n", refsGz)

	f, err := os.Open(refsGz)
	if err != nil {
		return err
	}
	defer f.Close()
	gz, err := gzip.NewReader(f)
	if err != nil {
		return err
	}
	defer gz.Close()

	var flat []float32
	var fraudFlags []bool

	dec := json.NewDecoder(gz)
	if _, err := dec.Token(); err != nil {
		return err
	}
	var entry struct {
		Vector []float64 `json:"vector"`
		Label  string    `json:"label"`
	}
	for dec.More() {
		entry.Vector = entry.Vector[:0]
		if err := dec.Decode(&entry); err != nil {
			break
		}
		for i := 0; i < 14; i++ {
			if i < len(entry.Vector) {
				flat = append(flat, float32(entry.Vector[i]))
			} else {
				flat = append(flat, 0)
			}
		}
		fraudFlags = append(fraudFlags, entry.Label == "fraud")
	}
	n := len(fraudFlags)
	fmt.Printf("BuildIVF: loaded %d vectors\n", n)

	centroids := kmeansInit(flat, n, nlist)

	assign := make([]int, n)
	for iter := 0; iter < iterations; iter++ {
		changed := kmeansAssign(flat, n, centroids, nlist, assign)
		kmeansUpdate(flat, n, centroids, nlist, assign)
		fmt.Printf("  iter %d: %d reassigned\n", iter+1, changed)
		if changed == 0 {
			break
		}
	}

	// Build per-cluster vector index lists.
	clusterLists := make([][]int, nlist)
	for i := 0; i < nlist; i++ {
		clusterLists[i] = make([]int, 0, n/nlist+1)
	}
	for i, ci := range assign {
		clusterLists[ci] = append(clusterLists[ci], i)
	}

	// Build SoA blocks and bit-packed labels in cluster order.
	// offsets[ci] = first vector index in cluster ci (in vector units)
	// blockOff[ci] = first block index in cluster ci
	offsets := make([]uint32, nlist+1)
	blockOff := make([]uint32, nlist+1)
	bitLabels := make([]byte, (n+7)/8)

	// Count total blocks needed.
	totalBlocks := 0
	for ci := 0; ci < nlist; ci++ {
		clusterN := len(clusterLists[ci])
		totalBlocks += (clusterN + blockSize - 1) / blockSize
	}

	soaBlocks := make([]int16, totalBlocks*DIM*blockSize)

	vecPos := 0
	blkPos := 0
	for ci := 0; ci < nlist; ci++ {
		offsets[ci] = uint32(vecPos)
		blockOff[ci] = uint32(blkPos)
		list := clusterLists[ci]
		for lb := 0; lb < len(list); lb += blockSize {
			blockBase := blkPos * DIM * blockSize
			for lane := 0; lane < blockSize; lane++ {
				vi := -1
				if lb+lane < len(list) {
					vi = list[lb+lane]
				}
				for d := 0; d < DIM; d++ {
					idx := blockBase + d*blockSize + lane
					if vi >= 0 {
						soaBlocks[idx] = quantizeFloat32(flat[vi*DIM+d])
					} else {
						soaBlocks[idx] = int16Pad // padding lane
					}
				}
			}
			blkPos++
		}
		for _, vi := range list {
			if fraudFlags[vi] {
				bitLabels[vecPos>>3] |= 1 << uint(vecPos&7)
			}
			vecPos++
		}
	}
	offsets[nlist] = uint32(vecPos)
	blockOff[nlist] = uint32(blkPos)

	fmt.Printf("BuildIVF: writing %s (v5, SoA blocks, bit-packed) ...\n", outPath)
	out, err := os.Create(outPath)
	if err != nil {
		return err
	}
	defer out.Close()

	write32 := func(v uint32) { binary.Write(out, binary.LittleEndian, v) }
	write32(ivfMagic)
	write32(5) // version
	write32(uint32(nlist))
	write32(DIM)
	write32(uint32(n))

	binary.Write(out, binary.LittleEndian, centroids)
	binary.Write(out, binary.LittleEndian, offsets)
	binary.Write(out, binary.LittleEndian, blockOff)
	binary.Write(out, binary.LittleEndian, soaBlocks)
	out.Write(bitLabels)

	fmt.Printf("BuildIVF: done. n=%d nlist=%d blocks=%d\n", n, nlist, blkPos)
	return nil
}

func kmeansInit(flat []float32, n, k int) []float32 {
	rng := rand.New(rand.NewSource(42))
	centroids := make([]float32, k*14)
	for ci := 0; ci < k; ci++ {
		src := rng.Intn(n)
		copy(centroids[ci*14:ci*14+14], flat[src*14:src*14+14])
	}
	return centroids
}

func kmeansAssign(flat []float32, n int, centroids []float32, k int, assign []int) int {
	workers := runtime.GOMAXPROCS(0)
	chunkSize := (n + workers - 1) / workers

	changedPerWorker := make([]int, workers)
	var wg sync.WaitGroup

	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			lo := w * chunkSize
			hi := lo + chunkSize
			if hi > n {
				hi = n
			}
			local := 0
			for i := lo; i < hi; i++ {
				best := 0
				bestD := float32(math.MaxFloat32)
				base := i * 14
				for ci := 0; ci < k; ci++ {
					cb := ci * 14
					// Partial Distance Elimination (PDE): accumulate dims
					// one at a time and bail out as soon as d ≥ bestD.
					d0 := flat[base+0] - centroids[cb+0]
					d := d0 * d0
					if d >= bestD {
						continue
					}
					d1 := flat[base+1] - centroids[cb+1]
					d += d1 * d1
					if d >= bestD {
						continue
					}
					d2 := flat[base+2] - centroids[cb+2]
					d += d2 * d2
					if d >= bestD {
						continue
					}
					d3 := flat[base+3] - centroids[cb+3]
					d += d3 * d3
					if d >= bestD {
						continue
					}
					d4 := flat[base+4] - centroids[cb+4]
					d += d4 * d4
					if d >= bestD {
						continue
					}
					d5 := flat[base+5] - centroids[cb+5]
					d += d5 * d5
					if d >= bestD {
						continue
					}
					d6 := flat[base+6] - centroids[cb+6]
					d += d6 * d6
					if d >= bestD {
						continue
					}
					d7 := flat[base+7] - centroids[cb+7]
					d += d7 * d7
					if d >= bestD {
						continue
					}
					d8 := flat[base+8] - centroids[cb+8]
					d += d8 * d8
					if d >= bestD {
						continue
					}
					d9 := flat[base+9] - centroids[cb+9]
					d += d9 * d9
					if d >= bestD {
						continue
					}
					d10 := flat[base+10] - centroids[cb+10]
					d += d10 * d10
					if d >= bestD {
						continue
					}
					d11 := flat[base+11] - centroids[cb+11]
					d += d11 * d11
					if d >= bestD {
						continue
					}
					d12 := flat[base+12] - centroids[cb+12]
					d += d12 * d12
					if d >= bestD {
						continue
					}
					d13 := flat[base+13] - centroids[cb+13]
					d += d13 * d13
					if d < bestD {
						bestD = d
						best = ci
					}
				}
				if assign[i] != best {
					assign[i] = best
					local++
				}
			}
			changedPerWorker[w] = local
		}(w)
	}
	wg.Wait()

	total := 0
	for _, c := range changedPerWorker {
		total += c
	}
	return total
}

func kmeansUpdate(flat []float32, n int, centroids []float32, k int, assign []int) {
	workers := runtime.GOMAXPROCS(0)
	chunkSize := (n + workers - 1) / workers

	type accumulator struct {
		sums   []float64
		counts []int
	}
	accs := make([]accumulator, workers)
	for w := range accs {
		accs[w] = accumulator{
			sums:   make([]float64, k*14),
			counts: make([]int, k),
		}
	}

	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			lo := w * chunkSize
			hi := lo + chunkSize
			if hi > n {
				hi = n
			}
			acc := &accs[w]
			for i := lo; i < hi; i++ {
				ci := assign[i]
				acc.counts[ci]++
				base := i * 14
				cb := ci * 14
				for d := 0; d < 14; d++ {
					acc.sums[cb+d] += float64(flat[base+d])
				}
			}
		}(w)
	}
	wg.Wait()

	sums := accs[0].sums
	counts := accs[0].counts
	for w := 1; w < workers; w++ {
		for j := range sums {
			sums[j] += accs[w].sums[j]
		}
		for j := range counts {
			counts[j] += accs[w].counts[j]
		}
	}

	for ci := 0; ci < k; ci++ {
		if counts[ci] == 0 {
			continue
		}
		cb := ci * 14
		for d := 0; d < 14; d++ {
			centroids[cb+d] = float32(sums[cb+d] / float64(counts[ci]))
		}
	}
}

func BuildBrute(refsGz, outPath string) error {
	fmt.Printf("BuildBrute: loading %s ...\n", refsGz)

	f, err := os.Open(refsGz)
	if err != nil {
		return err
	}
	defer f.Close()
	gz, err := gzip.NewReader(f)
	if err != nil {
		return err
	}
	defer gz.Close()

	var vectors []float32
	var fraudFlags []bool

	dec := json.NewDecoder(gz)
	if _, err := dec.Token(); err != nil {
		return err
	}
	var entry struct {
		Vector []float64 `json:"vector"`
		Label  string    `json:"label"`
	}
	for dec.More() {
		entry.Vector = entry.Vector[:0]
		if err := dec.Decode(&entry); err != nil {
			break
		}
		for i := 0; i < 14; i++ {
			if i < len(entry.Vector) {
				vectors = append(vectors, float32(entry.Vector[i]))
			} else {
				vectors = append(vectors, 0)
			}
		}
		fraudFlags = append(fraudFlags, entry.Label == "fraud")
	}
	N := len(fraudFlags)
	fmt.Printf("BuildBrute: loaded %d vectors\n", N)

	soa := make([]int16, N*DIM)
	labels := make([]byte, N)

	for i := 0; i < N; i++ {
		for d := 0; d < DIM; d++ {
			soa[d*N+i] = quantizeFloat32(vectors[i*DIM+d])
		}
		if fraudFlags[i] {
			labels[i] = 1
		}
	}

	fmt.Printf("BuildBrute: writing %s ...\n", outPath)
	out, err := os.Create(outPath)
	if err != nil {
		return err
	}
	defer out.Close()

	write32 := func(v uint32) { binary.Write(out, binary.LittleEndian, v) }
	write32(bruteMagic)
	write32(1)
	write32(uint32(N))
	write32(DIM)
	binary.Write(out, binary.LittleEndian, soa)
	out.Write(labels)

	fmt.Printf("BuildBrute: done. N=%d\n", N)
	return nil
}
