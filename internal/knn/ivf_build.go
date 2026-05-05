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

	clusterSizes := make([]int, nlist)
	for _, ci := range assign {
		clusterSizes[ci]++
	}

	nBlocksTotal := 0
	for _, sz := range clusterSizes {
		nBlocksTotal += (sz + 7) / 8
	}

	soaBlocks := make([]int16, nBlocksTotal*14*8)
	soaLabels := make([]byte, nBlocksTotal*8)

	blockOffsets := make([]uint32, nlist+1)
	var blockPos int
	for ci := 0; ci < nlist; ci++ {
		blockOffsets[ci] = uint32(blockPos)
		sz := clusterSizes[ci]
		nBlocks := (sz + 7) / 8

		vecIndices := make([]int, 0, sz)
		for i, a := range assign {
			if a == ci {
				vecIndices = append(vecIndices, i)
			}
		}

		for bi := 0; bi < nBlocks; bi++ {
			blockBase := (blockPos + bi) * 14 * 8
			labelBase := (blockPos + bi) * 8

			for d := 0; d < 14; d++ {
				for s := 0; s < 8; s++ {
					soaBlocks[blockBase+d*8+s] = int16Pad
				}
			}

			vecStart := bi * 8
			vecEnd := vecStart + 8
			if vecEnd > sz {
				vecEnd = sz
			}
			for si := vecStart; si < vecEnd; si++ {
				vi := vecIndices[si]
				slot := si - bi*8
				for d := 0; d < 14; d++ {
					soaBlocks[blockBase+d*8+slot] = quantizeFloat32(flat[vi*14+d])
				}
				soaLabels[labelBase+slot] = 0
				if fraudFlags[vi] {
					soaLabels[labelBase+slot] = 1
				}
			}
		}
		blockPos += nBlocks
	}
	blockOffsets[nlist] = uint32(blockPos)

	fmt.Printf("BuildIVF: writing %s (v3, SoA blocks) ...\n", outPath)
	out, err := os.Create(outPath)
	if err != nil {
		return err
	}
	defer out.Close()

	write32 := func(v uint32) { binary.Write(out, binary.LittleEndian, v) }
	write32(ivfMagic)
	write32(3)
	write32(uint32(nlist))
	write32(14)

	binary.Write(out, binary.LittleEndian, centroids)

	for ci := 0; ci < nlist; ci++ {
		sz := clusterSizes[ci]
		write32(uint32(sz))
		nBlocks := blockOffsets[ci+1] - blockOffsets[ci]
		blockBytes := nBlocks * 14 * 8 * 2
		labelBytes := nBlocks * 8
		binary.Write(out, binary.LittleEndian, soaBlocks[blockOffsets[ci]*14*8:blockOffsets[ci]*14*8+blockBytes/2])
		out.Write(soaLabels[blockOffsets[ci]*8 : blockOffsets[ci]*8+labelBytes])
	}

	fmt.Printf("BuildIVF: done. Blocks: %d\n", blockPos)
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
					var d float32
					for dim := 0; dim < 14; dim++ {
						diff := flat[base+dim] - centroids[cb+dim]
						d += diff * diff
					}
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
