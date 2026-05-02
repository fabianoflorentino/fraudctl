// Package knn implements k-nearest-neighbor search over 14-dimensional vectors.
package knn

import (
	"compress/gzip"
	"container/heap"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sync"
	"unsafe"

	"github.com/fabianoflorentino/fraudctl/internal/model"
)

const K = 51

const int16Scale = 10000

// ─── Brute-force index (used only in tests / small datasets) ─────────────────

type BruteIndex struct {
	flat       []float32
	fraudFlags []bool
	count      int
}

func NewBruteIndex() *BruteIndex { return &BruteIndex{} }

func (b *BruteIndex) Build(vectors []model.Vector14, labels []bool) {
	n := len(vectors)
	b.count = n
	b.fraudFlags = labels
	b.flat = make([]float32, n*14)
	for i, v := range vectors {
		copy(b.flat[i*14:], v[:])
	}
}

func (b *BruteIndex) BuildFromGzip(path string, capacity int) error {
	f, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("open gzip: %w", err)
	}
	defer f.Close()
	gz, err := gzip.NewReader(f)
	if err != nil {
		return fmt.Errorf("gzip reader: %w", err)
	}
	defer gz.Close()

	b.flat = make([]float32, 0, capacity*14)
	b.fraudFlags = make([]bool, 0, capacity)

	dec := json.NewDecoder(gz)
	if _, err := dec.Token(); err != nil {
		return fmt.Errorf("json open bracket: %w", err)
	}

	var entry struct {
		Vector []float64 `json:"vector"`
		Label  string    `json:"label"`
	}
	for dec.More() {
		entry.Vector = entry.Vector[:0]
		if err := dec.Decode(&entry); err != nil {
			if err == io.EOF {
				break
			}
			return fmt.Errorf("json decode: %w", err)
		}
		for i := 0; i < 14; i++ {
			if i < len(entry.Vector) {
				b.flat = append(b.flat, float32(entry.Vector[i]))
			} else {
				b.flat = append(b.flat, 0)
			}
		}
		b.fraudFlags = append(b.fraudFlags, entry.Label == "fraud")
	}
	b.count = len(b.fraudFlags)
	return nil
}

func (b *BruteIndex) Predict(query model.Vector14, k int) float64 {
	return brutePredict(b.flat, b.fraudFlags, b.count, query, k)
}

func (b *BruteIndex) Count() int { return b.count }

func (b *BruteIndex) FraudCount() int {
	n := 0
	for _, f := range b.fraudFlags {
		if f {
			n++
		}
	}
	return n
}

// ─── IVF index (Inverted File — cluster-based ANN) ───────────────────────────
//
// Binary file format v2 (little-endian):
//   [4]  magic   uint32 = 0x49564649 ("IVFI")
//   [4]  version uint32 = 2
//   [4]  nlist   uint32
//   [4]  dim     uint32  (=14)
//   nlist * 14 * 4  centroids float32
//   for each cluster:
//     [4]  n      uint32
//     n * 14 * 2  vectors int16 (quantized, scale=10000)
//     n * 1       labels  uint8

const ivfMagic uint32 = 0x49564649

type cdist struct {
	dist float32
	ci   int
}

type ivfClusterDesc struct {
	flatOff  uint32
	labelOff uint32
	n        uint32
}

type IVFIndex struct {
	centroids []float32
	descs     []ivfClusterDesc
	arena     []byte
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

func quantizeFloat32(v float32) int16 {
	if v > 1.0 {
		return int16Scale
	}
	if v < -1.0 {
		return -int16Scale
	}
	return int16(math.Round(float64(v * int16Scale)))
}

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
	if err := binary.Read(f, binary.LittleEndian, &version); err != nil || version != 2 {
		return nil, fmt.Errorf("unsupported ivf version %d (expected 2)", version)
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
		fileOffset int64
	}
	metas := make([]clusterMeta, nlist)
	totalVecs := uint64(0)
	for i := uint32(0); i < nlist; i++ {
		var n uint32
		if err := binary.Read(f, binary.LittleEndian, &n); err != nil {
			return nil, fmt.Errorf("read cluster size %d: %w", i, err)
		}
		off, _ := f.Seek(0, io.SeekCurrent)
		metas[i] = clusterMeta{n: n, fileOffset: off}
		totalVecs += uint64(n)
		skip := int64(n)*int64(dim)*2 + int64(n)
		if _, err := f.Seek(skip, io.SeekCurrent); err != nil {
			return nil, fmt.Errorf("seek cluster %d: %w", i, err)
		}
	}

	arenaSize := totalVecs*uint64(dim)*2 + totalVecs
	arena := make([]byte, arenaSize)

	descs := make([]ivfClusterDesc, nlist)
	var arenaPos uint64

	for i := uint32(0); i < nlist; i++ {
		n := metas[i].n
		flatBytes := uint64(n) * uint64(dim) * 2
		labelBytes := uint64(n)

		descs[i] = ivfClusterDesc{
			flatOff:  uint32(arenaPos),
			labelOff: uint32(arenaPos + flatBytes),
			n:        n,
		}

		if _, err := f.Seek(metas[i].fileOffset, io.SeekStart); err != nil {
			return nil, fmt.Errorf("seek cluster %d data: %w", i, err)
		}

		if _, err := io.ReadFull(f, arena[arenaPos:arenaPos+flatBytes]); err != nil {
			return nil, fmt.Errorf("read cluster %d vecs: %w", i, err)
		}

		if _, err := io.ReadFull(f, arena[arenaPos+flatBytes:arenaPos+flatBytes+labelBytes]); err != nil {
			return nil, fmt.Errorf("read cluster %d labels: %w", i, err)
		}

		arenaPos += flatBytes + labelBytes
	}

	return &IVFIndex{
		nlist:     int(nlist),
		nprobe:    1,
		centroids: centroids,
		descs:     descs,
		arena:     arena,
	}, nil
}

func quantizeQuery(q model.Vector14) [14]int32 {
	var qi [14]int32
	for d := 0; d < 14; d++ {
		if q[d] < -1.0 {
			qi[d] = -int16Scale
		} else if q[d] > 1.0 {
			qi[d] = int16Scale
		} else {
			qi[d] = int32(math.Round(float64(q[d] * int16Scale)))
		}
	}
	return qi
}

// findTopCentroids finds the top-n nearest centroids using a stack-allocated max-heap.
// Returns the number of probes filled.
func (idx *IVFIndex) findTopCentroids(query model.Vector14, probes *[32]cdist, nprobe int) int {
	bestCount := 0
	c := idx.centroids

	for ci := 0; ci < idx.nlist; ci++ {
		base := ci * 14
		d0 := query[0] - c[base+0]
		d1 := query[1] - c[base+1]
		d2 := query[2] - c[base+2]
		d3 := query[3] - c[base+3]
		d4 := query[4] - c[base+4]
		d5 := query[5] - c[base+5]
		d6 := query[6] - c[base+6]
		d7 := query[7] - c[base+7]
		d8 := query[8] - c[base+8]
		d9 := query[9] - c[base+9]
		d10 := query[10] - c[base+10]
		d11 := query[11] - c[base+11]
		d12 := query[12] - c[base+12]
		d13 := query[13] - c[base+13]
		d := d0*d0 + d1*d1 + d2*d2 + d3*d3 + d4*d4 + d5*d5 + d6*d6 +
			d7*d7 + d8*d8 + d9*d9 + d10*d10 + d11*d11 + d12*d12 + d13*d13

		if bestCount < nprobe {
			probes[bestCount] = cdist{d, ci}
			bestCount++
			if bestCount == nprobe {
				worstIdx := 0
				for j := 1; j < nprobe; j++ {
					if probes[j].dist > probes[worstIdx].dist {
						worstIdx = j
					}
				}
				probes[0], probes[worstIdx] = probes[worstIdx], probes[0]
			}
		} else if d < probes[0].dist {
			probes[0] = cdist{d, ci}
			worstIdx := 0
			for j := 1; j < nprobe; j++ {
				if probes[j].dist > probes[worstIdx].dist {
					worstIdx = j
				}
			}
			probes[0], probes[worstIdx] = probes[worstIdx], probes[0]
		}
	}
	return bestCount
}

type labeledCandidate struct {
	dist  uint64
	label byte
}

func (idx *IVFIndex) searchClusters(probes []cdist, qi [14]int32) ([K]labeledCandidate, int) {
	var topK [K]labeledCandidate
	count := 0
	worstDist := uint64(math.MaxUint64)

	for pi := range probes {
		desc := idx.descs[probes[pi].ci]
		n := int(desc.n)
		if n == 0 {
			continue
		}
		f := unsafe.Slice((*int16)(unsafe.Pointer(&idx.arena[desc.flatOff])), n*14)
		labels := idx.arena[desc.labelOff : desc.labelOff+desc.n]

		for i := 0; i < n; i++ {
			base := i * 14
			v0 := int64(f[base+0]) - int64(qi[0])
			v1 := int64(f[base+1]) - int64(qi[1])
			v2 := int64(f[base+2]) - int64(qi[2])
			v3 := int64(f[base+3]) - int64(qi[3])
			v4 := int64(f[base+4]) - int64(qi[4])
			v5 := int64(f[base+5]) - int64(qi[5])
			v6 := int64(f[base+6]) - int64(qi[6])
			v7 := int64(f[base+7]) - int64(qi[7])
			dist := uint64(v0*v0) + uint64(v1*v1) + uint64(v2*v2) + uint64(v3*v3) +
				uint64(v4*v4) + uint64(v5*v5) + uint64(v6*v6) + uint64(v7*v7)

			if dist >= worstDist {
				continue
			}

			v8 := int64(f[base+8]) - int64(qi[8])
			v9 := int64(f[base+9]) - int64(qi[9])
			v10 := int64(f[base+10]) - int64(qi[10])
			v11 := int64(f[base+11]) - int64(qi[11])
			v12 := int64(f[base+12]) - int64(qi[12])
			v13 := int64(f[base+13]) - int64(qi[13])
			dist += uint64(v8*v8) + uint64(v9*v9) + uint64(v10*v10) +
				uint64(v11*v11) + uint64(v12*v12) + uint64(v13*v13)

			if dist >= worstDist {
				continue
			}

			lb := labels[i]
			if count < K {
				topK[count] = labeledCandidate{dist, lb}
				count++
				if count == K {
					maxI := 0
					for j := 1; j < K; j++ {
						if topK[j].dist > topK[maxI].dist {
							maxI = j
						}
					}
					topK[0], topK[maxI] = topK[maxI], topK[0]
					worstDist = topK[0].dist
				}
			} else {
				topK[0] = labeledCandidate{dist, lb}
				maxI := 0
				for j := 1; j < K; j++ {
					if topK[j].dist > topK[maxI].dist {
						maxI = j
					}
				}
				topK[0], topK[maxI] = topK[maxI], topK[0]
				worstDist = topK[0].dist
			}
		}
	}
	return topK, count
}

func countFraud(topK [K]labeledCandidate, count int) int {
	n := 0
	for j := 0; j < count; j++ {
		if topK[j].label == 1 {
			n++
		}
	}
	return n
}

// Predict searches the nprobe nearest clusters and returns fraud probability.
// Uses adaptive nprobe: doubles when fraud_count is ambiguous (2 or 3).
func (idx *IVFIndex) Predict(query model.Vector14, k int) float64 {
	baseNprobe := idx.nprobe
	if baseNprobe > idx.nlist {
		baseNprobe = idx.nlist
	}

	qi := quantizeQuery(query)

	var probes [32]cdist
	bestCount := idx.findTopCentroids(query, &probes, baseNprobe)

	topK, count := idx.searchClusters(probes[:bestCount], qi)
	fraudCount := countFraud(topK, count)

	// Adaptive: if ambiguous (16-36 out of 51), double the nprobe.
	expanded := baseNprobe * 2
	if fraudCount >= 16 && fraudCount <= 36 && expanded <= 32 && expanded <= idx.nlist {
		expCount := idx.findTopCentroids(query, &probes, expanded)
		topK2, count2 := idx.searchClusters(probes[:expCount], qi)
		fraudCount2 := countFraud(topK2, count2)
		if count2 > 0 {
			return float64(fraudCount2) / float64(count2)
		}
	}

	if count == 0 {
		return 0
	}
	return float64(fraudCount) / float64(count)
}

func (idx *IVFIndex) Count() int {
	n := 0
	for _, d := range idx.descs {
		n += int(d.n)
	}
	return n
}

func (idx *IVFIndex) FraudCount() int {
	n := 0
	for _, d := range idx.descs {
		labels := idx.arena[d.labelOff : d.labelOff+d.n]
		for _, lb := range labels {
			if lb == 1 {
				n++
			}
		}
	}
	return n
}

// ─── IVF build (k-means) — runs at docker build time ─────────────────────────

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
	clusterFlat := make([][]int16, nlist)
	clusterLabels := make([][]byte, nlist)
	for ci := range clusterFlat {
		clusterFlat[ci] = make([]int16, 0, clusterSizes[ci]*14)
		clusterLabels[ci] = make([]byte, 0, clusterSizes[ci])
	}
	for i, ci := range assign {
		for d := 0; d < 14; d++ {
			clusterFlat[ci] = append(clusterFlat[ci], quantizeFloat32(flat[i*14+d]))
		}
		lb := byte(0)
		if fraudFlags[i] {
			lb = 1
		}
		clusterLabels[ci] = append(clusterLabels[ci], lb)
	}

	fmt.Printf("BuildIVF: writing %s (v2, int16) ...\n", outPath)
	out, err := os.Create(outPath)
	if err != nil {
		return err
	}
	defer out.Close()

	write32 := func(v uint32) { binary.Write(out, binary.LittleEndian, v) }
	write32(ivfMagic)
	write32(2) // version 2: int16 vectors
	write32(uint32(nlist))
	write32(14)

	binary.Write(out, binary.LittleEndian, centroids)

	for ci := 0; ci < nlist; ci++ {
		write32(uint32(len(clusterLabels[ci])))
		binary.Write(out, binary.LittleEndian, clusterFlat[ci])
		out.Write(clusterLabels[ci])
	}

	fmt.Printf("BuildIVF: done. Cluster sizes: avg=%d\n", n/nlist)
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

// ─── shared helpers ───────────────────────────────────────────────────────────

type candidate struct {
	dist float32
	idx  int
}

type maxHeap []candidate

func (h maxHeap) Len() int            { return len(h) }
func (h maxHeap) Less(i, j int) bool  { return h[i].dist > h[j].dist }
func (h maxHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *maxHeap) Push(x interface{}) { *h = append(*h, x.(candidate)) }
func (h *maxHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

func brutePredict(flat []float32, fraudFlags []bool, n int, query model.Vector14, k int) float64 {
	h := make(maxHeap, 0, k+1)
	for i := 0; i < n; i++ {
		base := i * 14
		var sum float32
		for d := 0; d < 14; d++ {
			diff := query[d] - flat[base+d]
			sum += diff * diff
		}
		if len(h) < k {
			heap.Push(&h, candidate{sum, i})
		} else if sum < h[0].dist {
			h[0] = candidate{sum, i}
			heap.Fix(&h, 0)
		}
	}
	fraudCount := 0
	for _, c := range h {
		if c.idx < len(fraudFlags) && fraudFlags[c.idx] {
			fraudCount++
		}
	}
	if len(h) == 0 {
		return 0
	}
	return float64(fraudCount) / float64(len(h))
}
