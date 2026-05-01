// Package knn implements k-nearest-neighbor search over 14-dimensional float32 vectors.
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

const K = 5

// ─── Brute-force index (used only in tests / small datasets) ─────────────────

// BruteIndex holds all reference vectors in a flat, cache-friendly layout.
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
// IVFIndex stores vectors grouped by cluster. At query time only the nearest
// nprobe clusters are searched, reducing work from 3M to ~nprobe*clusterSize.
//
// Binary file format (little-endian):
//   [4]  magic   uint32 = 0x49564649 ("IVFI")
//   [4]  version uint32 = 1
//   [4]  nlist   uint32  (number of clusters)
//   [4]  dim     uint32  (must be 14)
//   nlist * 14 * 4  centroids float32
//   for each cluster:
//     [4]  n      uint32  (vectors in cluster)
//     n * 14 * 4  vectors float32
//     n * 1       labels  bool (1 byte each)

const ivfMagic uint32 = 0x49564649

// cdist pairs a centroid index with its squared L2 distance from the query.
type cdist struct {
	dist float32
	ci   int
}

// ivfClusterDesc describes a cluster's position inside the arena.
// No pointer fields — GC does not scan this struct.
type ivfClusterDesc struct {
	flatOff  uint32 // byte offset of float32 flat data in arena (n*14*4 bytes)
	labelOff uint32 // byte offset of label bytes in arena (n bytes)
	n        uint32 // number of vectors in cluster
}

// IVFIndex implements fast approximate KNN via inverted file (cluster-based).
//
// All vector data is stored in a single []byte arena to minimise GC scan cost.
// Centroids are kept as float32 for query-time precision; cluster vectors are
// also stored as float32 (full precision) to preserve detection accuracy.
//
// Arena layout per cluster i:
//   [descs[i].flatOff  .. +n*14*4)  float32 vectors (little-endian)
//   [descs[i].labelOff .. +n)       uint8 labels (1=fraud, 0=legit)
//
// GC visibility: arena (1 pointer) + centroids (1 pointer) + descs (1 pointer)
// = 3 slice headers regardless of nlist or total vector count.
type IVFIndex struct {
	// centroids holds nlist*14 float32 values.
	centroids []float32
	// descs is a pointer-free slice — GC skips its contents entirely.
	descs []ivfClusterDesc
	// arena holds all float32 cluster data followed by label bytes.
	arena []byte
	nlist  int
	nprobe int
}

// SetNProbe sets the number of nearest clusters to search during Predict.
func (idx *IVFIndex) SetNProbe(n int) {
	if n < 1 {
		n = 1
	}
	idx.nprobe = n
}

// NewIVFIndex creates an empty IVFIndex.
func NewIVFIndex() *IVFIndex { return &IVFIndex{} }

// LoadIVF loads a pre-built IVF index from a binary file.
// Cluster vectors (float32) are stored as-is in the arena for full precision.
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
	if err := binary.Read(f, binary.LittleEndian, &version); err != nil || version != 1 {
		return nil, fmt.Errorf("unsupported ivf version %d", version)
	}
	binary.Read(f, binary.LittleEndian, &nlist)
	binary.Read(f, binary.LittleEndian, &dim)
	if dim != 14 {
		return nil, fmt.Errorf("expected dim=14, got %d", dim)
	}

	// Read centroids as float32.
	centroids := make([]float32, nlist*dim)
	if err := binary.Read(f, binary.LittleEndian, centroids); err != nil {
		return nil, fmt.Errorf("read centroids: %w", err)
	}

	// First pass: read all cluster sizes to compute arena layout.
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
		// Skip float32 flat + label bytes.
		skip := int64(n)*int64(dim)*4 + int64(n)
		if _, err := f.Seek(skip, io.SeekCurrent); err != nil {
			return nil, fmt.Errorf("seek cluster %d: %w", i, err)
		}
	}

	// Allocate arena: for each cluster, n*14*4 (float32) + n (labels).
	arenaSize := totalVecs*uint64(dim)*4 + totalVecs
	arena := make([]byte, arenaSize)

	descs := make([]ivfClusterDesc, nlist)
	var arenaPos uint64

	for i := uint32(0); i < nlist; i++ {
		n := metas[i].n
		flatBytes := uint64(n) * uint64(dim) * 4
		labelBytes := uint64(n)

		descs[i] = ivfClusterDesc{
			flatOff:  uint32(arenaPos),
			labelOff: uint32(arenaPos + flatBytes),
			n:        n,
		}

		// Seek to this cluster's float data (past the n uint32 already consumed in pass 1).
		if _, err := f.Seek(metas[i].fileOffset, io.SeekStart); err != nil {
			return nil, fmt.Errorf("seek cluster %d data: %w", i, err)
		}

		// Read float32 vectors directly into arena (no quantization).
		if _, err := io.ReadFull(f, arena[arenaPos:arenaPos+flatBytes]); err != nil {
			return nil, fmt.Errorf("read cluster %d vecs: %w", i, err)
		}

		// Read label bytes into arena.
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

// Predict searches nprobe nearest clusters and returns fraud probability.
func (idx *IVFIndex) Predict(query model.Vector14, k int) float64 {
	// Find nearest centroid.
	bestCI := 0
	bestDist := float32(math.MaxFloat32)
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
		if d < bestDist {
			bestDist = d
			bestCI = ci
		}
	}

	// Brute-force search in the nearest cluster (float32 arena, zero alloc).
	desc := idx.descs[bestCI]
	n := int(desc.n)
	if n == 0 {
		return 0
	}
	f := unsafe.Slice((*float32)(unsafe.Pointer(&idx.arena[desc.flatOff])), n*14)
	labels := idx.arena[desc.labelOff : desc.labelOff+desc.n]

	var topK [K]candidate
	count := 0

	for i := 0; i < n; i++ {
		base := i * 14
		e0 := query[0] - f[base+0]
		e1 := query[1] - f[base+1]
		e2 := query[2] - f[base+2]
		e3 := query[3] - f[base+3]
		e4 := query[4] - f[base+4]
		e5 := query[5] - f[base+5]
		e6 := query[6] - f[base+6]
		e7 := query[7] - f[base+7]
		e8 := query[8] - f[base+8]
		e9 := query[9] - f[base+9]
		e10 := query[10] - f[base+10]
		e11 := query[11] - f[base+11]
		e12 := query[12] - f[base+12]
		e13 := query[13] - f[base+13]
		sum := e0*e0 + e1*e1 + e2*e2 + e3*e3 + e4*e4 + e5*e5 + e6*e6 +
			e7*e7 + e8*e8 + e9*e9 + e10*e10 + e11*e11 + e12*e12 + e13*e13
		if count < K {
			topK[count] = candidate{sum, i}
			count++
			if count == K {
				maxI := 0
				if topK[1].dist > topK[maxI].dist { maxI = 1 }
				if topK[2].dist > topK[maxI].dist { maxI = 2 }
				if topK[3].dist > topK[maxI].dist { maxI = 3 }
				if topK[4].dist > topK[maxI].dist { maxI = 4 }
				topK[0], topK[maxI] = topK[maxI], topK[0]
			}
		} else if sum < topK[0].dist {
			topK[0] = candidate{sum, i}
			maxI := 0
			if topK[1].dist > topK[maxI].dist { maxI = 1 }
			if topK[2].dist > topK[maxI].dist { maxI = 2 }
			if topK[3].dist > topK[maxI].dist { maxI = 3 }
			if topK[4].dist > topK[maxI].dist { maxI = 4 }
			topK[0], topK[maxI] = topK[maxI], topK[0]
		}
	}

	// Count fraud in top-K.
	fraudCount := 0
	for j := 0; j < count; j++ {
		if labels[topK[j].idx] == 1 {
			fraudCount++
		}
	}

	if count == 0 {
		return 0
	}
	return float64(fraudCount) / float64(count)
}

// Count returns total vectors across all clusters.
func (idx *IVFIndex) Count() int {
	n := 0
	for _, d := range idx.descs {
		n += int(d.n)
	}
	return n
}

// FraudCount returns fraud vectors across all clusters.
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

// BuildIVF runs k-means on the references.json.gz and writes an IVF index file.
// nlist: number of clusters (e.g. 500). iterations: k-means iterations (e.g. 20).
func BuildIVF(refsGz, outPath string, nlist, iterations int) error {
	fmt.Printf("BuildIVF: loading %s ...\n", refsGz)

	// Load all vectors.
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

	// K-means initialization (random — k-means++ is O(N*K²) and too slow for 3M vectors).
	centroids := kmeansInit(flat, n, nlist)

	// K-means iterations.
	assign := make([]int, n)
	for iter := 0; iter < iterations; iter++ {
		changed := kmeansAssign(flat, n, centroids, nlist, assign)
		kmeansUpdate(flat, n, centroids, nlist, assign)
		fmt.Printf("  iter %d: %d reassigned\n", iter+1, changed)
		if changed == 0 {
			break
		}
	}

	// Group vectors by cluster.
	clusterSizes := make([]int, nlist)
	for _, ci := range assign {
		clusterSizes[ci]++
	}
	clusterFlat := make([][]float32, nlist)
	clusterLabels := make([][]byte, nlist)
	for ci := range clusterFlat {
		clusterFlat[ci] = make([]float32, 0, clusterSizes[ci]*14)
		clusterLabels[ci] = make([]byte, 0, clusterSizes[ci])
	}
	for i, ci := range assign {
		clusterFlat[ci] = append(clusterFlat[ci], flat[i*14:i*14+14]...)
		lb := byte(0)
		if fraudFlags[i] {
			lb = 1
		}
		clusterLabels[ci] = append(clusterLabels[ci], lb)
	}

	// Write binary file.
	fmt.Printf("BuildIVF: writing %s ...\n", outPath)
	out, err := os.Create(outPath)
	if err != nil {
		return err
	}
	defer out.Close()

	write32 := func(v uint32) { binary.Write(out, binary.LittleEndian, v) }
	write32(ivfMagic)
	write32(1) // version
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

	// Each worker accumulates into its own sums/counts to avoid contention.
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

	// Merge accumulators.
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
