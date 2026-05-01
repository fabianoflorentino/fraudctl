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

// IVFIndex implements fast approximate KNN via inverted file (cluster-based).
type IVFIndex struct {
	centroids []float32 // nlist * 14
	clusters  []ivfCluster
	nlist     int
}

type ivfCluster struct {
	flat   []float32
	labels []bool
}

// NewIVFIndex creates an empty IVFIndex.
func NewIVFIndex() *IVFIndex { return &IVFIndex{} }

// LoadIVF loads a pre-built IVF index from a binary file.
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

	idx := &IVFIndex{
		nlist:     int(nlist),
		centroids: make([]float32, nlist*dim),
		clusters:  make([]ivfCluster, nlist),
	}

	if err := binary.Read(f, binary.LittleEndian, idx.centroids); err != nil {
		return nil, fmt.Errorf("read centroids: %w", err)
	}

	for i := uint32(0); i < nlist; i++ {
		var n uint32
		if err := binary.Read(f, binary.LittleEndian, &n); err != nil {
			return nil, fmt.Errorf("read cluster size: %w", err)
		}
		c := ivfCluster{
			flat:   make([]float32, n*dim),
			labels: make([]bool, n),
		}
		if err := binary.Read(f, binary.LittleEndian, c.flat); err != nil {
			return nil, fmt.Errorf("read cluster flat: %w", err)
		}
		labelBytes := make([]byte, n)
		if _, err := io.ReadFull(f, labelBytes); err != nil {
			return nil, fmt.Errorf("read cluster labels: %w", err)
		}
		for j, lb := range labelBytes {
			c.labels[j] = lb == 1
		}
		idx.clusters[i] = c
	}

	return idx, nil
}

// Predict searches nprobe nearest clusters and returns fraud probability.
func (idx *IVFIndex) Predict(query model.Vector14, k int) float64 {
	const nprobe = 1

	// Find nprobe nearest centroids.
	type cdist struct {
		dist float32
		ci   int
	}
	best := make([]cdist, 0, nprobe+1)

	for ci := 0; ci < idx.nlist; ci++ {
		base := ci * 14
		var d float32
		for j := 0; j < 14; j++ {
			diff := query[j] - idx.centroids[base+j]
			d += diff * diff
		}
		if len(best) < nprobe {
			best = append(best, cdist{d, ci})
		} else {
			// find max
			maxI := 0
			for bi := 1; bi < len(best); bi++ {
				if best[bi].dist > best[maxI].dist {
					maxI = bi
				}
			}
			if d < best[maxI].dist {
				best[maxI] = cdist{d, ci}
			}
		}
	}

	// Brute-force search in selected clusters.
	h := make(maxHeap, 0, k+1)
	for _, cd := range best {
		c := idx.clusters[cd.ci]
		n := len(c.labels)
		for i := 0; i < n; i++ {
			base := i * 14
			var sum float32
			for d := 0; d < 14; d++ {
				diff := query[d] - c.flat[base+d]
				sum += diff * diff
			}
			if len(h) < k {
				heap.Push(&h, candidate{sum, i*idx.nlist + cd.ci})
			} else if sum < h[0].dist {
				h[0] = candidate{sum, i*idx.nlist + cd.ci}
				heap.Fix(&h, 0)
			}
		}
	}

	// Map back to fraud flags.
	fraudCount := 0
	for _, c := range h {
		ci := c.idx % idx.nlist
		vi := c.idx / idx.nlist
		if vi < len(idx.clusters[ci].labels) && idx.clusters[ci].labels[vi] {
			fraudCount++
		}
	}

	if len(h) == 0 {
		return 0
	}
	return float64(fraudCount) / float64(len(h))
}

// Count returns total vectors across all clusters.
func (idx *IVFIndex) Count() int {
	n := 0
	for _, c := range idx.clusters {
		n += len(c.labels)
	}
	return n
}

// FraudCount returns fraud vectors across all clusters.
func (idx *IVFIndex) FraudCount() int {
	n := 0
	for _, c := range idx.clusters {
		for _, f := range c.labels {
			if f {
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
