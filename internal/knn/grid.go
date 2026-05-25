package knn

import (
	"encoding/binary"
	"fmt"
	"os"
	"syscall"
	"unsafe"

	"github.com/fabianoflorentino/fraudctl/internal/model"
)

const gridMagic = uint32(0x4B445452)
const gridVersion = uint32(2)

var hammingOrder [256][256]uint8

type hammingEntry struct {
	key  uint8
	dist uint8
}

func init() {
	var entries [256]hammingEntry
	for i := 0; i < 256; i++ {
		for j := 0; j < 256; j++ {
			entries[j] = hammingEntry{key: uint8(j), dist: popcount8(uint8(i) ^ uint8(j))}
		}
		sortEntries8(entries[:])
		for j := 0; j < 256; j++ {
			hammingOrder[i][j] = entries[j].key
		}
	}
}

func popcount8(x uint8) uint8 {
	x = x - ((x >> 1) & 0x55)
	x = (x & 0x33) + ((x >> 2) & 0x33)
	return (x + (x >> 4)) & 0x0F
}

func sortEntries8(e []hammingEntry) {
	for i := 1; i < len(e); i++ {
		for j := i; j > 0 && e[j].dist < e[j-1].dist; j-- {
			e[j], e[j-1] = e[j-1], e[j]
		}
	}
}

// GridIndex is an exact KNN index using a KD-tree with 8-bit partition keys.
// It stores 3M vectors in block-major layout for AVX2-accelerated leaf scanning.
// Partition lookup is O(1) via a direct-indexed array.
type GridIndex struct {
	n               int
	nodes           []gridNode
	partitions      []gridPartition
	partitionsByKey [256]int32
	existingKeys    [256]uint8
	nExisting       int
	hammingExisting [256][]uint8 // existing keys in Hamming order per query key
	vectors         []int16
	labels          []byte
	mmapData        []byte // holds mmap'd region for munmap on Close
}

type gridNode struct {
	left, right int32
	blockStart  uint32
	blockLen    uint32
	vecOffset   uint32
	bboxMin     [DIM]int16
	bboxMax     [DIM]int16
}

type gridPartition struct {
	key        uint32
	root       int32
	bboxMin    [DIM]int16
	bboxMax    [DIM]int16
	bboxMinF32 [DIM]float32
	bboxMaxF32 [DIM]float32
}

type gridScratch struct {
	stackNodes  [64]int32
	stackBounds [64]uint64
	blockSum    [8]float32
}

// Count returns the total number of vectors in the index.
func (idx *GridIndex) Count() int { return idx.n }

// FraudCount returns the total number of fraud-labeled vectors in the index.
func (idx *GridIndex) FraudCount() int {
	n := 0
	for i := 0; i < idx.n; i++ {
		if (idx.labels[i>>3] & (1 << uint(i&7))) != 0 {
			n++
		}
	}
	return n
}

// Predict returns the fraud score (fraudCount/K) for the given query vector.
func (idx *GridIndex) Predict(query model.Vector14, k int) float64 {
	return float64(idx.PredictRaw(query, 0)) / float64(K)
}

// NProbe returns 0 since the GridIndex does not use adaptive probing.
func (idx *GridIndex) NProbe() int { return 0 }

// PredictRaw returns the count of fraudulent neighbors among the K=5 nearest
// neighbors. The nprobe parameter is ignored (GridIndex is exact).
func (idx *GridIndex) PredictRaw(query model.Vector14, _ int) int {
	qi := quantizeQuery(query)
	key := gridPartitionKey(&qi)

	h := newTopK5()

	var scratch gridScratch

	sameIdx := idx.partitionsByKey[key]

	if sameIdx >= 0 {
		p := &idx.partitions[sameIdx]
		lb := gridBBoxLowerBound(&qi, &p.bboxMin, &p.bboxMax)
		if lb < h.worstDist() {
			gridSearchKDTree(p.root, lb, &qi, &h, idx, &scratch)
		}
		if h.worstDist() == 0 {
			return h.fraudCount(idx.labels)
		}
	}

	existing := idx.hammingExisting[key]
	start := 0
	if sameIdx >= 0 {
		start = 1
	}
	for j := start; j < len(existing); j++ {
		worst := h.worstDist()
		if worst == 0 {
			break
		}
		candKey := existing[j]
		p := &idx.partitions[idx.partitionsByKey[candKey]]
		var lb uint64
		for d := 0; d < 14; d++ {
			qd := qi[d]
			if qd < p.bboxMin[d] {
				diff := int32(p.bboxMin[d]) - int32(qd)
				lb += uint64(diff) * uint64(diff)
			} else if qd > p.bboxMax[d] {
				diff := int32(qd) - int32(p.bboxMax[d])
				lb += uint64(diff) * uint64(diff)
			}
			if lb >= worst {
				break
			}
		}
		if lb < worst {
			gridSearchKDTree(p.root, lb, &qi, &h, idx, &scratch)
		}
	}

	return h.fraudCount(idx.labels)
}

func gridSearchKDTree(root int32, rootBound uint64, qi *[DIM]int16, h *topK5, idx *GridIndex, scratch *gridScratch) {
	stackLen := 0
	current := root
	currentBound := rootBound

	for {
		if currentBound < h.worstDist() {
			n := &idx.nodes[current]
			if n.left < 0 || n.right < 0 {
				gridScanLeaf(idx.vectors, idx.labels, int(n.blockStart), int(n.vecOffset), int(n.blockLen), *qi, h, &scratch.blockSum)
			} else {
				lNode := &idx.nodes[n.left]
				rNode := &idx.nodes[n.right]
				lb := gridBBoxLowerBound(qi, &lNode.bboxMin, &lNode.bboxMax)
				rb := gridBBoxLowerBound(qi, &rNode.bboxMin, &rNode.bboxMax)

				var nearIdx, farIdx int32
				var nearBound, farBound uint64
				if lb <= rb {
					nearIdx, nearBound = n.left, lb
					farIdx, farBound = n.right, rb
				} else {
					nearIdx, nearBound = n.right, rb
					farIdx, farBound = n.left, lb
				}

				worst := h.worstDist()
				if farBound < worst && stackLen < 64 {
					scratch.stackNodes[stackLen] = farIdx
					scratch.stackBounds[stackLen] = farBound
					stackLen++
				}

				if nearBound < worst {
					current = nearIdx
					currentBound = nearBound
					continue
				}
			}
		}

		if stackLen == 0 {
			return
		}
		stackLen--
		current = scratch.stackNodes[stackLen]
		currentBound = scratch.stackBounds[stackLen]
	}
}

func gridBBoxLowerBound(q *[DIM]int16, bmin, bmax *[DIM]int16) uint64 {
	var d uint64
	qi0 := int32(q[0])
	if qi0 < int32(bmin[0]) {
		diff := int32(bmin[0]) - qi0
		d += uint64(diff) * uint64(diff)
	} else if qi0 > int32(bmax[0]) {
		diff := qi0 - int32(bmax[0])
		d += uint64(diff) * uint64(diff)
	}
	qi1 := int32(q[1])
	if qi1 < int32(bmin[1]) {
		diff := int32(bmin[1]) - qi1
		d += uint64(diff) * uint64(diff)
	} else if qi1 > int32(bmax[1]) {
		diff := qi1 - int32(bmax[1])
		d += uint64(diff) * uint64(diff)
	}
	qi2 := int32(q[2])
	if qi2 < int32(bmin[2]) {
		diff := int32(bmin[2]) - qi2
		d += uint64(diff) * uint64(diff)
	} else if qi2 > int32(bmax[2]) {
		diff := qi2 - int32(bmax[2])
		d += uint64(diff) * uint64(diff)
	}
	qi3 := int32(q[3])
	if qi3 < int32(bmin[3]) {
		diff := int32(bmin[3]) - qi3
		d += uint64(diff) * uint64(diff)
	} else if qi3 > int32(bmax[3]) {
		diff := qi3 - int32(bmax[3])
		d += uint64(diff) * uint64(diff)
	}
	qi4 := int32(q[4])
	if qi4 < int32(bmin[4]) {
		diff := int32(bmin[4]) - qi4
		d += uint64(diff) * uint64(diff)
	} else if qi4 > int32(bmax[4]) {
		diff := qi4 - int32(bmax[4])
		d += uint64(diff) * uint64(diff)
	}
	qi5 := int32(q[5])
	if qi5 < int32(bmin[5]) {
		diff := int32(bmin[5]) - qi5
		d += uint64(diff) * uint64(diff)
	} else if qi5 > int32(bmax[5]) {
		diff := qi5 - int32(bmax[5])
		d += uint64(diff) * uint64(diff)
	}
	qi6 := int32(q[6])
	if qi6 < int32(bmin[6]) {
		diff := int32(bmin[6]) - qi6
		d += uint64(diff) * uint64(diff)
	} else if qi6 > int32(bmax[6]) {
		diff := qi6 - int32(bmax[6])
		d += uint64(diff) * uint64(diff)
	}
	qi7 := int32(q[7])
	if qi7 < int32(bmin[7]) {
		diff := int32(bmin[7]) - qi7
		d += uint64(diff) * uint64(diff)
	} else if qi7 > int32(bmax[7]) {
		diff := qi7 - int32(bmax[7])
		d += uint64(diff) * uint64(diff)
	}
	qi8 := int32(q[8])
	if qi8 < int32(bmin[8]) {
		diff := int32(bmin[8]) - qi8
		d += uint64(diff) * uint64(diff)
	} else if qi8 > int32(bmax[8]) {
		diff := qi8 - int32(bmax[8])
		d += uint64(diff) * uint64(diff)
	}
	qi9 := int32(q[9])
	if qi9 < int32(bmin[9]) {
		diff := int32(bmin[9]) - qi9
		d += uint64(diff) * uint64(diff)
	} else if qi9 > int32(bmax[9]) {
		diff := qi9 - int32(bmax[9])
		d += uint64(diff) * uint64(diff)
	}
	qi10 := int32(q[10])
	if qi10 < int32(bmin[10]) {
		diff := int32(bmin[10]) - qi10
		d += uint64(diff) * uint64(diff)
	} else if qi10 > int32(bmax[10]) {
		diff := qi10 - int32(bmax[10])
		d += uint64(diff) * uint64(diff)
	}
	qi11 := int32(q[11])
	if qi11 < int32(bmin[11]) {
		diff := int32(bmin[11]) - qi11
		d += uint64(diff) * uint64(diff)
	} else if qi11 > int32(bmax[11]) {
		diff := qi11 - int32(bmax[11])
		d += uint64(diff) * uint64(diff)
	}
	qi12 := int32(q[12])
	if qi12 < int32(bmin[12]) {
		diff := int32(bmin[12]) - qi12
		d += uint64(diff) * uint64(diff)
	} else if qi12 > int32(bmax[12]) {
		diff := qi12 - int32(bmax[12])
		d += uint64(diff) * uint64(diff)
	}
	qi13 := int32(q[13])
	if qi13 < int32(bmin[13]) {
		diff := int32(bmin[13]) - qi13
		d += uint64(diff) * uint64(diff)
	} else if qi13 > int32(bmax[13]) {
		diff := qi13 - int32(bmax[13])
		d += uint64(diff) * uint64(diff)
	}
	return d
}

func gridPartitionKey(v *[DIM]int16) uint32 {
	var key uint32
	if v[5] >= 0 {
		key |= 1 << 0
	}
	if v[9] > 0 {
		key |= 1 << 1
	}
	if v[10] > 0 {
		key |= 1 << 2
	}
	if v[11] > 0 {
		key |= 1 << 3
	}
	mcc := v[12]
	var mccBucket uint32
	switch {
	case mcc <= 2499:
		mccBucket = 0
	case mcc <= 4999:
		mccBucket = 1
	case mcc <= 7499:
		mccBucket = 2
	default:
		mccBucket = 3
	}
	key |= mccBucket << 4
	if v[2] > 5000 {
		key |= 1 << 6
	}
	if v[8] > 2500 {
		key |= 1 << 7
	}
	return key
}

const gridBlockLANES = 8

func gridScanLeaf(vectors []int16, labels []byte, startBlock, vecOffset, nVecs int, q [DIM]int16, h *topK5, scratch *[8]float32) {
	if nVecs == 0 {
		return
	}

	qf := [16]float32{
		float32(q[0]), float32(q[1]), float32(q[2]), float32(q[3]),
		float32(q[4]), float32(q[5]), float32(q[6]), float32(q[7]),
		float32(q[8]), float32(q[9]), float32(q[10]), float32(q[11]),
		float32(q[12]), float32(q[13]), 0, 0,
	}

	blockSize := DIM * gridBlockLANES
	nBlocks := (nVecs + gridBlockLANES - 1) / gridBlockLANES

	for b := 0; b < nBlocks; b++ {
		blockBase := (startBlock + b) * blockSize
		worstF32 := float32(h.worstDist())

		alive := ScanBlock8AVX2(&qf[0], &vectors[blockBase], worstF32, scratch)
		if !alive {
			continue
		}

		for lane := 0; lane < gridBlockLANES; lane++ {
			vecIdx := vecOffset + b*gridBlockLANES + lane
			if vecIdx >= vecOffset+nVecs {
				continue
			}
			if scratch[lane] >= worstF32 {
				continue
			}
			ssd := exactI64Dist(&q, &vectors[blockBase], lane)
			h.tryInsert(ssd, vecIdx)
			worstF32 = float32(h.worstDist())
		}
	}
}

func exactI64Dist(qi *[DIM]int16, block *int16, lane int) uint64 {
	var ssd uint64
	blockArr := (*[DIM * gridBlockLANES]int16)(unsafe.Pointer(block))
	for d := 0; d < DIM; d++ {
		diff := int64(qi[d]) - int64(blockArr[d*gridBlockLANES+lane])
		ssd += uint64(diff * diff)
	}
	return ssd
}

// LoadGrid loads a GridIndex from a binary file. The file format is:
// magic (4B) + version (4B) + n (4B) + dim (4B) + nparts (4B) + nnodes (4B)
// + partitions + nodes + vectors (block-major int16) + labels (bit-packed).
// It also starts a background goroutine to periodically touch the vector
// array, keeping pages resident in memory.
func LoadGrid(path string) (*GridIndex, error) {
	f, err := os.Open(path)

	if err != nil {
		return nil, fmt.Errorf("open grid: %w", err)
	}

	defer func() {
		if err := f.Close(); err != nil {
			fmt.Fprintf(os.Stderr, "failed to load the grid: %v\n", err)
		}
	}()

	stat, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("stat: %w", err)
	}
	fileSize := stat.Size()

	var magic, version, n, dim, nparts, nnodes uint32
	if err := binary.Read(f, binary.LittleEndian, &magic); err != nil || magic != gridMagic {
		return nil, fmt.Errorf("invalid grid magic (got 0x%08x)", magic)
	}
	if err := binary.Read(f, binary.LittleEndian, &version); err != nil || version != gridVersion {
		return nil, fmt.Errorf("unsupported grid version %d", version)
	}
	if err := binary.Read(f, binary.LittleEndian, &n); err != nil {
		return nil, fmt.Errorf("read n: %w", err)
	}
	if err := binary.Read(f, binary.LittleEndian, &dim); err != nil || dim != DIM {
		return nil, fmt.Errorf("expected dim=%d got %d", DIM, dim)
	}
	if err := binary.Read(f, binary.LittleEndian, &nparts); err != nil {
		return nil, fmt.Errorf("read nparts: %w", err)
	}
	if err := binary.Read(f, binary.LittleEndian, &nnodes); err != nil {
		return nil, fmt.Errorf("read nnodes: %w", err)
	}

	idx := &GridIndex{
		n:          int(n),
		partitions: make([]gridPartition, nparts),
		nodes:      make([]gridNode, nnodes),
	}

	for i := range idx.partitionsByKey {
		idx.partitionsByKey[i] = -1
	}

	for i := uint32(0); i < nparts; i++ {
		p := &idx.partitions[i]
		if err := binary.Read(f, binary.LittleEndian, &p.key); err != nil {
			return nil, fmt.Errorf("read part %d key: %w", i, err)
		}
		if err := binary.Read(f, binary.LittleEndian, &p.root); err != nil {
			return nil, fmt.Errorf("read part %d root: %w", i, err)
		}
		if err := binary.Read(f, binary.LittleEndian, &p.bboxMin); err != nil {
			return nil, fmt.Errorf("read part %d bboxMin: %w", i, err)
		}
		if err := binary.Read(f, binary.LittleEndian, &p.bboxMax); err != nil {
			return nil, fmt.Errorf("read part %d bboxMax: %w", i, err)
		}
		for d := 0; d < DIM; d++ {
			p.bboxMinF32[d] = float32(p.bboxMin[d])
			p.bboxMaxF32[d] = float32(p.bboxMax[d])
		}
		idx.partitionsByKey[p.key] = int32(i)
	}

	idx.nExisting = 0
	for key := uint32(0); key < 256; key++ {
		if idx.partitionsByKey[key] >= 0 {
			idx.existingKeys[idx.nExisting] = uint8(key)
			idx.nExisting++
		}
	}

	for key := uint32(0); key < 256; key++ {
		n := 0
		for j := 0; j < 256; j++ {
			if idx.partitionsByKey[hammingOrder[key][j]] >= 0 {
				n++
			}
		}
		if n > 0 {
			idx.hammingExisting[key] = make([]uint8, 0, n)
			for j := 0; j < 256; j++ {
				cand := hammingOrder[key][j]
				if idx.partitionsByKey[cand] >= 0 {
					idx.hammingExisting[key] = append(idx.hammingExisting[key], cand)
				}
			}
		}
	}

	for i := uint32(0); i < nnodes; i++ {
		nd := &idx.nodes[i]
		if err := binary.Read(f, binary.LittleEndian, &nd.left); err != nil {
			return nil, fmt.Errorf("read node %d left: %w", i, err)
		}
		if err := binary.Read(f, binary.LittleEndian, &nd.right); err != nil {
			return nil, fmt.Errorf("read node %d right: %w", i, err)
		}
		if err := binary.Read(f, binary.LittleEndian, &nd.blockStart); err != nil {
			return nil, fmt.Errorf("read node %d blockStart: %w", i, err)
		}
		if err := binary.Read(f, binary.LittleEndian, &nd.blockLen); err != nil {
			return nil, fmt.Errorf("read node %d blockLen: %w", i, err)
		}
		if err := binary.Read(f, binary.LittleEndian, &nd.vecOffset); err != nil {
			return nil, fmt.Errorf("read node %d vecOffset: %w", i, err)
		}
		if err := binary.Read(f, binary.LittleEndian, &nd.bboxMin); err != nil {
			return nil, fmt.Errorf("read node %d bboxMin: %w", i, err)
		}
		if err := binary.Read(f, binary.LittleEndian, &nd.bboxMax); err != nil {
			return nil, fmt.Errorf("read node %d bboxMax: %w", i, err)
		}
	}

	nodeSize := int64(4 + 4 + 4 + 4 + 4 + DIM*2 + DIM*2)
	nodesEnd := 24 + int64(nparts)*(4+4+DIM*2+DIM*2) + int64(nnodes)*nodeSize

	labelBytes := (int(n) + 7) / 8
	vecBytes := fileSize - nodesEnd - int64(labelBytes)
	vecCount := int(vecBytes) / 2

	// Memory-map the vectors and labels region (nodesEnd to EOF).
	// Keeps ~84MB of vector data off the Go heap — the GC never scans
	// mmap'd memory, reducing live heap from ~170MB to ~86MB.
	if vecCount > 0 {
		pageSize := int64(os.Getpagesize())
		mmapStart := (nodesEnd / pageSize) * pageSize
		mmapAdjust := nodesEnd - mmapStart
		mmapLen := int(vecBytes + int64(labelBytes) + mmapAdjust)

		mmapData, err := syscall.Mmap(int(f.Fd()), mmapStart, mmapLen,
			syscall.PROT_READ, syscall.MAP_PRIVATE|syscall.MAP_POPULATE)
		if err != nil {
			return nil, fmt.Errorf("mmap grid data: %w", err)
		}
		idx.mmapData = mmapData

		vecSlice := mmapData[mmapAdjust : mmapAdjust+vecBytes]
		idx.vectors = unsafe.Slice((*int16)(unsafe.Pointer(&vecSlice[0])), vecCount)
		idx.labels = mmapData[mmapAdjust+vecBytes : mmapAdjust+vecBytes+int64(labelBytes)]

		// Fault in all pages once (MAP_POPULATE is a hint on some kernels).
		for i := 0; i < len(idx.vectors); i += 4096 {
			_ = idx.vectors[i]
		}

		// Pin pages in RAM to prevent kernel eviction under cgroup memory
		// pressure (container limit 170M = heap 86M + mmap 84M).
		if err := syscall.Mlock(mmapData); err != nil {
			fmt.Fprintf(os.Stderr, "grid mlock (non-fatal): %v\n", err)
		}
	}

	return idx, nil
}

// Close unmaps the memory-mapped regions. Safe to call multiple times.
func (idx *GridIndex) Close() error {
	if idx.mmapData != nil {
		if err := syscall.Munmap(idx.mmapData); err != nil {
			return fmt.Errorf("munmap grid: %w", err)
		}
		idx.mmapData = nil
		idx.vectors = nil
		idx.labels = nil
	}
	return nil
}
