package knn

import (
	"compress/gzip"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"slices"
	"sort"
)

const gridLeafSize = 128

type gridBuilder struct {
	vecs   []int16
	labels []bool
	n      int
}

// BuildGrid constructs a GridIndex (KD-tree with 8-bit partition keys) from
// a gzipped JSON references file and writes it to outPath in binary format.
// The index uses block-major vector layout for AVX2-accelerated scanning.
func BuildGrid(refsGz, outPath string) error {
	fmt.Printf("BuildGrid: loading %s ...\n", refsGz)

	f, err := os.Open(refsGz)
	if err != nil {
		return fmt.Errorf("open refs: %w", err)
	}
	defer func() {
		if err := f.Close(); err != nil {
			fmt.Fprintf(os.Stderr, "failed to close refs file: %v\n", err)
		}
	}()

	gz, err := gzip.NewReader(f)
	if err != nil {
		return fmt.Errorf("gzip: %w", err)
	}
	defer func() {
		if err := gz.Close(); err != nil {
			fmt.Fprintf(os.Stderr, "failed to close gzip file: %v\n", err)
		}
	}()

	gb := &gridBuilder{}

	dec := json.NewDecoder(gz)
	if _, err := dec.Token(); err != nil {
		return fmt.Errorf("json token: %w", err)
	}

	var entry struct {
		RawVec []float64 `json:"vector"`
		Label  string    `json:"label"`
	}
	for dec.More() {
		entry.RawVec = entry.RawVec[:0]
		if err := dec.Decode(&entry); err != nil {
			break
		}
		for i := 0; i < DIM; i++ {
			var v float32
			if i < len(entry.RawVec) {
				v = float32(entry.RawVec[i])
			}
			gb.vecs = append(gb.vecs, quantizeFloat32(v))
		}
		gb.labels = append(gb.labels, entry.Label == "fraud")
	}
	gb.n = len(gb.labels)
	fmt.Printf("BuildGrid: loaded %d vectors\n", gb.n)

	buckets := make(map[uint32][]uint32, 256)
	var v [DIM]int16
	for i := 0; i < gb.n; i++ {
		base := i * DIM
		for d := range DIM {
			v[d] = gb.vecs[base+d]
		}
		key := gridPartitionKey(&v)
		buckets[key] = append(buckets[key], uint32(i))
	}

	keys := make([]uint32, 0, len(buckets))
	for k := range buckets {
		keys = append(keys, k)
	}
	slices.Sort(keys)

	type kdPartition struct {
		key     uint32
		root    int32
		bboxMin [DIM]int16
		bboxMax [DIM]int16
	}

	var partitions []kdPartition
	var nodes []gridNode
	var reordered []int16
	bitLabels := make([]byte, (gb.n+7)/8)

	nodeAlloc := 0
	vecCount := 0

	var buildFn func(members []uint32) int32
	buildFn = func(members []uint32) int32 {
		var nmin, nmax [DIM]int16
		for d := range DIM {
			nmin[d] = math.MaxInt16
			nmax[d] = math.MinInt16
		}
		for _, m := range members {
			base := int(m) * DIM
			for d := range DIM {
				val := gb.vecs[base+d]
				if val < nmin[d] {
					nmin[d] = val
				}
				if val > nmax[d] {
					nmax[d] = val
				}
			}
		}

		nodeIdx := nodeAlloc
		nodeAlloc++
		nodes = append(nodes, gridNode{
			left: -1, right: -1,
			bboxMin: nmin, bboxMax: nmax,
		})

		blockVectors := make([]int16, DIM*gridBlockLANES)

		if len(members) <= gridLeafSize {
			startBlock := len(reordered) / (DIM * gridBlockLANES)
			nBlocks := (len(members) + gridBlockLANES - 1) / gridBlockLANES
			for b := 0; b < nBlocks; b++ {
				for d := range DIM {
					for lane := range gridBlockLANES {
						vecIdx := b*gridBlockLANES + lane
						if vecIdx < len(members) {
							m := members[vecIdx]
							blockVectors[d*gridBlockLANES+lane] = gb.vecs[int(m)*DIM+d]
						}
					}
				}
				reordered = append(reordered, blockVectors...)
			}
			for lane := 0; lane < len(members); lane++ {
				globalIdx := vecCount + lane
				if gb.labels[int(members[lane])] {
					bitLabels[globalIdx>>3] |= 1 << uint(globalIdx&7)
				}
			}
			nodes[nodeIdx].blockStart = uint32(startBlock)
			nodes[nodeIdx].blockLen = uint32(len(members))
			nodes[nodeIdx].vecOffset = uint32(vecCount)
			vecCount += len(members)
			return int32(nodeIdx)
		}

		splitDim := widestDim(&nmin, &nmax)
		sort.SliceStable(members, func(i, j int) bool {
			ai := int(members[i])*DIM + splitDim
			bi := int(members[j])*DIM + splitDim
			return gb.vecs[ai] < gb.vecs[bi]
		})

		mid := len(members) / 2
		left := buildFn(members[:mid])
		right := buildFn(members[mid:])
		nodes[nodeIdx].left = left
		nodes[nodeIdx].right = right
		return int32(nodeIdx)
	}

	for _, key := range keys {
		members := buckets[key]
		root := buildFn(members)
		p := kdPartition{
			key:     key,
			root:    root,
			bboxMin: nodes[root].bboxMin,
			bboxMax: nodes[root].bboxMax,
		}
		partitions = append(partitions, p)
	}

	if vecCount != gb.n {
		return fmt.Errorf("vector count mismatch: %d vs %d", vecCount, gb.n)
	}

	fmt.Printf("BuildGrid: writing %s (%d partitions, %d nodes, %d vectors) ...\n",
		outPath, len(partitions), len(nodes), gb.n)

	out, err := os.Create(outPath)
	if err != nil {
		return fmt.Errorf("create: %w", err)
	}
	defer func() {
		if err := out.Close(); err != nil {
			fmt.Fprintf(os.Stderr, "failed to close output file: %v\n", err)
		}
	}()

	write32 := func(v uint32) error { return binary.Write(out, binary.LittleEndian, v) }
	if err := write32(gridMagic); err != nil {
		return err
	}
	if err := write32(gridVersion); err != nil {
		return err
	}
	if err := write32(uint32(gb.n)); err != nil {
		return err
	}
	if err := write32(DIM); err != nil {
		return err
	}
	if err := write32(uint32(len(partitions))); err != nil {
		return err
	}
	if err := write32(uint32(len(nodes))); err != nil {
		return err
	}

	for _, p := range partitions {
		if err := binary.Write(out, binary.LittleEndian, p.key); err != nil {
			return err
		}
		if err := binary.Write(out, binary.LittleEndian, p.root); err != nil {
			return err
		}
		if err := binary.Write(out, binary.LittleEndian, p.bboxMin); err != nil {
			return err
		}
		if err := binary.Write(out, binary.LittleEndian, p.bboxMax); err != nil {
			return err
		}
	}

	for _, n := range nodes {
		if err := binary.Write(out, binary.LittleEndian, n.left); err != nil {
			return err
		}
		if err := binary.Write(out, binary.LittleEndian, n.right); err != nil {
			return err
		}
		if err := binary.Write(out, binary.LittleEndian, n.blockStart); err != nil {
			return err
		}
		if err := binary.Write(out, binary.LittleEndian, n.blockLen); err != nil {
			return err
		}
		if err := binary.Write(out, binary.LittleEndian, n.vecOffset); err != nil {
			return err
		}
		if err := binary.Write(out, binary.LittleEndian, n.bboxMin); err != nil {
			return err
		}
		if err := binary.Write(out, binary.LittleEndian, n.bboxMax); err != nil {
			return err
		}
	}

	if err := binary.Write(out, binary.LittleEndian, reordered); err != nil {
		return err
	}

	if _, err := out.Write(bitLabels); err != nil {
		return err
	}

	fmt.Println("BuildGrid: done")
	return nil
}

func widestDim(nmin, nmax *[DIM]int16) int {
	best := 0
	bestWidth := int32(nmax[0]) - int32(nmin[0])

	for d := 1; d < DIM; d++ {
		w := int32(nmax[d]) - int32(nmin[d])
		if w > bestWidth {
			bestWidth = w
			best = d
		}
	}

	return best
}
