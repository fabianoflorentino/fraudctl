package main

import (
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"sort"
)

const DIM = 14

type gridNode struct {
	left, right int32
	blockStart  uint32
	blockLen    uint32
	vecOffset   uint32
	bboxMin     [DIM]int16
	bboxMax     [DIM]int16
}

func main() {
	path := filepath.Join("resources", "grid.bin")
	data, err := os.ReadFile(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "read: %v\n", err)
		os.Exit(1)
	}

	magic := binary.LittleEndian.Uint32(data[0:4])
	version := binary.LittleEndian.Uint32(data[4:8])
	n := int(binary.LittleEndian.Uint32(data[8:12]))
	dim := int(binary.LittleEndian.Uint32(data[12:16]))
	numPartitions := int(binary.LittleEndian.Uint32(data[16:20]))
	numNodes := int(binary.LittleEndian.Uint32(data[20:24]))

	fmt.Printf("Grid: magic=0x%X version=%d n=%d dim=%d partitions=%d nodes=%d\n",
		magic, version, n, dim, numPartitions, numNodes)

	// Parse partitions
	type partition struct {
		key  uint32
		root int32
	}
	offset := 24
	parts := make([]partition, numPartitions)
	for i := range numPartitions {
		key := binary.LittleEndian.Uint32(data[offset : offset+4])
		root := int32(binary.LittleEndian.Uint32(data[offset+4 : offset+8]))
		parts[i] = partition{key: key, root: root}
		offset += 8 + 14*2 + 14*2 // key + root + bboxMin + bboxMax
	}

	// Parse nodes
	nodes := make([]gridNode, numNodes)
	for i := range numNodes {
		nodes[i].left = int32(binary.LittleEndian.Uint32(data[offset : offset+4]))
		nodes[i].right = int32(binary.LittleEndian.Uint32(data[offset+4 : offset+8]))
		nodes[i].blockStart = binary.LittleEndian.Uint32(data[offset+8 : offset+12])
		nodes[i].blockLen = binary.LittleEndian.Uint32(data[offset+12 : offset+16])
		nodes[i].vecOffset = binary.LittleEndian.Uint32(data[offset+16 : offset+20])
		for d := range DIM {
			nodes[i].bboxMin[d] = int16(binary.LittleEndian.Uint16(data[offset+20+d*2 : offset+22+d*2]))
			nodes[i].bboxMax[d] = int16(binary.LittleEndian.Uint16(data[offset+48+d*2 : offset+50+d*2]))
		}
		offset += 20 + DIM*4 // left+right+blockStart+blockLen+vecOffset + bboxMin+bboxMax
	}

	// Count vectors per partition
	type partStat struct {
		key     uint32
		vectors int
	}
	stats := make([]partStat, numPartitions)
	for pi, p := range parts {
		// DFS to count leaf blockLen
		var countLeaf func(nodeIdx int32) int
		countLeaf = func(nodeIdx int32) int {
			n := &nodes[nodeIdx]
			if n.left == -1 && n.right == -1 {
				return int(n.blockLen)
			}
			total := 0
			if n.left >= 0 {
				total += countLeaf(n.left)
			}
			if n.right >= 0 {
				total += countLeaf(n.right)
			}
			return total
		}
		stats[pi] = partStat{key: p.key, vectors: countLeaf(p.root)}
	}

	// Sort by size descending
	sort.Slice(stats, func(i, j int) bool {
		return stats[i].vectors > stats[j].vectors
	})

	fmt.Printf("\nPartition distribution (sorted by size):\n")
	fmt.Printf("Rank  Key(bin)     Vectors   Pct%%   Cum%%\n")
	cum := 0
	for i, s := range stats {
		pct := float64(s.vectors) / float64(n) * 100
		cum += s.vectors
		cumPct := float64(cum) / float64(n) * 100
		fmt.Printf("  %2d  %08b  %8d  %5.1f%%  %5.1f%%\n", i, s.key, s.vectors, pct, cumPct)
		if i >= 19 && len(stats) > 25 {
			fmt.Printf("  ... (%d more partitions)\n", len(stats)-20)
			break
		}
	}

	// Summary
	minV := stats[len(stats)-1].vectors
	maxV := stats[0].vectors
	var total int
	for _, s := range stats {
		total += s.vectors
	}
	mean := float64(total) / float64(len(stats))
	fmt.Printf("\nSummary:\n")
	fmt.Printf("  Used partitions: %d / 256 (%.1f%%)\n", len(stats), float64(len(stats))/256*100)
	fmt.Printf("  Min vectors:     %d\n", minV)
	fmt.Printf("  Max vectors:     %d\n", maxV)
	fmt.Printf("  Mean vectors:    %.0f\n", mean)
	fmt.Printf("  Max/Min ratio:   %.1fx\n", float64(maxV)/float64(max(minV, 1)))
	fmt.Printf("  Empty keys:      %d / 256\n", 256-len(stats))
}
