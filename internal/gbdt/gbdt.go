// Package gbdt provides fast inference using a pre-trained Gradient Boosted Decision Tree.
package gbdt

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
)

type node struct {
	feat   uint8
	thresh float64
	left   uint16
	right  uint16
	value  float32
	leaf   bool
}

type tree struct {
	nodes []node
}

// GBDT holds the loaded model for fraud prediction.
type GBDT struct {
	NumTrees   int
	lr         float64
	initPred   float64
	numFeatures int
	trees      []tree
}

// Load reads a GBDT model from a binary file.
func Load(path string) (*GBDT, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	if len(data) < 24 {
		return nil, fmt.Errorf("gbdt: file too short")
	}
	if string(data[:4]) != "GBDT" {
		return nil, fmt.Errorf("gbdt: invalid magic")
	}

	numTrees := int(binary.LittleEndian.Uint32(data[8:12]))
	lr := math.Float64frombits(binary.LittleEndian.Uint64(data[12:20]))
	initPred := math.Float64frombits(binary.LittleEndian.Uint64(data[20:28]))

	g := &GBDT{
		trees:      make([]tree, numTrees),
		lr:         lr,
		initPred:   initPred,
		numFeatures: 14,
		NumTrees:   numTrees,
	}

	offset := 28
	for i := 0; i < numTrees; i++ {
		if offset+4 > len(data) {
			return nil, fmt.Errorf("gbdt: unexpected end of file")
		}
		numNodes := int(binary.LittleEndian.Uint32(data[offset : offset+4]))
		offset += 4

		nodes := make([]node, numNodes)
		for j := 0; j < numNodes; j++ {
			if offset+18 > len(data) {
				return nil, fmt.Errorf("gbdt: unexpected end of file")
			}
			nodes[j].feat = data[offset]
			nodes[j].thresh = math.Float64frombits(binary.LittleEndian.Uint64(data[offset+1 : offset+9]))
			nodes[j].left = binary.LittleEndian.Uint16(data[offset+9 : offset+11])
			nodes[j].right = binary.LittleEndian.Uint16(data[offset+11 : offset+13])
			nodes[j].value = math.Float32frombits(binary.LittleEndian.Uint32(data[offset+13 : offset+17]))
			nodes[j].leaf = data[offset+17] != 0
			offset += 18
		}
		g.trees[i].nodes = nodes
	}

	return g, nil
}

// Predict computes the fraud probability for a given vector.
func (g *GBDT) Predict(vec []float32) float64 {
	pred := g.initPred

	for i := range g.trees {
		t := &g.trees[i]
		idx := 0
		for !t.nodes[idx].leaf {
			n := &t.nodes[idx]
			if vec[n.feat] <= float32(n.thresh) {
				idx = int(n.left)
			} else {
				idx = int(n.right)
			}
		}
		pred += g.lr * float64(t.nodes[idx].value)
	}

	// Sigmoid with overflow protection
	if pred > 20 {
		return 1.0
	}
	if pred < -20 {
		return 0.0
	}
	return 1.0 / (1.0 + math.Exp(-pred))
}
