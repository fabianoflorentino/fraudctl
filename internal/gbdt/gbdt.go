// Package gbdt provides fast inference using a pre-trained Gradient Boosted Decision Tree.
package gbdt

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
)

type node struct {
	thresh float32
	feat   uint8
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
	NumTrees    int
	initPred    float32
	sigmoid     bool
	numFeatures int
	trees       []tree
}

// Load reads a GBDT model from a binary file.
//
// Supports two binary formats (little-endian):
//
// Version 1 (legacy):
//
//	magic       [4]byte  "GBDT"
//	_           uint32   (ignored — treated as padding)
//	numTrees    uint32
//	lr          float64  (learning rate, multiplied into leaves at load time)
//	initPred    float64
//	per-tree:
//	  numNodes  uint32
//	  per-node (18 bytes):
//	    feat    uint8
//	    thresh  float64
//	    left    uint16
//	    right   uint16
//	    value   float32
//	    leaf    uint8
//	Uses sigmoid activation (binary classification with log-loss).
//
// Version 2:
//
//	magic       [4]byte  "GBDT"
//	version     uint32   = 2
//	numTrees    uint32
//	numFeatures uint32
//	initPred    float32
//	sigmoid     uint8    0=no 1=yes
//	pad         [3]byte
//	per-tree:
//	  numNodes  uint32
//	  per-node (12 bytes):
//	    feat    uint8
//	    pad     uint8
//	    left    uint16
//	    right   uint16
//	    leaf    uint8
//	    pad     uint8
//	    value   float32  (thresh for internal nodes, leaf value for leaves)
func Load(path string) (*GBDT, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	if len(data) < 16 {
		return nil, fmt.Errorf("gbdt: file too short")
	}
	if string(data[:4]) != "GBDT" {
		return nil, fmt.Errorf("gbdt: invalid magic %q", data[:4])
	}

	version := binary.LittleEndian.Uint32(data[4:8])

	switch version {
	case 2:
		return loadV2(data)
	default:
		// version field was unused in v1 — treat any non-2 value as v1
		return loadV1(data)
	}
}

// loadV1 reads the legacy v1 format (18 bytes/node, float64 thresh, lr scaling).
func loadV1(data []byte) (*GBDT, error) {
	if len(data) < 28 {
		return nil, fmt.Errorf("gbdt v1: file too short")
	}
	numTrees := int(binary.LittleEndian.Uint32(data[8:12]))
	lr := math.Float64frombits(binary.LittleEndian.Uint64(data[12:20]))
	initPred := math.Float64frombits(binary.LittleEndian.Uint64(data[20:28]))

	g := &GBDT{
		trees:       make([]tree, numTrees),
		initPred:    float32(initPred),
		sigmoid:     true, // v1 was always binary classification
		numFeatures: 14,
		NumTrees:    numTrees,
	}

	offset := 28
	for i := 0; i < numTrees; i++ {
		if offset+4 > len(data) {
			return nil, fmt.Errorf("gbdt v1: unexpected end at tree %d", i)
		}
		numNodes := int(binary.LittleEndian.Uint32(data[offset : offset+4]))
		offset += 4

		nodes := make([]node, numNodes)
		for j := 0; j < numNodes; j++ {
			if offset+18 > len(data) {
				return nil, fmt.Errorf("gbdt v1: unexpected end at tree %d node %d", i, j)
			}
			nodes[j].feat = data[offset]
			thresh := math.Float64frombits(binary.LittleEndian.Uint64(data[offset+1 : offset+9]))
			nodes[j].thresh = float32(thresh)
			nodes[j].left = binary.LittleEndian.Uint16(data[offset+9 : offset+11])
			nodes[j].right = binary.LittleEndian.Uint16(data[offset+11 : offset+13])
			leafVal := math.Float32frombits(binary.LittleEndian.Uint32(data[offset+13 : offset+17]))
			nodes[j].value = float32(lr) * leafVal
			nodes[j].leaf = data[offset+17] != 0
			offset += 18
		}
		g.trees[i].nodes = nodes
	}

	return g, nil
}

// loadV2 reads the v2 format (12 bytes/node, float32 thresh).
func loadV2(data []byte) (*GBDT, error) {
	if len(data) < 24 {
		return nil, fmt.Errorf("gbdt v2: file too short")
	}
	numTrees := int(binary.LittleEndian.Uint32(data[8:12]))
	numFeatures := int(binary.LittleEndian.Uint32(data[12:16]))
	initPred := math.Float32frombits(binary.LittleEndian.Uint32(data[16:20]))
	sigmoid := data[20] != 0

	g := &GBDT{
		trees:       make([]tree, numTrees),
		initPred:    initPred,
		sigmoid:     sigmoid,
		numFeatures: numFeatures,
		NumTrees:    numTrees,
	}

	offset := 24
	for i := 0; i < numTrees; i++ {
		if offset+4 > len(data) {
			return nil, fmt.Errorf("gbdt v2: unexpected end at tree %d", i)
		}
		numNodes := int(binary.LittleEndian.Uint32(data[offset : offset+4]))
		offset += 4

		nodes := make([]node, numNodes)
		for j := 0; j < numNodes; j++ {
			if offset+12 > len(data) {
				return nil, fmt.Errorf("gbdt v2: unexpected end at tree %d node %d", i, j)
			}
			nodes[j].feat = data[offset]
			// [offset+1] = pad
			nodes[j].left = binary.LittleEndian.Uint16(data[offset+2 : offset+4])
			nodes[j].right = binary.LittleEndian.Uint16(data[offset+4 : offset+6])
			nodes[j].leaf = data[offset+6] != 0
			// [offset+7] = pad
			val := math.Float32frombits(binary.LittleEndian.Uint32(data[offset+8 : offset+12]))
			if nodes[j].leaf {
				nodes[j].value = val
			} else {
				nodes[j].thresh = val
			}
			offset += 12
		}
		g.trees[i].nodes = nodes
	}

	return g, nil
}

// Predict computes the fraud score for a 14-dim feature vector.
// Returns a value in [0, 1].
func (g *GBDT) Predict(vec [14]float32) float32 {
	pred := g.initPred

	for i := range g.trees {
		t := &g.trees[i]
		idx := 0
		for !t.nodes[idx].leaf {
			n := &t.nodes[idx]
			if vec[n.feat] <= n.thresh {
				idx = int(n.left)
			} else {
				idx = int(n.right)
			}
		}
		pred += t.nodes[idx].value
	}

	if g.sigmoid {
		if pred > 20 {
			return 1.0
		}
		if pred < -20 {
			return 0.0
		}
		return float32(1.0 / (1.0 + math.Exp(-float64(pred))))
	}

	// Regression: clamp to [0, 1]
	if pred < 0 {
		return 0
	}
	if pred > 1 {
		return 1
	}
	return pred
}
