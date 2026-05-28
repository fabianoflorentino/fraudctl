// Package gbdt provides fast inference and training for Gradient Boosted Decision Trees.
package gbdt

import (
	"bytes"
	"encoding/binary"
	"math"
	"math/rand"
	"os"

	"github.com/fabianoflorentino/fraudctl/internal/model"
)

// Sample is a labeled training example.
type Sample struct {
	Vector model.Vector14
	Label  bool // true = fraud
}

// ============================================================================
// Decision Tree Training
// ============================================================================

const (
	defaultMaxDepth    = 8
	defaultMinSamples  = 500
	defaultMinGain     = 0.001
	numBins            = 256
	numFeatures        = 14
	minSamplesForSplit = 20
)

type dtNode struct {
	feature   int
	threshold float32
	leaf      bool
	value     float32 // leaf: fraud probability; internal: unused
	left      *dtNode
	right     *dtNode
}

// giniImpurity computes the Gini impurity for a set of samples.
// Gini = 1 - p(fraud)² - p(legit)².
func giniImpurity(fraud, total int) float64 {
	if total == 0 {
		return 0
	}
	p := float64(fraud) / float64(total)
	return 1 - p*p - (1-p)*(1-p)
}

type binCount struct {
	count int
	fraud int
}

// findBestSplit evaluates all 14 features via histograms and returns the best split.
// Returns (feature, threshold, gain). gain=0 means no valid split found.
func findBestSplit(samples []Sample, indices []int) (int, float32, float64) {
	if len(indices) < minSamplesForSplit {
		return -1, 0, 0
	}

	totalCount := len(indices)
	var totalFraud int
	for _, idx := range indices {
		if samples[idx].Label {
			totalFraud++
		}
	}

	// If all same label, no point splitting
	if totalFraud == 0 || totalFraud == totalCount {
		return -1, 0, 0
	}

	parentGini := giniImpurity(totalFraud, totalCount)

	bestFeature := -1
	var bestThreshold float32
	bestGain := 0.0

	// Pre-compute feature values to avoid repeated array access
	featVals := make([]float32, len(indices))

	for feat := range numFeatures {
		// Find min/max for this feature across the subset
		minVal := float32(math.MaxFloat32)
		maxVal := float32(-math.MaxFloat32)
		for i, idx := range indices {
			v := samples[idx].Vector[feat]
			featVals[i] = v
			if v < minVal {
				minVal = v
			}
			if v > maxVal {
				maxVal = v
			}
		}

		rangeVal := maxVal - minVal
		if rangeVal < 1e-8 {
			continue
		}

		// Build histogram
		var bins [numBins]binCount
		scale := float32(numBins-1) / rangeVal
		for i, idx := range indices {
			binIdx := int((featVals[i] - minVal) * scale)
			if binIdx >= numBins {
				binIdx = numBins - 1
			}
			bins[binIdx].count++
			if samples[idx].Label {
				bins[binIdx].fraud++
			}
		}

		// Evaluate split at each bin boundary
		leftCount := 0
		leftFraud := 0
		for i := range numBins - 1 {
			leftCount += bins[i].count
			leftFraud += bins[i].fraud
			rightCount := totalCount - leftCount
			rightFraud := totalFraud - leftFraud

			if leftCount < minSamplesForSplit || rightCount < minSamplesForSplit {
				continue
			}

			leftGini := giniImpurity(leftFraud, leftCount)
			rightGini := giniImpurity(rightFraud, rightCount)

			weighted := (float64(leftCount)/float64(totalCount))*leftGini +
				(float64(rightCount)/float64(totalCount))*rightGini
			gain := parentGini - weighted

			if gain > bestGain {
				bestGain = gain
				bestFeature = feat
				// Threshold at the midpoint between adjacent bin boundaries
				bestThreshold = minVal + (float32(i)+0.5)*(rangeVal/float32(numBins))
			}
		}
	}

	return bestFeature, bestThreshold, bestGain
}

// partition splits indices by whether samples[idx].Vector[feat] <= threshold.
// Returns the split point (first index of the > threshold side).
func partition(indices []int, samples []Sample, feat int, threshold float32) int {
	left, right := 0, len(indices)-1
	for left <= right {
		for left <= right && samples[indices[left]].Vector[feat] <= threshold {
			left++
		}
		for left <= right && samples[indices[right]].Vector[feat] > threshold {
			right--
		}
		if left < right {
			indices[left], indices[right] = indices[right], indices[left]
			left++
			right--
		}
	}
	return left
}

// buildTree recursively builds a decision tree from the given subset.
func buildTree(samples []Sample, indices []int, depth int, maxDepth int, minSamples int, minGain float64) *dtNode {
	// Count fraud in this subset
	var fraudCount int
	for _, idx := range indices {
		if samples[idx].Label {
			fraudCount++
		}
	}
	totalCount := len(indices)
	fraudProb := float32(fraudCount) / float32(totalCount)

	// Stopping conditions
	if depth >= maxDepth || totalCount < minSamples ||
		fraudCount == 0 || fraudCount == totalCount {
		return &dtNode{leaf: true, value: fraudProb}
	}

	feat, threshold, gain := findBestSplit(samples, indices)
	if gain < minGain || feat < 0 {
		return &dtNode{leaf: true, value: fraudProb}
	}

	// Partition indices in-place
	split := partition(indices, samples, feat, threshold)
	leftIndices := indices[:split]
	rightIndices := indices[split:]

	sp := minSamplesForSplit
	if minSamples > 0 && minSamples < sp {
		sp = minSamples
	}
	if len(leftIndices) < sp || len(rightIndices) < sp {
		return &dtNode{leaf: true, value: fraudProb}
	}

	node := &dtNode{
		feature:   feat,
		threshold: threshold,
	}
	node.left = buildTree(samples, leftIndices, depth+1, maxDepth, minSamples, minGain)
	node.right = buildTree(samples, rightIndices, depth+1, maxDepth, minSamples, minGain)
	return node
}

// ============================================================================
// Serialization to v2 binary format
// ============================================================================

type flatNode struct {
	feat   uint8
	thresh float32
	left   uint16
	right  uint16
	leaf   bool
	value  float32
}

// flattenTree serializes a tree into a flat slice using pre-order DFS.
func flattenTree(n *dtNode, flat *[]flatNode) uint16 {
	idx := uint16(len(*flat))
	*flat = append(*flat, flatNode{})
	if n.leaf {
		(*flat)[idx] = flatNode{
			leaf:  true,
			value: n.value,
		}
	} else {
		leftIdx := flattenTree(n.left, flat)
		rightIdx := flattenTree(n.right, flat)
		(*flat)[idx] = flatNode{
			feat:   uint8(n.feature),
			thresh: n.threshold,
			left:   leftIdx,
			right:  rightIdx,
		}
	}
	return idx
}

// exportV2Multi writes a v2 binary file with one or more trees.
// Header: magic(4), version(4), numTrees(4), numFeatures(4), initPred(4), sigmoid(1), pad(3)
// Then for each tree: numNodes(4) + nodes (12 bytes each).
func exportV2Multi(path string, allTrees [][]flatNode) error {
	header := make([]byte, 24)
	copy(header[0:4], []byte("GBDT"))
	binary.LittleEndian.PutUint32(header[4:8], 2)                       // version
	binary.LittleEndian.PutUint32(header[8:12], uint32(len(allTrees)))  // numTrees
	binary.LittleEndian.PutUint32(header[12:16], numFeatures)           // numFeatures
	binary.LittleEndian.PutUint32(header[16:20], math.Float32bits(0.0)) // initPred
	header[20] = 0                                                      // sigmoid=0 (regression, averaged [0,1])

	var buf bytes.Buffer
	buf.Write(header)

	for _, flat := range allTrees {
		treeHdr := make([]byte, 4)
		binary.LittleEndian.PutUint32(treeHdr, uint32(len(flat)))
		buf.Write(treeHdr)

		nodeData := make([]byte, len(flat)*12)
		for i, n := range flat {
			base := i * 12
			nodeData[base] = n.feat
			nodeData[base+1] = 0 // pad
			binary.LittleEndian.PutUint16(nodeData[base+2:base+4], n.left)
			binary.LittleEndian.PutUint16(nodeData[base+4:base+6], n.right)
			if n.leaf {
				nodeData[base+6] = 1
			} else {
				nodeData[base+6] = 0
			}
			nodeData[base+7] = 0 // pad
			// Write threshold for internal nodes, leaf value for leaf nodes
			v := n.value
			if !n.leaf {
				v = n.thresh
			}
			binary.LittleEndian.PutUint32(nodeData[base+8:base+12], math.Float32bits(v))
		}
		buf.Write(nodeData)
	}

	return os.WriteFile(path, buf.Bytes(), 0644)
}

// TrainAndExport trains a decision tree and writes it in v2 binary format.
// samples: labeled training vectors.
// path: output file path (e.g., "resources/gbdt.bin").
// maxDepth: maximum tree depth (0 = use default of 8).
// minGain: minimum Gini gain to split (0 = use default of 0.001).
// minSamples: minimum samples per leaf (0 = use default of 500).
func TrainAndExport(samples []Sample, path string, maxDepth int, minSamples int, minGain float64) error {
	if maxDepth <= 0 {
		maxDepth = defaultMaxDepth
	}
	if minSamples <= 0 {
		minSamples = defaultMinSamples
	}
	if minGain <= 0 {
		minGain = defaultMinGain
	}
	// Create index array
	indices := make([]int, len(samples))
	for i := range indices {
		indices[i] = i
	}

	root := buildTree(samples, indices, 0, maxDepth, minSamples, minGain)

	var flat []flatNode
	flattenTree(root, &flat)

	return exportV2Multi(path, [][]flatNode{flat})
}

// TrainRandomForest trains a Random Forest ensemble using bootstrap sampling.
// Each tree is trained on a random subset (with replacement) of the data.
// When bootSize > 0, the bootstrap sample is capped to that many indices
// for faster training on large datasets. Output uses v2 format with
// sigmoid=false (regression mode, Predict averages tree outputs).
func TrainRandomForest(samples []Sample, path string, numTrees int, maxDepth int, minSamples int, minGain float64, seed int64, bootSize int) error {
	if maxDepth <= 0 {
		maxDepth = defaultMaxDepth
	}
	if minSamples <= 0 {
		minSamples = defaultMinSamples
	}
	if minGain <= 0 {
		minGain = defaultMinGain
	}

	rng := rand.New(rand.NewSource(seed))

	n := len(samples)
	if bootSize <= 0 || bootSize > n {
		bootSize = n
	}

	allTrees := make([][]flatNode, numTrees)

	for t := range numTrees {
		indices := make([]int, bootSize)
		for i := range indices {
			indices[i] = rng.Intn(n)
		}

		root := buildTree(samples, indices, 0, maxDepth, minSamples, minGain)

		var flat []flatNode
		flattenTree(root, &flat)
		allTrees[t] = flat
	}

	return exportV2Multi(path, allTrees)
}
