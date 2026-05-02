package logreg

import (
	"encoding/binary"
	"io"
	"math"
)

const (
	dimInput    = 14
	dimExpanded = 14 + 14*15/2
	iterations  = 30
	lr          = 0.1
	threshold   = 0.5
)

type Model struct {
	weights []float64
	bias    float64
	dim     int
}

func NewModel() *Model {
	return &Model{
		weights: make([]float64, dimExpanded),
		dim:     dimExpanded,
	}
}

func sigmoid(x float64) float64 {
	if x > 20 {
		return 1
	}
	if x < -20 {
		return 0
	}
	return 1 / (1 + math.Exp(-x))
}

func expand(v [dimInput]float32) []float64 {
	feat := make([]float64, dimExpanded)
	i := 0
	for d := 0; d < dimInput; d++ {
		feat[i] = float64(v[d])
		i++
	}
	for a := 0; a < dimInput; a++ {
		va := feat[a]
		for b := a; b < dimInput; b++ {
			feat[i] = va * feat[b]
			i++
		}
	}
	return feat
}

func (m *Model) Train(vectors []float32, labels []bool, n int) {
	rng := randNew(42)

	posCount := 0
	for i := 0; i < n; i++ {
		if labels[i] {
			posCount++
		}
	}
	negCount := n - posCount
	posWeight := float64(n) / (2.0 * float64(posCount))
	negWeight := float64(n) / (2.0 * float64(negCount))

	feat := make([]float64, dimExpanded)

	for iter := 0; iter < iterations; iter++ {
		indices := make([]int, n)
		for i := 0; i < n; i++ {
			indices[i] = i
		}
		for i := n - 1; i > 0; i-- {
			j := rng.Intn(i + 1)
			indices[i], indices[j] = indices[j], indices[i]
		}

		sampleN := n
		if n > 300000 {
			sampleN = 300000
		}

		var totalLoss float64
		var correct int

		for s := 0; s < sampleN; s++ {
			idx := indices[s]
			for d := 0; d < dimInput; d++ {
				feat[d] = float64(vectors[idx*dimInput+d])
			}
			fi := dimInput
			for a := 0; a < dimInput; a++ {
				va := feat[a]
				for b := a; b < dimInput; b++ {
					feat[fi] = va * feat[b]
					fi++
				}
			}

			var dot float64
			for d := 0; d < m.dim; d++ {
				dot += m.weights[d] * feat[d]
			}
			dot += m.bias

			pred := sigmoid(dot)
			y := 0.0
			if labels[idx] {
				y = 1.0
			}

			err := pred - y
			w := negWeight
			if y == 1 {
				w = posWeight
			}

			if (pred > 0.5 && y == 1) || (pred < 0.5 && y == 0) {
				correct++
			}

			learnRate := lr / (1 + float64(iter)*0.1)

			for d := 0; d < m.dim; d++ {
				m.weights[d] -= learnRate * err * feat[d] * w
			}
			m.bias -= learnRate * err * w

			if y == 1 {
				totalLoss += -math.Log(math.Max(pred, 1e-15))
			} else {
				totalLoss += -math.Log(math.Max(1-pred, 1e-15))
			}
		}

		acc := float64(correct) / float64(sampleN)
		avgLoss := totalLoss / float64(sampleN)
		printf("  iter %d: loss=%.4f acc=%.4f\n", iter+1, avgLoss, acc)
	}
}

func (m *Model) Predict(v [dimInput]float32) float64 {
	feat := expand(v)
	var dot float64
	for d := 0; d < m.dim; d++ {
		dot += m.weights[d] * feat[d]
	}
	dot += m.bias
	return sigmoid(dot)
}

func (m *Model) WriteTo(w io.Writer) error {
	buf := make([]byte, 0, 8+m.dim*8)
	buf = binary.LittleEndian.AppendUint32(buf, 0x4C4F4752)
	buf = binary.LittleEndian.AppendUint32(buf, 1)
	buf = binary.LittleEndian.AppendUint32(buf, uint32(m.dim))
	buf = binary.LittleEndian.AppendUint64(buf, math.Float64bits(m.bias))
	for i := 0; i < m.dim; i++ {
		buf = binary.LittleEndian.AppendUint64(buf, math.Float64bits(m.weights[i]))
	}
	_, err := w.Write(buf)
	return err
}

func LoadFrom(r io.Reader) (*Model, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return nil, err
	}
	m := &Model{}
	magic := binary.LittleEndian.Uint32(data[0:])
	if magic != 0x4C4F4752 {
		return nil, errInvalidMagic
	}
	version := binary.LittleEndian.Uint32(data[4:])
	if version != 1 {
		return nil, errInvalidVersion
	}
	m.dim = int(binary.LittleEndian.Uint32(data[8:]))
	m.bias = math.Float64frombits(binary.LittleEndian.Uint64(data[12:]))
	m.weights = make([]float64, m.dim)
	for i := 0; i < m.dim; i++ {
		m.weights[i] = math.Float64frombits(binary.LittleEndian.Uint64(data[16+i*8:]))
	}
	return m, nil
}

func (m *Model) Evaluate(vectors []float32, labels []bool, n int) (tp, tn, fp, fn int) {
	for i := 0; i < n; i++ {
		var v [dimInput]float32
		for d := 0; d < dimInput; d++ {
			v[d] = vectors[i*dimInput+d]
		}
		prob := m.Predict(v)
		pred := prob >= threshold
		actual := labels[i]

		if pred && actual {
			tp++
		} else if pred && !actual {
			fp++
		} else if !pred && actual {
			fn++
		} else {
			tn++
		}
	}
	return
}
