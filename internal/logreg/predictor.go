// Package logreg implements a logistic regression model for binary classification.
package logreg

import (
	"os"

	"github.com/fabianoflorentino/fraudctl/internal/model"
)

// Predictor wraps a trained Model and implements the KNNIndex interface
// for compatibility with the fraud score handler.
type Predictor struct {
	model *Model
}

// LoadPredictor loads a trained logistic regression model from a binary file.
func LoadPredictor(path string) (*Predictor, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()
	m, err := LoadFrom(f)
	if err != nil {
		return nil, err
	}
	return &Predictor{model: m}, nil
}

// Predict returns the fraud probability for the given vector.
// Implements the KNNIndex interface (returns probability instead of score).
func (p *Predictor) Predict(v model.Vector14, k int) float64 {
	if p.model == nil {
		return 0
	}
	var arr [dimInput]float32
	copy(arr[:], v[:])
	return p.model.Predict(arr)
}

// Count returns 0 since the Predictor does not store reference vectors.
func (p *Predictor) Count() int { return 0 }

// FraudCount returns 0 since the Predictor does not store reference vectors.
func (p *Predictor) FraudCount() int { return 0 }

// Exists checks if a file exists at the given path.
func Exists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}
