package logreg

import (
	"os"

	"github.com/fabianoflorentino/fraudctl/internal/model"
)

type Predictor struct {
	model *Model
}

func LoadPredictor(path string) (*Predictor, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	m, err := LoadFrom(f)
	if err != nil {
		return nil, err
	}
	return &Predictor{model: m}, nil
}

func (p *Predictor) Predict(v model.Vector14, k int) float64 {
	if p.model == nil {
		return 0
	}
	var arr [dimInput]float32
	copy(arr[:], v[:])
	return p.model.Predict(arr)
}

func (p *Predictor) Count() int     { return 0 }
func (p *Predictor) FraudCount() int { return 0 }

func Exists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}
