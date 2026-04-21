package dataset

import (
	"compress/gzip"
	"encoding/json"
	"errors"
	"fraudctl/internal/model"
	"io"
	"os"
	"path/filepath"
)

var (
	ErrInvalidPath = errors.New("invalid file path")
	ErrReadFile    = errors.New("failed to read file")
	ErrOpenGzip    = errors.New("failed to open gzip")
	ErrDecodeJSON  = errors.New("failed to decode JSON")
)

type Loader struct {
	basePath string
}

func NewLoader(basePath string) *Loader {
	return &Loader{basePath: basePath}
}

func (l *Loader) LoadNormalization(path string) (model.NormalizationConstants, error) {
	var norm model.NormalizationConstants

	if path == "" {
		path = filepath.Join(l.basePath, "normalization.json")
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return norm, errors.New(ErrReadFile.Error() + ": " + path)
	}

	if err := json.Unmarshal(data, &norm); err != nil {
		return norm, errors.New(ErrDecodeJSON.Error() + ": normalization")
	}

	return norm, nil
}

func (l *Loader) LoadMCCRisk(path string) (model.MCCRisk, error) {
	var risk model.MCCRisk

	if path == "" {
		path = filepath.Join(l.basePath, "mcc_risk.json")
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return risk, errors.New(ErrReadFile.Error() + ": " + path)
	}

	if err := json.Unmarshal(data, &risk); err != nil {
		return risk, errors.New(ErrDecodeJSON.Error() + ": mcc_risk")
	}

	return risk, nil
}

func (l *Loader) LoadReferences(path string) ([]model.Reference, error) {
	if path == "" {
		path = filepath.Join(l.basePath, "references.json.gz")
	}

	file, err := os.Open(path)
	if err != nil {
		return nil, errors.New(ErrOpenGzip.Error() + ": " + path)
	}
	defer file.Close()

	gz, err := gzip.NewReader(file)
	if err != nil {
		return nil, errors.New(ErrOpenGzip.Error() + ": create reader")
	}
	defer gz.Close()

	data, err := io.ReadAll(gz)
	if err != nil {
		return nil, errors.New(ErrReadFile.Error() + ": gzip content")
	}

	var references []model.Reference
	if err := json.Unmarshal(data, &references); err != nil {
		return nil, errors.New(ErrDecodeJSON.Error() + ": references")
	}

	return references, nil
}

func (l *Loader) LoadAll() (*Dataset, error) {
	norm, err := l.LoadNormalization("")
	if err != nil {
		return nil, err
	}

	mccRisk, err := l.LoadMCCRisk("")
	if err != nil {
		return nil, err
	}

	references, err := l.LoadReferences("")
	if err != nil {
		return nil, err
	}

	ds := NewDataset(references)
	ds.SetConfig(norm, mccRisk)

	return ds, nil
}
