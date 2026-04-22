// Package dataset provides functionality for loading and managing the reference dataset.
//
// This package handles loading of the reference data files required for the fraud
// detection KNN algorithm:
//
//   - references.json.gz: 100k labeled reference vectors
//   - mcc_risk.json: MCC code to risk score mapping
//   - normalization.json: Constants for feature normalization
//
// # Usage
//
// The package provides a Loader for loading individual files and a convenience
// function LoadDefault for loading all files at once:
//
//	ds, err := dataset.LoadDefault("./rinha2026/resources")
//	if err != nil {
//	    log.Fatal(err)
//	}
//
//	knn := ds.KNN(1)
//	vectorizer := ds.Vectorizer()
//
// For testing, use NewLoader with a custom base path:
//
//	loader := dataset.NewLoader("/path/to/resources")
//	norm, err := loader.LoadNormalization("")
package dataset

import (
	"compress/gzip"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"

	"github.com/fabianoflorentino/fraudctl/internal/model"
)

// Common errors returned by Loader methods.
var (
	// ErrInvalidPath is returned when the file path is invalid.
	ErrInvalidPath = errors.New("invalid file path")
	// ErrReadFile is returned when a file cannot be read.
	ErrReadFile = errors.New("failed to read file")
	// ErrOpenGzip is returned when a gzip file cannot be opened.
	ErrOpenGzip = errors.New("failed to open gzip")
	// ErrDecodeJSON is returned when JSON decoding fails.
	ErrDecodeJSON = errors.New("failed to decode JSON")
)

// Loader handles loading of reference data files.
// It accepts a base path to the resources directory.
type Loader struct {
	basePath string
}

// NewLoader creates a new Loader with the specified base path.
// If basePath is empty, it defaults to "./resources".
func NewLoader(basePath string) *Loader {
	return &Loader{basePath: basePath}
}

// LoadNormalization loads the normalization constants from a JSON file.
// If path is empty, it uses the basePath from the Loader.
//
// Returns ErrReadFile if the file cannot be read.
// Returns ErrDecodeJSON if the file contains invalid JSON.
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

// LoadMCCRisk loads the MCC risk scores from a JSON file.
// If path is empty, it uses the basePath from the Loader.
//
// Returns ErrReadFile if the file cannot be read.
// Returns ErrDecodeJSON if the file contains invalid JSON.
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

// LoadReferences loads the reference vectors from a gzipped JSON file.
// If path is empty, it uses the basePath from the Loader.
//
// The file is expected to be in the format:
//
//	[
//	  {"vector": [0.1, 0.2, ...], "label": "fraud"},
//	  {"vector": [0.9, 0.8, ...], "label": "legit"}
//	]
//
// Returns ErrOpenGzip if the file cannot be opened.
// Returns ErrReadFile if the gzip content cannot be read.
// Returns ErrDecodeJSON if the file contains invalid JSON.
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

// LoadAll loads all reference data files and returns a Dataset.
// This is a convenience function that calls LoadNormalization, LoadMCCRisk,
// and LoadReferences in sequence.
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
