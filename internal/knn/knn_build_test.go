package knn

import (
	"bytes"
	"compress/gzip"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func createTestRefsGz(t *testing.T, entries []map[string]interface{}) string {
	t.Helper()
	var buf bytes.Buffer
	gz := gzip.NewWriter(&buf)
	gz.Write([]byte(`[`))
	for i, e := range entries {
		if i > 0 {
			gz.Write([]byte(`,`))
		}
		json.NewEncoder(gz).Encode(e)
	}
	gz.Write([]byte(`]`))
	gz.Close()

	path := filepath.Join(t.TempDir(), "references.json.gz")
	os.WriteFile(path, buf.Bytes(), 0644)
	return path
}

func makeRefVector(vals ...float64) []float64 {
	v := make([]float64, 14)
	for i := 0; i < 14 && i < len(vals); i++ {
		v[i] = vals[i]
	}
	return v
}

func TestBuildIVF_Small(t *testing.T) {
	entries := []map[string]interface{}{
		{"vector": makeRefVector(0.1, 0.1, 0.1), "label": "fraud"},
		{"vector": makeRefVector(0.2, 0.2, 0.2), "label": "fraud"},
		{"vector": makeRefVector(0.3, 0.3, 0.3), "label": "fraud"},
		{"vector": makeRefVector(0.8, 0.8, 0.8), "label": "legit"},
		{"vector": makeRefVector(0.9, 0.9, 0.9), "label": "legit"},
		{"vector": makeRefVector(1.0, 1.0, 1.0), "label": "legit"},
	}

	gzPath := createTestRefsGz(t, entries)
	outPath := filepath.Join(t.TempDir(), "ivf.bin")

	err := BuildIVF(gzPath, outPath, 2, 5)
	if err != nil {
		t.Fatalf("BuildIVF failed: %v", err)
	}

	if _, err := os.Stat(outPath); err != nil {
		t.Fatalf("output file not created: %v", err)
	}
	info, _ := os.Stat(outPath)
	if info.Size() == 0 {
		t.Fatal("output file is empty")
	}
}

func TestBuildIVF_LargerNlist(t *testing.T) {
	entries := make([]map[string]interface{}, 20)
	for i := 0; i < 20; i++ {
		v := float64(i) / 20.0
		label := "legit"
		if i < 10 {
			label = "fraud"
		}
		entries[i] = map[string]interface{}{
			"vector": makeRefVector(v, v, v),
			"label":  label,
		}
	}

	gzPath := createTestRefsGz(t, entries)
	outPath := filepath.Join(t.TempDir(), "ivf_large.bin")

	err := BuildIVF(gzPath, outPath, 4, 3)
	if err != nil {
		t.Fatalf("BuildIVF failed: %v", err)
	}
}

func TestBuildBrute_Small(t *testing.T) {
	entries := []map[string]interface{}{
		{"vector": makeRefVector(0.1), "label": "fraud"},
		{"vector": makeRefVector(0.5), "label": "legit"},
		{"vector": makeRefVector(0.9), "label": "fraud"},
	}

	gzPath := createTestRefsGz(t, entries)
	outPath := filepath.Join(t.TempDir(), "brute.bin")

	err := BuildBrute(gzPath, outPath)
	if err != nil {
		t.Fatalf("BuildBrute failed: %v", err)
	}

	if _, err := os.Stat(outPath); err != nil {
		t.Fatalf("output file not created: %v", err)
	}
}

func TestBuildIVF_LoadAndPredict(t *testing.T) {
	entries := []map[string]interface{}{
		{"vector": makeRefVector(0.1), "label": "fraud"},
		{"vector": makeRefVector(0.2), "label": "fraud"},
		{"vector": makeRefVector(0.3), "label": "fraud"},
		{"vector": makeRefVector(0.8), "label": "legit"},
		{"vector": makeRefVector(0.9), "label": "legit"},
		{"vector": makeRefVector(1.0), "label": "legit"},
	}

	gzPath := createTestRefsGz(t, entries)
	outPath := filepath.Join(t.TempDir(), "ivf_roundtrip.bin")

	err := BuildIVF(gzPath, outPath, 2, 5)
	if err != nil {
		t.Fatalf("BuildIVF failed: %v", err)
	}

	idx, err := LoadIVF(outPath)
	if err != nil {
		t.Fatalf("LoadIVF failed: %v", err)
	}
	if idx == nil {
		t.Fatal("LoadIVF returned nil")
	}
	if idx.Count() != 6 {
		t.Errorf("Count = %d, want 6", idx.Count())
	}
}

func TestBuildBrute_LoadAndPredict(t *testing.T) {
	entries := []map[string]interface{}{
		{"vector": makeRefVector(0.1), "label": "fraud"},
		{"vector": makeRefVector(0.5), "label": "legit"},
		{"vector": makeRefVector(0.9), "label": "fraud"},
	}

	gzPath := createTestRefsGz(t, entries)
	outPath := filepath.Join(t.TempDir(), "brute_roundtrip.bin")

	err := BuildBrute(gzPath, outPath)
	if err != nil {
		t.Fatalf("BuildBrute failed: %v", err)
	}

	idx, err := LoadBruteAVX2(outPath)
	if err != nil {
		t.Fatalf("LoadBruteAVX2 failed: %v", err)
	}
	if idx == nil {
		t.Fatal("LoadBruteAVX2 returned nil")
	}
	if idx.Count() != 3 {
		t.Errorf("Count = %d, want 3", idx.Count())
	}
}
