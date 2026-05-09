package main

import (
	"flag"
	"testing"
)

func TestBuildIndexFlags(t *testing.T) {
	fs := flag.NewFlagSet("test", flag.ContinueOnError)
	resources := fs.String("resources", "./resources", "")
	nlist := fs.Int("nlist", 500, "")
	iterations := fs.Int("iterations", 20, "")

	fs.Parse([]string{"-resources", "/data/resources", "-nlist", "100", "-iterations", "10"})

	if *resources != "/data/resources" {
		t.Errorf("resources = %q, want /data/resources", *resources)
	}
	if *nlist != 100 {
		t.Errorf("nlist = %d, want 100", *nlist)
	}
	if *iterations != 10 {
		t.Errorf("iterations = %d, want 10", *iterations)
	}
}

func TestBuildIndexDefaultFlags(t *testing.T) {
	fs := flag.NewFlagSet("test", flag.ContinueOnError)
	resources := fs.String("resources", "./resources", "")
	nlist := fs.Int("nlist", 500, "")
	iterations := fs.Int("iterations", 20, "")

	fs.Parse(nil)

	if *resources != "./resources" {
		t.Errorf("default resources = %q, want ./resources", *resources)
	}
	if *nlist != 500 {
		t.Errorf("default nlist = %d, want 500", *nlist)
	}
	if *iterations != 20 {
		t.Errorf("default iterations = %d, want 20", *iterations)
	}
}
