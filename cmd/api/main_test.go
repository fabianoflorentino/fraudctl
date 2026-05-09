package main

import (
	"testing"
)

func TestHealthCheckFlags(t *testing.T) {
	if resourcesPath == nil {
		t.Error("resourcesPath flag is nil")
	}
	if healthCheck == nil {
		t.Error("healthCheck flag is nil")
	}
}

func TestInit(t *testing.T) {
	// init() just sets log output, nothing to assert
}
