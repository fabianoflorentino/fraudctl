package handler

import (
	"testing"
)

func TestReady(t *testing.T) {
	resp := Ready()
	if string(resp) != "OK" {
		t.Errorf("Ready() = %q; want %q", resp, "OK")
	}
}
