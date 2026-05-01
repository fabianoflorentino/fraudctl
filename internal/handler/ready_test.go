package handler

import (
	"testing"

	"github.com/valyala/fasthttp"
)

func TestReady(t *testing.T) {
	tests := []struct {
		name     string
		method   string
		wantCode int
		wantBody string
	}{
		{"GET returns OK", "GET", fasthttp.StatusOK, "OK"},
		{"POST returns OK", "POST", fasthttp.StatusOK, "OK"},
		{"PUT returns OK", "PUT", fasthttp.StatusOK, "OK"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := &fasthttp.RequestCtx{}
			ctx.Request.Header.SetMethod(tt.method)
			ctx.Request.SetRequestURI("/ready")

			Ready(ctx)

			if ctx.Response.StatusCode() != tt.wantCode {
				t.Errorf("Ready() code = %v, want %v", ctx.Response.StatusCode(), tt.wantCode)
			}
			if string(ctx.Response.Body()) != tt.wantBody {
				t.Errorf("Ready() body = %q, want %q", ctx.Response.Body(), tt.wantBody)
			}
		})
	}
}
