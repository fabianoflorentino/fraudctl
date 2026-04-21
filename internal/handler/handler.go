// Package handler provides HTTP handlers for the fraud detection API.
//
// The package implements the HTTP layer following SOLID principles:
//   - DIP: Handlers depend on ResponseWriter interface, not net/http concrete types
//   - ISP: Small, focused interfaces
//   - SRP: Each handler has a single responsibility
//
// # ResponseWriter Interface
//
// The ResponseWriter interface abstracts http.ResponseWriter,
// allowing for easier testing and potential future changes.
//
//	type ResponseWriter interface {
//	    WriteHeader(statusCode int)
//	    Write(body []byte) (int, error)
//	}
//
// # HandlerFunc Type
//
// Handlers are defined as:
//
//	type HandlerFunc func(w ResponseWriter, r *http.Request) error
//
// Return nil for success, or an error that will be converted to HTTP 500.
package handler

import (
	"io"
	"net/http"
)

// ResponseWriter is an interface that abstracts http.ResponseWriter.
// This allows handlers to be tested without a real HTTP connection
// and provides flexibility for future changes.
type ResponseWriter interface {
	WriteHeader(statusCode int)
	Write(body []byte) (int, error)
}

// ResponseWriterAdapter adapts http.ResponseWriter to the ResponseWriter interface.
type ResponseWriterAdapter struct {
	W http.ResponseWriter
}

// WriteHeader delegates to the underlying http.ResponseWriter.
func (a *ResponseWriterAdapter) WriteHeader(statusCode int) {
	a.W.WriteHeader(statusCode)
}

// Write delegates to the underlying http.ResponseWriter.
func (a *ResponseWriterAdapter) Write(body []byte) (int, error) {
	return a.W.Write(body)
}

// NewResponseWriterAdapter creates an adapter for http.ResponseWriter.
func NewResponseWriterAdapter(w http.ResponseWriter) ResponseWriter {
	return &ResponseWriterAdapter{W: w}
}

// HandlerFunc is the function signature for HTTP handlers.
// Return nil for success, or an error that will result in HTTP 500.
type HandlerFunc func(w ResponseWriter, r *http.Request) error

// Adapt converts a HandlerFunc to http.HandlerFunc.
// Any error returned by the handler is converted to HTTP 500.
func Adapt(h HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if err := h(NewResponseWriterAdapter(w), r); err != nil {
			w.WriteHeader(http.StatusInternalServerError)
			io.WriteString(w, err.Error())
		}
	}
}