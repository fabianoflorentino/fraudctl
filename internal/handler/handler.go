package handler

import (
	"io"
	"net/http"
)

type ResponseWriter interface {
	WriteHeader(statusCode int)
	Write(body []byte) (int, error)
}

type ResponseWriterAdapter struct {
	W http.ResponseWriter
}

func (a *ResponseWriterAdapter) WriteHeader(statusCode int) {
	a.W.WriteHeader(statusCode)
}

func (a *ResponseWriterAdapter) Write(body []byte) (int, error) {
	return a.W.Write(body)
}

func NewResponseWriterAdapter(w http.ResponseWriter) ResponseWriter {
	return &ResponseWriterAdapter{W: w}
}

type HandlerFunc func(w ResponseWriter, r *http.Request) error

func Adapt(h HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if err := h(NewResponseWriterAdapter(w), r); err != nil {
			w.WriteHeader(http.StatusInternalServerError)
			io.WriteString(w, err.Error())
		}
	}
}
