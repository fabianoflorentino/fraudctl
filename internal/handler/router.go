// Package handler provides HTTP handlers for the fraud detection API.
//
// This file provides the Router for managing HTTP routes.
package handler

import (
	"net/http"
)

// Router manages HTTP routes using http.ServeMux.
type Router struct {
	mux *http.ServeMux
}

// NewRouter creates a new Router.
func NewRouter() *Router {
	return &Router{mux: http.NewServeMux()}
}

// Handle registers a HandlerFunc for the given path.
func (r *Router) Handle(path string, handler HandlerFunc) {
	r.mux.Handle(path, Adapt(handler))
}

// HandleFunc registers a function as a handler for the given path.
func (r *Router) HandleFunc(path string, f func(w ResponseWriter, r *http.Request) error) {
	r.mux.Handle(path, Adapt(f))
}

// ServeHTTP implements http.Handler for the Router.
func (r *Router) ServeHTTP(w http.ResponseWriter, rq *http.Request) {
	r.mux.ServeHTTP(w, rq)
}
