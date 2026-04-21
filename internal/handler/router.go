package handler

import (
	"net/http"
)

type Router struct {
	mux *http.ServeMux
}

func NewRouter() *Router {
	return &Router{mux: http.NewServeMux()}
}

func (r *Router) Handle(path string, handler HandlerFunc) {
	r.mux.Handle(path, Adapt(handler))
}

func (r *Router) HandleFunc(path string, f func(w ResponseWriter, r *http.Request) error) {
	r.mux.Handle(path, Adapt(f))
}

func (r *Router) ServeHTTP(w http.ResponseWriter, rq *http.Request) {
	r.mux.ServeHTTP(w, rq)
}