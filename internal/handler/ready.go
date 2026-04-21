// Package handler provides HTTP handlers for the fraud detection API.
//
// This file provides the Ready handler for health checks.
package handler

import "net/http"

// Ready is a health check handler.
// It returns HTTP 200 OK when the service is ready to handle requests.
//
// This endpoint is used by load balancers and orchestration systems
// to check if the service is healthy.
func Ready(w ResponseWriter, r *http.Request) error {
	w.WriteHeader(http.StatusOK)
	w.Write([]byte("OK"))
	return nil
}
