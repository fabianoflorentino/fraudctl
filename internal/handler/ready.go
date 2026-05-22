// Package handler provides HTTP request handlers for the fraud detection API.
package handler

// Ready returns the precomputed response body for the /ready health check endpoint.
func Ready() []byte {
	return []byte("OK")
}
