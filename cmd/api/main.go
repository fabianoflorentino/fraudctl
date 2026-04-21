package main

import (
	"fraudctl/internal/handler"
	"log"
	"net/http"
)

func main() {
	router := handler.NewRouter()
	router.Handle("/ready", handler.Ready)

	log.Println("Server starting on port 9999")
	if err := http.ListenAndServe(":9999", router); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}
