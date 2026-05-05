package main

import (
	"flag"
	"fmt"
	"log"
	"net"
	"os"
	"os/signal"
	"runtime"
	"strconv"
	"syscall"
	"time"

	"github.com/valyala/fasthttp"

	"github.com/fabianoflorentino/fraudctl/internal/dataset"
	"github.com/fabianoflorentino/fraudctl/internal/handler"
)

var (
	resourcesPath = flag.String("resources", "/resources", "Path to resources directory")
	healthCheck   = flag.Bool("healthcheck", false, "Run healthcheck and exit")
)

func main() {
	runtime.GOMAXPROCS(1)
	runtime.LockOSThread()

	flag.Parse()

	if *healthCheck {
		if err := checkHealth(); err != nil {
			log.Fatalf("Healthcheck failed: %v", err)
		}
		log.Println("Healthcheck OK")
		return
	}

	port := 9999
	if p := os.Getenv("PORT"); p != "" {
		if parsed, err := strconv.Atoi(p); err == nil {
			port = parsed
		}
	}
	socketPath := os.Getenv("UNIX_SOCKET")

	log.Printf("Loading dataset (IVF index) from %s ...", *resourcesPath)
	ds, err := dataset.LoadDefault(*resourcesPath)
	if err != nil {
		log.Fatalf("Failed to load dataset: %v", err)
	}
	log.Printf("Dataset loaded: %d vectors (%d fraud)", ds.Count(), ds.FraudCount())

	fraudHandler := handler.NewFraudScoreHandler(ds.Vectorizer(), ds.KNN())

	requestHandler := func(ctx *fasthttp.RequestCtx) {
		switch string(ctx.Path()) {
		case "/ready":
			handler.Ready(ctx)
		case "/fraud-score":
			fraudHandler.Handle(ctx)
		default:
			ctx.SetStatusCode(fasthttp.StatusNotFound)
		}
	}

	var ln net.Listener
	if socketPath != "" {
		_ = os.Remove(socketPath)
		ln, err = net.Listen("unix", socketPath)
		if err != nil {
			log.Fatalf("Failed to listen on unix socket %s: %v", socketPath, err)
		}
		if err := os.Chmod(socketPath, 0666); err != nil {
			log.Fatalf("Failed to chmod socket: %v", err)
		}
		log.Printf("Server starting on unix socket %s (fasthttp)", socketPath)
	} else {
		ln, err = net.Listen("tcp", fmt.Sprintf(":%d", port))
		if err != nil {
			log.Fatalf("Failed to listen on port %d: %v", port, err)
		}
		log.Printf("Server starting on port %d (fasthttp)", port)
	}

	srv := &fasthttp.Server{
		Handler:               requestHandler,
		ReadTimeout:           2 * time.Second,
		WriteTimeout:          5 * time.Second,
		IdleTimeout:           30 * time.Second,
		MaxRequestBodySize:    4096,
		NoDefaultServerHeader: true,
		NoDefaultContentType:  true,
		ReadBufferSize:        4096,
		WriteBufferSize:       4096,
		Concurrency:           1024,
	}

	go func() {
		if err := srv.Serve(ln); err != nil {
			log.Fatalf("Server failed: %v", err)
		}
	}()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	log.Println("Shutting down server...")

	if err := srv.Shutdown(); err != nil {
		log.Fatalf("Server forced to shutdown: %v", err)
	}

	log.Println("Server exited properly")
}

func checkHealth() error {
	socketPath := os.Getenv("UNIX_SOCKET")

	c := &fasthttp.Client{}

	var uri string
	if socketPath != "" {
		uri = "http://unix/ready"
		c.Dial = func(addr string) (net.Conn, error) {
			return net.Dial("unix", socketPath)
		}
	} else {
		uri = "http://localhost:9999/ready"
	}

	status, _, err := c.Get(nil, uri)
	if err != nil {
		return err
	}
	if status != fasthttp.StatusOK {
		return fmt.Errorf("unexpected status: %d", status)
	}
	return nil
}

func init() {
	log.SetOutput(os.Stderr)
}
