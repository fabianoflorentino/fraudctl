package main

import (
	"flag"
	"fmt"
	"log"
	"net"
	"os"
	"os/signal"
	"runtime"
	"runtime/debug"
	"syscall"
	"time"

	"github.com/valyala/fasthttp"

	"github.com/fabianoflorentino/fraudctl/internal/dataset"
	"github.com/fabianoflorentino/fraudctl/internal/handler"
	"github.com/fabianoflorentino/fraudctl/internal/middleware"
)

var (
	resourcesPath = flag.String("resources", "/resources", "path to resources directory")
	healthCheck   = flag.Bool("healthcheck", false, "run healthcheck and exit")
)

func main() {
	flag.Parse()

	if *healthCheck {
		if err := checkHealth(); err != nil {
			log.Fatalf("healthcheck: %v", err)
		}
		log.Println("healthcheck OK")
		return
	}

	telemetryEnabled := os.Getenv("TELEMETRY_ENABLED") != "false"
	middleware.SetEnabled(telemetryEnabled)

	log.Printf("loading dataset from %s ...", *resourcesPath)
	ds, err := dataset.LoadDefault(*resourcesPath)
	if err != nil {
		log.Fatalf("dataset: %v", err)
	}
	log.Printf("dataset loaded: %d vectors (%d fraud)", ds.Count(), ds.FraudCount())

	runtime.GC()
	debug.FreeOSMemory()

	fraudHandler := handler.NewFraudScoreHandler(ds.Vectorizer(), ds.KNN())

	if telemetryEnabled {
		middleware.StartReporter(10 * time.Second)
	} else {
		log.Print("telemetry disabled")
	}

	srv := &fasthttp.Server{
		Handler: func(ctx *fasthttp.RequestCtx) {
			switch {
			case len(ctx.Path()) == 6 && ctx.Path()[1] == 'r':
				handler.Ready(ctx)
			case len(ctx.Path()) == 12 && ctx.Path()[1] == 'f':
				fraudHandler.Handle(ctx)
			default:
				ctx.SetStatusCode(fasthttp.StatusNotFound)
			}
		},
		ReadTimeout:                   750 * time.Millisecond,
		WriteTimeout:                  750 * time.Millisecond,
		IdleTimeout:                   10 * time.Second,
		MaxRequestBodySize:            4 * 1024,
		NoDefaultServerHeader:         true,
		NoDefaultContentType:          true,
		ReadBufferSize:                4096,
		WriteBufferSize:               4096,
		Concurrency:                   4096,
		DisableHeaderNamesNormalizing: true,
		DisablePreParseMultipartForm:  true,
		ReduceMemoryUsage:             false,
	}

	ctrlSocket := os.Getenv("CTRL_SOCKET")
	if ctrlSocket == "" {
		log.Fatal("CTRL_SOCKET not set")
	}
	_ = os.Remove(ctrlSocket)
	ctrlLn, err := net.Listen("unix", ctrlSocket)
	if err != nil {
		log.Fatalf("ctrl listen: %v", err)
	}
	if err := os.Chmod(ctrlSocket, 0666); err != nil {
		log.Fatalf("chmod: %v", err)
	}
	log.Printf("control socket at %s", ctrlSocket)

	go func() {
		for {
			ctrlConn, err := ctrlLn.Accept()
			if err != nil {
				return
			}
			go serveControl(ctrlConn, srv)
		}
	}()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	log.Println("shutting down")
	_ = ctrlLn.Close()
	_ = srv.Shutdown()
}

func serveControl(ctrlConn net.Conn, srv *fasthttp.Server) {
	defer func() { _ = ctrlConn.Close() }()
	uc := ctrlConn.(*net.UnixConn)
	buf := make([]byte, 1)
	oob := make([]byte, 64)

	for {
		_, oobn, _, _, err := uc.ReadMsgUnix(buf, oob)
		if err != nil {
			return
		}

		fds, err := parseUnixRights(oob[:oobn])
		if err != nil || len(fds) == 0 {
			continue
		}

		file := os.NewFile(uintptr(fds[0]), "")
		conn, err := net.FileConn(file)
		_ = file.Close()
		if err != nil {
			continue
		}

		go func() { _ = srv.ServeConn(conn) }()
	}
}

func parseUnixRights(oob []byte) ([]int, error) {
	cmsgs, err := syscall.ParseSocketControlMessage(oob)
	if err != nil {
		return nil, err
	}
	var fds []int
	for i := range cmsgs {
		if cmsgs[i].Header.Level == syscall.SOL_SOCKET && cmsgs[i].Header.Type == syscall.SCM_RIGHTS {
			parsed, err := syscall.ParseUnixRights(&cmsgs[i])
			if err != nil {
				return nil, err
			}
			fds = append(fds, parsed...)
		}
	}
	return fds, nil
}

func checkHealth() error {
	ctrlSocket := os.Getenv("CTRL_SOCKET")
	if ctrlSocket == "" {
		return fmt.Errorf("CTRL_SOCKET not set")
	}
	conn, err := net.DialTimeout("unix", ctrlSocket, time.Second)
	if err != nil {
		return err
	}
	_ = conn.Close()
	return nil
}

func init() {
	log.SetOutput(os.Stderr)
}
