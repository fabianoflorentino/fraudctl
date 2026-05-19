package main

import (
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	_ "net/http/pprof"
	"os"
	"os/signal"
	"runtime"
	"runtime/debug"
	"syscall"
	"time"

	"github.com/fabianoflorentino/fraudctl/internal/dataset"
	"github.com/fabianoflorentino/fraudctl/internal/handler"
	"github.com/fabianoflorentino/fraudctl/internal/middleware"
	"github.com/fabianoflorentino/fraudctl/internal/rawhttp"
)

var (
	resourcesPath = flag.String("resources", "/resources", "path to resources directory")
	healthCheck   = flag.Bool("healthcheck", false, "run healthcheck and exit")
	pprofAddr     = flag.String("pprof", "", "pprof listen address (e.g. :6060)")
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

	if *pprofAddr != "" {
		go func() {
			log.Printf("pprof on %s", *pprofAddr)
			if err := http.ListenAndServe(*pprofAddr, nil); err != nil {
				log.Printf("pprof: %v", err)
			}
		}()
	}

	srv := rawhttp.New(&httpHandler{fraudHandler: fraudHandler})

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

	startWorkerPool(srv)

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	log.Println("shutting down")
	_ = ctrlLn.Close()
}

type httpHandler struct {
	fraudHandler *handler.FraudScoreHandler
}

func (h *httpHandler) ServeFraudScore(body []byte) []byte {
	start := time.Now()
	count := h.fraudHandler.HandleFraudScore(body)
	elapsed := time.Since(start)

	if middleware.IsEnabled() {
		approved := count < 3
		parseErr := count == 0 && !isLikelyValidJSON(body)
		middleware.Record(elapsed, elapsed/2, elapsed/2, count, approved, parseErr)
	}

	return rawhttp.FraudResponse(count)
}

func (h *httpHandler) ServeReady() []byte {
	return rawhttp.ReadyResponse()
}

func isLikelyValidJSON(body []byte) bool {
	for _, b := range body {
		if b == '{' {
			return true
		}
		if b != ' ' && b != '\t' && b != '\n' && b != '\r' {
			return false
		}
	}
	return false
}

var workerCh = make(chan net.Conn)

func startWorkerPool(srv *rawhttp.Server) {
	n := runtime.GOMAXPROCS(0)
	for i := 0; i < n; i++ {
		go func() {
			for conn := range workerCh {
				_ = srv.ServeConn(conn)
			}
		}()
	}
}

func serveControl(ctrlConn net.Conn, srv *rawhttp.Server) {
	defer func() { _ = ctrlConn.Close() }()
	uc := ctrlConn.(*net.UnixConn)
	buf := make([]byte, 1)
	oob := make([]byte, 64)

	for {
		n, oobn, _, _, err := uc.ReadMsgUnix(buf, oob)
		if err != nil {
			return
		}
		_ = n

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

		if tc, ok := conn.(*net.TCPConn); ok {
			_ = tc.SetNoDelay(true)
			_ = tc.SetKeepAlive(true)
		}

		select {
		case workerCh <- conn:
		default:
			go func(c net.Conn) { _ = srv.ServeConn(c) }(conn)
		}
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
