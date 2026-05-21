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

	debug.SetMemoryLimit(150 << 20)

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
	var start time.Time
	if middleware.IsEnabled() {
		start = time.Now()
	}
	count := h.fraudHandler.HandleFraudScore(body)

	if middleware.IsEnabled() {
		elapsed := time.Since(start)
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

		fd, err := parseUnixRights(oob[:oobn])
		if err != nil || fd < 0 {
			continue
		}

		file := os.NewFile(uintptr(fd), "")
		conn, err := net.FileConn(file)
		_ = file.Close()
		if err != nil {
			continue
		}

		if tc, ok := conn.(*net.TCPConn); ok {
			_ = tc.SetNoDelay(true)
			_ = tc.SetKeepAlive(true)
		}

		go srv.ServeConn(conn) //nolint:errcheck
	}
}

func parseUnixRights(oob []byte) (int, error) {
	// Manual parse to avoid allocation from ParseSocketControlMessage.
	// SCM_RIGHTS layout: 16-byte cmsghdr + 4-byte fd (on amd64).
	if len(oob) < 20 {
		return -1, nil
	}
	level := int(oob[8]) | int(oob[9])<<8 | int(oob[10])<<16 | int(oob[11])<<24
	typ := int(oob[12]) | int(oob[13])<<8 | int(oob[14])<<16 | int(oob[15])<<24
	if level != syscall.SOL_SOCKET || typ != syscall.SCM_RIGHTS {
		return -1, nil
	}
	fd := int(oob[16]) | int(oob[17])<<8 | int(oob[18])<<16 | int(oob[19])<<24
	return fd, nil
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
