package main

import (
	"flag"
	"log"
	"net"
	"os"
	"os/signal"
	"strings"
	"sync/atomic"
	"syscall"
	"time"
)

var (
	listenAddr  = flag.String("listen", ":9999", "TCP listen address")
	workerAddrs = flag.String("workers", "/sockets/lb-ctrl-1.sock,/sockets/lb-ctrl-2.sock", "comma-separated worker control socket paths")
)

func main() {
	flag.Parse()

	paths := strings.Split(*workerAddrs, ",")
	cons := make([]*net.UnixConn, len(paths))
	for i, p := range paths {
		p = strings.TrimSpace(p)
		conn, err := net.DialUnix("unix", nil, &net.UnixAddr{Name: p, Net: "unix"})
		if err != nil {
			log.Fatalf("dial worker %s: %v", p, err)
		}
		cons[i] = conn
	}
	log.Printf("connected to %d workers", len(cons))

	ln, err := net.Listen("tcp", *listenAddr)
	if err != nil {
		log.Fatalf("listen: %v", err)
	}

	tcpLn := ln.(*net.TCPListener)
	if raw, err := tcpLn.SyscallConn(); err == nil {
		_ = raw.Control(func(fd uintptr) {
			_ = syscall.SetsockoptInt(int(fd), syscall.IPPROTO_TCP, syscall.TCP_DEFER_ACCEPT, 3)
		})
	}

	log.Printf("listening on %s", *listenAddr)

	go func() {
		ticker := time.NewTicker(30 * time.Second)
		defer ticker.Stop()
		for range ticker.C {
			deadline := time.Now().Add(60 * time.Second)
			for _, c := range cons {
				c.SetWriteDeadline(deadline)
			}
		}
	}()

	var rr uint64
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		<-sigCh
		ln.Close()
		for _, c := range cons {
			c.Close()
		}
	}()

	for {
		tcpConn, err := ln.Accept()
		if err != nil {
			break
		}

		// Disable Nagle on the accepted client connection before passing the
		// FD via SCM_RIGHTS. Without this, the first response byte can be
		// delayed by up to 40ms (Nagle + delayed ACK interaction) when the
		// API writes a small HTTP response (< MSS + not full-sized segment).
		if tc, ok := tcpConn.(*net.TCPConn); ok {
			_ = tc.SetNoDelay(true)
		}

		wi := atomic.AddUint64(&rr, 1) % uint64(len(cons))

		raw, err := tcpConn.(*net.TCPConn).SyscallConn()
		if err != nil {
			tcpConn.Close()
			continue
		}

		var fd int
		if err := raw.Control(func(f uintptr) {
			fd = int(f)
		}); err != nil {
			tcpConn.Close()
			continue
		}

		rights := syscall.UnixRights(fd)
		if _, _, err = cons[wi].WriteMsgUnix(nil, rights, nil); err != nil {
			log.Printf("write to worker %d: %v", wi, err)
		}
		tcpConn.Close()
	}
}
