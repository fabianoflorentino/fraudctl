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
	log.Printf("listening on %s", *listenAddr)

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

		wi := atomic.AddUint64(&rr, 1) % uint64(len(cons))

		f, err := tcpConn.(*net.TCPConn).File()
		if err != nil {
			tcpConn.Close()
			continue
		}
		fd := f.Fd()

		rights := syscall.UnixRights(int(fd))
		cons[wi].SetWriteDeadline(time.Now().Add(50 * time.Millisecond))
		_, _, err = cons[wi].WriteMsgUnix(nil, rights, nil)

		f.Close()
		tcpConn.Close()

		if err != nil {
			log.Printf("write to worker %d: %v", wi, err)
		}
	}
}
