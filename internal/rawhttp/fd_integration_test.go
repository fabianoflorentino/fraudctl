package rawhttp

import (
	"fmt"
	"net"
	"os"
	"syscall"
	"testing"
	"time"
)

func TestFullFlowLikeMain(t *testing.T) {
	srv := New(&mockHandler{})

	ctrlPath := "/tmp/test_ctrl_full.sock"
	_ = os.Remove(ctrlPath)
	ctrlLn, err := net.ListenUnix("unix", &net.UnixAddr{Name: ctrlPath, Net: "unix"})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = ctrlLn.Close() }()
	defer func() { _ = os.Remove(ctrlPath) }()

	workerCh := make(chan net.Conn)

	go func() {
		for conn := range workerCh {
			_ = srv.ServeConn(conn)
		}
	}()

	done := make(chan struct{})
	go func() {
		for {
			ctrlConn, err := ctrlLn.Accept()
			if err != nil {
				return
			}
			go serveConn(ctrlConn, workerCh, srv, t)
			done <- struct{}{}
		}
	}()

	time.Sleep(100 * time.Millisecond)

	uc, err := net.DialUnix("unix", nil, &net.UnixAddr{Name: ctrlPath, Net: "unix"})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = uc.Close() }()

	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	addr := l.Addr().String()

	type accepted struct {
		conn net.Conn
		err  error
	}
	acceptCh := make(chan accepted, 1)
	go func() {
		c, err := l.Accept()
		acceptCh <- accepted{c, err}
	}()

	client, err := net.Dial("tcp", addr)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = client.Close() }()

	server := (<-acceptCh).conn
	defer func() { _ = server.Close() }()

	tcpConn := server.(*net.TCPConn)
	f, err := tcpConn.File()
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = f.Close() }()

	rawConn, err := uc.SyscallConn()
	if err != nil {
		t.Fatal(err)
	}

	var sendErr error
	if err := rawConn.Write(func(fd uintptr) bool {
		rights := syscall.UnixRights(int(f.Fd()))
		sendErr = syscall.Sendmsg(int(fd), nil, rights, nil, 0)
		return sendErr == nil
	}); err != nil {
		t.Fatal(err)
	}
	if sendErr != nil {
		t.Fatal(sendErr)
	}

	<-done
	time.Sleep(200 * time.Millisecond)

	req := "GET /ready HTTP/1.1\r\nHost: localhost\r\n\r\n"
	if _, err := client.Write([]byte(req)); err != nil {
		t.Fatal(err)
	}

	_ = client.SetReadDeadline(time.Now().Add(5 * time.Second))
	buf := make([]byte, 4096)
	n, err := client.Read(buf)
	if err != nil {
		t.Fatalf("read error: %v", err)
	}

	fmt.Printf("Response: %s\n", string(buf[:n]))
}

func serveConn(ctrlConn net.Conn, workerCh chan net.Conn, srv *Server, _ *testing.T) {
	defer func() { _ = ctrlConn.Close() }()
	uc := ctrlConn.(*net.UnixConn)
	buf := make([]byte, 1)
	oob := make([]byte, 64)

	for {
		_, oobn, _, _, err := uc.ReadMsgUnix(buf, oob)
		if err != nil {
			return
		}

		cmsgs, err := syscall.ParseSocketControlMessage(oob[:oobn])
		if err != nil || len(cmsgs) == 0 {
			continue
		}

		var fds []int
		for i := range cmsgs {
			if cmsgs[i].Header.Level == syscall.SOL_SOCKET && cmsgs[i].Header.Type == syscall.SCM_RIGHTS {
				parsed, err := syscall.ParseUnixRights(&cmsgs[i])
				if err != nil {
					continue
				}
				fds = append(fds, parsed...)
			}
		}
		if len(fds) == 0 {
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
