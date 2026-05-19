package rawhttp

import (
	"fmt"
	"net"
	"os"
	"syscall"
	"testing"
	"time"
)

func TestServeConnViaSCMRights(t *testing.T) {
	srv := New(&mockHandler{})

	ctrlAddr := &net.UnixAddr{Name: "/tmp/test_fd_ctrl.sock", Net: "unix"}
	_ = os.Remove(ctrlAddr.Name)
	ctrlLn, err := net.ListenUnix("unix", ctrlAddr)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = ctrlLn.Close() }()
	defer func() { _ = os.Remove(ctrlAddr.Name) }()

	go func() {
		ctrlConn, err := ctrlLn.Accept()
		if err != nil {
			return
		}
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

			go func() {
				_ = srv.ServeConn(conn)
			}()
		}
	}()

	time.Sleep(50 * time.Millisecond)

	uc, err := net.DialUnix("unix", nil, &net.UnixAddr{Name: ctrlAddr.Name, Net: "unix"})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = uc.Close() }()

	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	addr := l.Addr().String()

	ch := make(chan net.Conn, 1)
	go func() {
		c, err := l.Accept()
		if err == nil {
			ch <- c
		}
	}()

	client, err := net.Dial("tcp", addr)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = client.Close() }()

	server := <-ch
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

	time.Sleep(100 * time.Millisecond)

	req := "GET /ready HTTP/1.1\r\nHost: localhost\r\n\r\n"
	if _, err := client.Write([]byte(req)); err != nil {
		t.Fatal(err)
	}

	_ = client.SetReadDeadline(time.Now().Add(3 * time.Second))
	buf := make([]byte, 4096)
	n, err := client.Read(buf)
	if err != nil {
		fmt.Printf("Server logs: ServeConn error=%v\n", err)
		t.Fatalf("read: %v", err)
	}

	fmt.Printf("Response: %s\n", string(buf[:n]))
}
