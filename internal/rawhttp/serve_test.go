//nolint:errcheck
package rawhttp

import (
	"fmt"
	"net"
	"testing"
	"time"
)

func TestDirectServe(t *testing.T) {
	srv := New(&mockHandler{})

	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	addr := l.Addr().String()

	done := make(chan error, 1)
	go func() {
		conn, err := l.Accept()
		if err != nil {
			done <- err
			return
		}
		done <- srv.ServeConn(conn)
	}()

	time.Sleep(50 * time.Millisecond)

	client, err := net.DialTimeout("tcp", addr, 5*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()

	req := "GET /ready HTTP/1.1\r\nHost: localhost\r\n\r\n"
	if _, err := client.Write([]byte(req)); err != nil {
		t.Fatal(err)
	}

	client.SetReadDeadline(time.Now().Add(3 * time.Second))
	buf := make([]byte, 4096)
	n, err := client.Read(buf)
	if err != nil {
		t.Fatal(err)
	}

	fmt.Printf("Response: %s\n", string(buf[:n]))

	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	default:
	}
}
