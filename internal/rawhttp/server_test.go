package rawhttp

import (
	"bytes"
	"io"
	"net"
	"sync"
	"testing"
)

type mockHandler struct{}

func (m *mockHandler) ServeFraudScore(body []byte) []byte {
	if !bytes.Contains(body, []byte(`"amount"`)) {
		return fraudResponses[0]
	}
	if bytes.Contains(body, []byte(`"fraud"`)) {
		return fraudResponses[5]
	}
	return fraudResponses[2]
}

func (m *mockHandler) ServeReady() []byte {
	return readyResponse
}

func TestServeConn_POSTFraudScore(t *testing.T) {
	srv := New(&mockHandler{})
	server, client := net.Pipe()
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		_ = srv.ServeConn(server)
	}()
	body := `{"id":"tx-1","transaction":{"amount":100,"installments":1,"requested_at":"2026-03-11T10:00:00Z"},"customer":{"avg_amount":100,"tx_count_24h":5,"known_merchants":["m1"]},"merchant":{"id":"m1","mcc":"5411","avg_amount":50},"terminal":{"is_online":false,"card_present":true,"km_from_home":10}}`
	req := "POST /fraud-score HTTP/1.1\r\nContent-Type: application/json\r\nContent-Length: " + itoa(len(body)) + "\r\n\r\n" + body
	if _, err := client.Write([]byte(req)); err != nil {
		t.Fatal(err)
	}
	resp, err := readResponse(client)
	if err != nil {
		t.Fatal(err)
	}
	if resp.status != 200 {
		t.Errorf("expected 200, got %d", resp.status)
	}
	if !bytes.Contains(resp.body, []byte(`"approved":true`)) {
		t.Errorf("expected approved=true, got %s", resp.body)
	}
	_ = client.Close()
	wg.Wait()
}

func TestServeConn_GETReady(t *testing.T) {
	srv := New(&mockHandler{})
	server, client := net.Pipe()
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		_ = srv.ServeConn(server)
	}()
	req := "GET /ready HTTP/1.1\r\n\r\n"
	if _, err := client.Write([]byte(req)); err != nil {
		t.Fatal(err)
	}
	resp, err := readResponse(client)
	if err != nil {
		t.Fatal(err)
	}
	if resp.status != 200 {
		t.Errorf("expected 200, got %d", resp.status)
	}
	if string(resp.body) != "OK" {
		t.Errorf("expected body OK, got %q", resp.body)
	}
	_ = client.Close()
	wg.Wait()
}

func TestServeConn_NotFound(t *testing.T) {
	srv := New(&mockHandler{})
	server, client := net.Pipe()
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		_ = srv.ServeConn(server)
	}()
	req := "GET /unknown HTTP/1.1\r\n\r\n"
	if _, err := client.Write([]byte(req)); err != nil {
		t.Fatal(err)
	}
	resp, err := readResponse(client)
	if err != nil {
		t.Fatal(err)
	}
	if resp.status != 404 {
		t.Errorf("expected 404, got %d", resp.status)
	}
	_ = client.Close()
	wg.Wait()
}

func TestServeConn_WrongMethod(t *testing.T) {
	srv := New(&mockHandler{})
	server, client := net.Pipe()
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		_ = srv.ServeConn(server)
	}()
	body := `{"id":"tx-1","transaction":{"amount":100,"installments":1,"requested_at":"2026-03-11T10:00:00Z"},"customer":{"avg_amount":100,"tx_count_24h":5,"known_merchants":["m1"]},"merchant":{"id":"m1","mcc":"5411","avg_amount":50},"terminal":{"is_online":false,"card_present":true,"km_from_home":10}}`
	req := "GET /fraud-score HTTP/1.1\r\nContent-Length: " + itoa(len(body)) + "\r\n\r\n" + body
	if _, err := client.Write([]byte(req)); err != nil {
		t.Fatal(err)
	}
	resp, err := readResponse(client)
	if err != nil {
		t.Fatal(err)
	}
	if resp.status != 404 {
		t.Errorf("expected 404, got %d", resp.status)
	}
	_ = client.Close()
	wg.Wait()
}

func TestServeConn_KeepAlive(t *testing.T) {
	srv := New(&mockHandler{})
	server, client := net.Pipe()
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		_ = srv.ServeConn(server)
	}()
	body1 := `{"amount":100,"installments":1,"requested_at":"2026-03-11T10:00:00Z","customer":{"avg_amount":100,"tx_count_24h":5,"known_merchants":["m1"]},"merchant":{"id":"m1","mcc":"5411","avg_amount":50},"terminal":{"is_online":false,"card_present":true,"km_from_home":10},"id":"tx-1"}`
	req1 := "POST /fraud-score HTTP/1.1\r\nContent-Length: " + itoa(len(body1)) + "\r\n\r\n" + body1
	if _, err := client.Write([]byte(req1)); err != nil {
		t.Fatal(err)
	}
	resp1, err := readResponse(client)
	if err != nil {
		t.Fatal(err)
	}
	if resp1.status != 200 {
		t.Errorf("request 1: expected 200, got %d", resp1.status)
	}
	body2 := `{"amount":200,"installments":3,"requested_at":"2026-03-11T11:00:00Z","customer":{"avg_amount":50,"tx_count_24h":2,"known_merchants":["m2"]},"merchant":{"id":"m2","mcc":"5999","avg_amount":30},"terminal":{"is_online":true,"card_present":false,"km_from_home":5},"id":"tx-2"}`
	req2 := "POST /fraud-score HTTP/1.1\r\nContent-Length: " + itoa(len(body2)) + "\r\n\r\n" + body2
	if _, err := client.Write([]byte(req2)); err != nil {
		t.Fatal(err)
	}
	resp2, err := readResponse(client)
	if err != nil {
		t.Fatal(err)
	}
	if resp2.status != 200 {
		t.Errorf("request 2: expected 200, got %d", resp2.status)
	}
	_ = client.Close()
	wg.Wait()
}

func TestServeConn_InvalidJSON(t *testing.T) {
	srv := New(&mockHandler{})
	server, client := net.Pipe()
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		_ = srv.ServeConn(server)
	}()
	body := `not json at all`
	req := "POST /fraud-score HTTP/1.1\r\nContent-Length: " + itoa(len(body)) + "\r\n\r\n" + body
	if _, err := client.Write([]byte(req)); err != nil {
		t.Fatal(err)
	}
	resp, err := readResponse(client)
	if err != nil {
		t.Fatal(err)
	}
	if resp.status != 200 {
		t.Errorf("expected 200, got %d", resp.status)
	}
	if !bytes.Contains(resp.body, []byte(`"approved":true`)) {
		t.Errorf("expected fallback approved=true, got %s", resp.body)
	}
	_ = client.Close()
	wg.Wait()
}

func TestIndexHeaderEnd(t *testing.T) {
	tests := []struct {
		input []byte
		want  int
	}{
		{[]byte("POST / HTTP/1.1\r\n\r\n"), 15},
		{[]byte("GET / HTTP/1.1\r\nHost: x\r\n\r\n"), 23},
		{[]byte("GET / HTTP/1.1\r\n\r\nbody"), 14},
		{[]byte("no header end"), -1},
		{[]byte(""), -1},
		{[]byte("\r\n\r\n"), 0},
		{[]byte("GET / HTTP/1.1\r\nContent-Length: 5\r\n\r\nhello"), 33},
	}
	for _, tt := range tests {
		got := indexHeaderEnd(tt.input)
		if got != tt.want {
			t.Errorf("indexHeaderEnd(%q) = %d; want %d", tt.input, got, tt.want)
		}
	}
}

func TestParseRequestLine(t *testing.T) {
	tests := []struct {
		input       []byte
		wantMethod  string
		wantPath    string
		wantCLenMin int
	}{
		{[]byte("POST /fraud-score HTTP/1.1\r\nContent-Length: 123\r\n\r\n"), "POST", "/fraud-score", 123},
		{[]byte("GET /ready HTTP/1.1\r\n\r\n"), "GET", "/ready", 0},
		{[]byte("PUT /other HTTP/1.1\r\n\r\n"), "PUT", "/other", 0},
	}
	for _, tt := range tests {
		method, path, clen := parseRequestLine(tt.input)
		if string(method) != tt.wantMethod {
			t.Errorf("parseRequestLine(%q) method = %q; want %q", tt.input, method, tt.wantMethod)
		}
		if string(path) != tt.wantPath {
			t.Errorf("parseRequestLine(%q) path = %q; want %q", tt.input, path, tt.wantPath)
		}
		if clen != tt.wantCLenMin {
			t.Errorf("parseRequestLine(%q) contentLen = %d; want %d", tt.input, clen, tt.wantCLenMin)
		}
	}
}

func TestFindContentLength(t *testing.T) {
	tests := []struct {
		input []byte
		want  int
	}{
		{[]byte("Content-Length: 42\r\n"), 42},
		{[]byte("content-length: 100\r\n"), 100},
		{[]byte("CONTENT-LENGTH: 5\r\n"), 5},
		{[]byte("Content-Length: 0\r\n"), 0},
		{[]byte("No length here"), 0},
		{[]byte(""), 0},
	}
	for _, tt := range tests {
		got := findContentLength(tt.input)
		if got != tt.want {
			t.Errorf("findContentLength(%q) = %d; want %d", tt.input, got, tt.want)
		}
	}
}

func TestPrecomputedResponses(t *testing.T) {
	if len(fraudResponses) != 6 {
		t.Errorf("expected 6 fraud responses, got %d", len(fraudResponses))
	}
	for i := 0; i < 6; i++ {
		if len(fraudResponses[i]) == 0 {
			t.Errorf("fraudResponses[%d] is empty", i)
		}
		if !bytes.Contains(fraudResponses[i], []byte("200 OK")) {
			t.Errorf("fraudResponses[%d] missing status line", i)
		}
	}
	if len(readyResponse) == 0 {
		t.Error("readyResponse is empty")
	}
	if !bytes.Contains(readyResponse, []byte("OK")) {
		t.Error("readyResponse missing OK body")
	}
	if len(notFoundResponse) == 0 {
		t.Error("notFoundResponse is empty")
	}
	if !bytes.Contains(notFoundResponse, []byte("404")) {
		t.Error("notFoundResponse missing 404 status")
	}
}

type response struct {
	status int
	body   []byte
}

func readResponse(r io.Reader) (*response, error) {
	buf := make([]byte, 4096)
	n, err := r.Read(buf)
	if err != nil {
		return nil, err
	}
	data := buf[:n]
	headEnd := indexHeaderEnd(data)
	if headEnd < 0 {
		return nil, &requestError{"no header end in response"}
	}
	body := data[headEnd+4:]
	status := 0
	if len(data) > 9 && data[9] == '2' && data[10] == '0' && data[11] == '0' {
		status = 200
	} else if len(data) > 9 && data[9] == '4' && data[10] == '0' && data[11] == '4' {
		status = 404
	}
	return &response{status: status, body: body}, nil
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var buf [16]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[i:])
}
