package rawhttp

import (
	"io"
	"net"
	"strconv"
	"sync"
	"time"
)

type Handler interface {
	ServeFraudScore(body []byte) []byte
	ServeReady() []byte
}

var (
	fraudResponses   [6][]byte
	readyResponse    []byte
	notFoundResponse []byte
)

func init() {
	bodies := [6]string{
		`{"approved":true,"fraud_score":0.0}`,
		`{"approved":true,"fraud_score":0.2}`,
		`{"approved":true,"fraud_score":0.4}`,
		`{"approved":false,"fraud_score":0.6}`,
		`{"approved":false,"fraud_score":0.8}`,
		`{"approved":false,"fraud_score":1.0}`,
	}

	for i, body := range bodies {
		fraudResponses[i] = buildOKResponse("application/json", body)
	}

	readyResponse = buildOKResponse("", "OK")
	notFoundResponse = buildResponse(404, "Not Found", "", "")
}

func buildOKResponse(ct, body string) []byte {
	hdr := "HTTP/1.1 200 OK\r\n"

	if ct != "" {
		hdr += "Content-Type: " + ct + "\r\n"
	}

	hdr += "Content-Length: " + strconv.Itoa(len(body)) + "\r\n\r\n"
	out := make([]byte, len(hdr)+len(body))

	copy(out, hdr)
	copy(out[len(hdr):], body)

	return out
}

func buildResponse(code int, text, ct, body string) []byte {
	hdr := "HTTP/1.1 " + strconv.Itoa(code) + " " + text + "\r\n"

	if ct != "" {
		hdr += "Content-Type: " + ct + "\r\n"
	}

	hdr += "Content-Length: " + strconv.Itoa(len(body)) + "\r\n\r\n"
	out := make([]byte, len(hdr)+len(body))

	copy(out, hdr)
	copy(out[len(hdr):], body)

	return out
}

var bufPool = sync.Pool{
	New: func() any {
		b := make([]byte, readBufSize)
		return &b
	},
}

const (
	readBufSize    = 1024
	maxRequestSize = 2 * 1024
)

type Server struct {
	handler Handler
}

func New(handler Handler) *Server {
	return &Server{handler: handler}
}

func (s *Server) ServeConn(conn net.Conn) error {
	defer func() { _ = conn.Close() }()

	bp := bufPool.Get().(*[]byte)
	buf := *bp
	pos := 0
	used := 0
	defer bufPool.Put(bp)

	for {
		if err := conn.SetReadDeadline(time.Now().Add(10 * time.Second)); err != nil {
			return err
		}

		headEnd := indexHeaderEnd(buf[pos:used])
		for headEnd < 0 {
			if n, err := conn.Read(buf[used:]); n > 0 {
				used += n
				if used-pos > maxRequestSize {
					return errRequestTooLarge
				}
				headEnd = indexHeaderEnd(buf[pos:used])
				continue
			} else if err != nil {
				if err == io.EOF && headEnd >= 0 {
					break
				}
				return err
			}
		}

		headEnd += pos + 4

		method, path, contentLen := parseRequestLine(buf[pos:headEnd])

		bodyEnd := headEnd + contentLen
		for used < bodyEnd {
			if n, err := conn.Read(buf[used:]); n > 0 {
				used += n
				continue
			} else if err != nil {
				return err
			}
		}

		var resp []byte

		switch {
		case len(path) == 12 && path[1] == 'f' && len(method) == 4 && method[0] == 'P':
			resp = s.handler.ServeFraudScore(buf[headEnd:bodyEnd])
		case len(path) == 6 && path[1] == 'r' && len(method) == 3 && method[0] == 'G':
			resp = s.handler.ServeReady()
		default:
			resp = notFoundResponse
		}

		if err := conn.SetWriteDeadline(time.Now().Add(750 * time.Millisecond)); err != nil {
			return err
		}
		if _, err := conn.Write(resp); err != nil {
			return err
		}

		pos = bodyEnd
		if pos >= used {
			pos = 0
			used = 0
		}
	}
}

func indexHeaderEnd(b []byte) int {
	for i := 0; i+3 < len(b); i++ {
		if b[i] == '\r' && b[i+1] == '\n' && b[i+2] == '\r' && b[i+3] == '\n' {
			return i
		}
	}

	return -1
}

func parseRequestLine(buf []byte) (method, path []byte, contentLen int) {
	i := 0
	for i < len(buf) && buf[i] != ' ' {
		i++
	}

	method = buf[:i]
	i++

	pathStart := i

	for i < len(buf) && buf[i] != ' ' {
		i++
	}

	path = buf[pathStart:i]
	contentLen = findContentLength(buf)

	return
}

func findContentLength(buf []byte) int {
	for i := 0; i+16 < len(buf); i++ {
		if (buf[i] == 'C' || buf[i] == 'c') && isContentLength(buf[i:]) {
			j := i + 16
			for j < len(buf) && buf[j] == ' ' {
				j++
			}
			n := 0
			for j < len(buf) && buf[j] >= '0' && buf[j] <= '9' {
				n = n*10 + int(buf[j]-'0')
				j++
			}
			return n
		}
	}
	return 0
}

func isContentLength(b []byte) bool {
	const prefix = "content-length:"

	if len(b) < len(prefix) {
		return false
	}

	for i := range len(prefix) {
		c := b[i]
		if c >= 'A' && c <= 'Z' {
			c += 'a' - 'A'
		}
		if c != prefix[i] {
			return false
		}
	}

	return true
}

type requestError struct {
	msg string
}

func (e *requestError) Error() string { return e.msg }

var errRequestTooLarge = &requestError{"request too large"}

func FraudResponse(count int) []byte {
	if count < 0 || count >= len(fraudResponses) {
		return fraudResponses[0]
	}
	return fraudResponses[count]
}

func ReadyResponse() []byte {
	return readyResponse
}

func NotFoundResponse() []byte {
	return notFoundResponse
}
