package handler

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

type mockResponseWriter struct {
	code int
	body []byte
}

func (m *mockResponseWriter) WriteHeader(statusCode int) {
	m.code = statusCode
}

func (m *mockResponseWriter) Write(body []byte) (int, error) {
	m.body = body
	return len(body), nil
}

func newMockWriter() ResponseWriter {
	return &mockResponseWriter{}
}

func TestReady(t *testing.T) {
	tests := []struct {
		name     string
		method   string
		wantCode int
		wantBody string
	}{
		{
			name:     "GET returns OK",
			method:   http.MethodGet,
			wantCode: http.StatusOK,
			wantBody: "OK",
		},
		{
			name:     "POST returns OK",
			method:   http.MethodPost,
			wantCode: http.StatusOK,
			wantBody: "OK",
		},
		{
			name:     "PUT returns OK",
			method:   http.MethodPut,
			wantCode: http.StatusOK,
			wantBody: "OK",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			w := newMockWriter()
			r := httptest.NewRequest(tt.method, "/ready", nil)

			_ = Ready(w, r)

			if w.(*mockResponseWriter).code != tt.wantCode {
				t.Errorf("Ready() code = %v, want %v", w.(*mockResponseWriter).code, tt.wantCode)
			}

			if string(w.(*mockResponseWriter).body) != tt.wantBody {
				t.Errorf("Ready() body = %q, want %q", w.(*mockResponseWriter).body, tt.wantBody)
			}
		})
	}
}

func TestResponseWriterAdapter(t *testing.T) {
	t.Run("adapts http.ResponseWriter", func(t *testing.T) {
		w := httptest.NewRecorder()
		adapter := NewResponseWriterAdapter(w)

		adapter.WriteHeader(http.StatusOK)
		_, _ = adapter.Write([]byte("test"))

		if w.Code != http.StatusOK {
			t.Errorf("WriteHeader() status = %v, want %v", w.Code, http.StatusOK)
		}

		if w.Body.String() != "test" {
			t.Errorf("Write() body = %q, want %q", w.Body.String(), "test")
		}
	})
}

func TestAdapt(t *testing.T) {
	t.Run("wraps HandlerFunc to http.HandlerFunc", func(t *testing.T) {
		w := httptest.NewRecorder()
		r := httptest.NewRequest(http.MethodGet, "/test", nil)

		handler := Adapt(func(w ResponseWriter, r *http.Request) error {
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte("adapted"))
			return nil
		})

		handler.ServeHTTP(w, r)

		if w.Code != http.StatusOK {
			t.Errorf("ServeHTTP() status = %v, want %v", w.Code, http.StatusOK)
		}

		if w.Body.String() != "adapted" {
			t.Errorf("ServeHTTP() body = %q, want %q", w.Body.String(), "adapted")
		}
	})

	t.Run("returns error on handler failure", func(t *testing.T) {
		w := httptest.NewRecorder()
		r := httptest.NewRequest(http.MethodGet, "/test", nil)

		handler := Adapt(func(w ResponseWriter, r *http.Request) error {
			return http.ErrNoCookie
		})

		handler.ServeHTTP(w, r)

		if w.Code != http.StatusInternalServerError {
			t.Errorf("ServeHTTP() status on error = %v, want %v", w.Code, http.StatusInternalServerError)
		}
	})
}

func TestRouter(t *testing.T) {
	t.Run("routes to HandlerFunc", func(t *testing.T) {
		router := NewRouter()
		router.Handle("/test", func(w ResponseWriter, r *http.Request) error {
			w.WriteHeader(http.StatusCreated)
			_, _ = w.Write([]byte("created"))
			return nil
		})

		w := httptest.NewRecorder()
		r := httptest.NewRequest(http.MethodGet, "/test", nil)

		router.ServeHTTP(w, r)

		if w.Code != http.StatusCreated {
			t.Errorf("Router status = %v, want %v", w.Code, http.StatusCreated)
		}

		if w.Body.String() != "created" {
			t.Errorf("Router body = %q, want %q", w.Body.String(), "created")
		}
	})

	t.Run("404 for unknown path", func(t *testing.T) {
		router := NewRouter()

		w := httptest.NewRecorder()
		r := httptest.NewRequest(http.MethodGet, "/unknown", nil)

		router.ServeHTTP(w, r)

		if w.Code != http.StatusNotFound {
			t.Errorf("Router unknown path status = %v, want %v", w.Code, http.StatusNotFound)
		}
	})
}
