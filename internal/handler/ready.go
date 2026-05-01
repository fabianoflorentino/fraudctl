package handler

import (
	"github.com/valyala/fasthttp"
)

func Ready(ctx *fasthttp.RequestCtx) {
	ctx.SetStatusCode(fasthttp.StatusOK)
	ctx.WriteString("OK")
}
