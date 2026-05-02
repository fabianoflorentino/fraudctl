package logreg

import (
	"errors"
	"fmt"
)

var (
	errInvalidMagic   = errors.New("invalid logreg model magic")
	errInvalidVersion = errors.New("unsupported logreg model version")
)

var printf = fmt.Printf
