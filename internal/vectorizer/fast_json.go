package vectorizer

import (
	"bytes"

	"github.com/fabianoflorentino/fraudctl/internal/model"
)

var (
	jsonFieldTx           = []byte(`"transaction"`)
	jsonFieldCust         = []byte(`"customer"`)
	jsonFieldMerch        = []byte(`"merchant"`)
	jsonFieldTerm         = []byte(`"terminal"`)
	jsonFieldLastTx       = []byte(`"last_transaction"`)
	jsonFieldAmount       = []byte(`"amount"`)
	jsonFieldInstallments = []byte(`"installments"`)
	jsonFieldRequestedAt  = []byte(`"requested_at"`)
	jsonFieldAvgAmount    = []byte(`"avg_amount"`)
	jsonFieldTxCount24h   = []byte(`"tx_count_24h"`)
	jsonFieldKnownMerch   = []byte(`"known_merchants"`)
	jsonFieldID           = []byte(`"id"`)
	jsonFieldMCC          = []byte(`"mcc"`)
	jsonFieldIsOnline     = []byte(`"is_online"`)
	jsonFieldCardPresent  = []byte(`"card_present"`)
	jsonFieldKmFromHome   = []byte(`"km_from_home"`)
	jsonFieldTimestamp    = []byte(`"timestamp"`)
	jsonFieldKmFromCurr   = []byte(`"km_from_current"`)
)

func findJSONObject(data []byte, field []byte) []byte {
	idx := bytes.Index(data, field)
	if idx == -1 {
		return nil
	}
	rest := data[idx+len(field):]

	colonIdx := bytes.IndexByte(rest, ':')
	if colonIdx == -1 {
		return nil
	}
	rest = rest[colonIdx+1:]

	start := -1
	for i, b := range rest {
		if b == '{' {
			start = i
			break
		}
		if b != ' ' && b != '\t' && b != '\n' && b != '\r' {
			return nil
		}
	}
	if start == -1 {
		return nil
	}

	depth := 1
	for i := start + 1; i < len(rest); i++ {
		switch rest[i] {
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				return rest[start : i+1]
			}
		}
	}
	return nil
}

func parseJSONFloat64Fast(obj []byte, field []byte) float64 {
	idx := bytes.Index(obj, field)
	if idx == -1 {
		return 0
	}
	rest := obj[idx+len(field):]

	colonIdx := bytes.IndexByte(rest, ':')
	if colonIdx == -1 {
		return 0
	}
	rest = rest[colonIdx+1:]

	start := -1
	for i, b := range rest {
		if (b >= '0' && b <= '9') || b == '.' || b == '-' || b == '+' {
			start = i
			break
		}
		if b != ' ' && b != '\t' && b != '\n' && b != '\r' {
			return 0
		}
	}
	if start == -1 {
		return 0
	}

	end := start
	for i := start; i < len(rest); i++ {
		b := rest[i]
		if (b >= '0' && b <= '9') || b == '.' || b == 'e' || b == 'E' || b == '-' || b == '+' {
			end = i
		} else {
			break
		}
	}

	return parseFloatFast(rest[start : end+1])
}

func parseFloatFast(b []byte) float64 {
	if len(b) == 0 {
		return 0
	}
	neg := false
	switch b[0] {
	case '-':
		neg = true
		fallthrough
	case '+':
		b = b[1:]
	}

	var intPart int64
	var fracPart int64
	var fracDiv int64 = 1
	dotIdx := -1
	for i := 0; i < len(b); i++ {
		c := b[i]
		if c == '.' {
			dotIdx = i
			break
		}
		if c < '0' || c > '9' {
			break
		}
		intPart = intPart*10 + int64(c-'0')
	}

	if dotIdx >= 0 {
		for i := dotIdx + 1; i < len(b); i++ {
			c := b[i]
			if c < '0' || c > '9' {
				break
			}
			fracPart = fracPart*10 + int64(c-'0')
			fracDiv *= 10
		}
		if intPart == 0 && fracPart == 0 {
			return 0
		}
		val := float64(intPart) + float64(fracPart)/float64(fracDiv)
		if neg {
			val = -val
		}
		return val
	}

	val := float64(intPart)
	if neg {
		val = -val
	}
	return val
}

func parseJSONIntFast(obj []byte, field []byte) int {
	idx := bytes.Index(obj, field)
	if idx == -1 {
		return 0
	}
	rest := obj[idx+len(field):]

	colonIdx := bytes.IndexByte(rest, ':')
	if colonIdx == -1 {
		return 0
	}
	rest = rest[colonIdx+1:]

	start := -1
	for i, b := range rest {
		if (b >= '0' && b <= '9') || b == '-' {
			start = i
			break
		}
		if b != ' ' && b != '\t' && b != '\n' && b != '\r' {
			return 0
		}
	}
	if start == -1 {
		return 0
	}

	end := start
	for i := start; i < len(rest); i++ {
		b := rest[i]
		if b >= '0' && b <= '9' {
			end = i
		} else {
			break
		}
	}

	return int(parseFloatFast(rest[start : end+1]))
}

func parseJSONStringFast(obj []byte, field []byte) string {
	idx := bytes.Index(obj, field)
	if idx == -1 {
		return ""
	}
	rest := obj[idx+len(field):]

	colonIdx := bytes.IndexByte(rest, ':')
	if colonIdx == -1 {
		return ""
	}
	rest = rest[colonIdx+1:]

	quoteStart := -1
	for i, b := range rest {
		if b == '"' {
			quoteStart = i
			break
		}
		if b != ' ' && b != '\t' && b != '\n' && b != '\r' {
			return ""
		}
	}
	if quoteStart == -1 {
		return ""
	}

	for i := quoteStart + 1; i < len(rest); i++ {
		if rest[i] == '"' && rest[i-1] != '\\' {
			return string(rest[quoteStart+1 : i])
		}
	}
	return ""
}

func parseJSONBoolFast(obj []byte, field []byte) bool {
	idx := bytes.Index(obj, field)
	if idx == -1 {
		return false
	}
	rest := obj[idx+len(field):]

	colonIdx := bytes.IndexByte(rest, ':')
	if colonIdx == -1 {
		return false
	}
	rest = rest[colonIdx+1:]

	for i := 0; i < len(rest); i++ {
		b := rest[i]
		if b == ' ' || b == '\t' || b == '\n' || b == '\r' {
			continue
		}
		if b == 't' && i+3 < len(rest) && rest[i+1] == 'r' && rest[i+2] == 'u' && rest[i+3] == 'e' {
			return true
		}
		return false
	}
	return false
}

func parseJSONStringArrayFast(obj []byte, field []byte) []string {
	idx := bytes.Index(obj, field)
	if idx == -1 {
		return nil
	}
	rest := obj[idx+len(field):]

	colonIdx := bytes.IndexByte(rest, ':')
	if colonIdx == -1 {
		return nil
	}
	rest = rest[colonIdx+1:]

	bracketStart := -1
	for i, b := range rest {
		if b == '[' {
			bracketStart = i
			break
		}
		if b != ' ' && b != '\t' && b != '\n' && b != '\r' {
			return nil
		}
	}
	if bracketStart == -1 {
		return nil
	}

	depth := 1
	bracketEnd := -1
	inString := false
	for i := bracketStart + 1; i < len(rest); i++ {
		if rest[i] == '"' && (i == 0 || rest[i-1] != '\\') {
			inString = !inString
			continue
		}
		if inString {
			continue
		}
		if rest[i] == '[' {
			depth++
		} else if rest[i] == ']' {
			depth--
			if depth == 0 {
				bracketEnd = i
				break
			}
		}
	}
	if bracketEnd == -1 {
		return nil
	}

	arrayContent := rest[bracketStart+1 : bracketEnd]
	var result []string

	inStr := false
	strStart := -1
	for i := 0; i < len(arrayContent); i++ {
		b := arrayContent[i]
		if b == '"' && (i == 0 || arrayContent[i-1] != '\\') {
			if !inStr {
				inStr = true
				strStart = i
			} else {
				inStr = false
				result = append(result, string(arrayContent[strStart+1:i]))
			}
		}
	}

	return result
}

func buildKnownMerchMap(arr []string) map[string]struct{} {
	if arr == nil {
		return nil
	}
	m := make(map[string]struct{}, len(arr))
	for _, s := range arr {
		m[s] = struct{}{}
	}
	return m
}

var errInvalidJSON = &invalidJSONError{}

type invalidJSONError struct{}

func (e *invalidJSONError) Error() string { return "invalid json" }

func (v *Vectorizer) VectorizeJSON(data []byte) (model.Vector14, error) {
	var vec model.Vector14

	if len(data) < 2 || data[0] != '{' {
		return vec, errInvalidJSON
	}

	txObj := findJSONObject(data, jsonFieldTx)
	if txObj == nil {
		return vec, nil
	}

	txAmount := parseJSONFloat64Fast(txObj, jsonFieldAmount)
	txInstallments := parseJSONIntFast(txObj, jsonFieldInstallments)
	txRequestedAt := parseJSONStringFast(txObj, jsonFieldRequestedAt)

	vec[0] = clampFloat32(float32(txAmount) * v.invMaxAmount)
	vec[1] = clampFloat32(float32(txInstallments) * v.invMaxInstall)

	custObj := findJSONObject(data, jsonFieldCust)
	if custObj != nil {
		custAvgAmount := parseJSONFloat64Fast(custObj, jsonFieldAvgAmount)
		custTxCount24h := parseJSONIntFast(custObj, jsonFieldTxCount24h)
		custKnownArr := parseJSONStringArrayFast(custObj, jsonFieldKnownMerch)
		custKnownMerch := buildKnownMerchMap(custKnownArr)

		var amountVsAvg float64
		if custAvgAmount > 0 {
			amountVsAvg = txAmount / custAvgAmount
		}
		vec[2] = clampFloat32(float32(amountVsAvg) * v.invAmountRatio)
		vec[8] = clampFloat32(float32(custTxCount24h) * v.invMaxTxCount)

		merchObj := findJSONObject(data, jsonFieldMerch)
		if merchObj != nil {
			merchID := parseJSONStringFast(merchObj, jsonFieldID)
			merchMCC := parseJSONStringFast(merchObj, jsonFieldMCC)
			merchAvg := parseJSONFloat64Fast(merchObj, jsonFieldAvgAmount)

			isKnown := false
			if custKnownMerch != nil && merchID != "" {
				_, isKnown = custKnownMerch[merchID]
			}
			if merchID != "" && !isKnown {
				vec[11] = 1
			}

			vec[12] = float32(v.mccRisk.Get(merchMCC))
			vec[13] = clampFloat32(float32(merchAvg) * v.invMaxMerchantAvg)
		}
	}

	hour, dayOfWeek := parseHourAndWeekday(txRequestedAt)
	vec[3] = float32(hour) / 23.0
	vec[4] = float32(dayOfWeek) / 6.0

	lastTxObj := findJSONObject(data, jsonFieldLastTx)
	if lastTxObj != nil {
		lastTxTime := parseJSONStringFast(lastTxObj, jsonFieldTimestamp)
		lastTxKm := parseJSONFloat64Fast(lastTxObj, jsonFieldKmFromCurr)

		reqSec := parseUnixSeconds(txRequestedAt)
		lastSec := parseUnixSeconds(lastTxTime)
		minutes := float32(reqSec-lastSec) / 60.0
		vec[5] = clampFloat32(minutes * v.invMaxMinutes)
		vec[6] = clampFloat32(float32(lastTxKm) * v.invMaxKm)
	} else {
		vec[5] = -1
		vec[6] = -1
	}

	termObj := findJSONObject(data, jsonFieldTerm)
	if termObj != nil {
		termIsOnline := parseJSONBoolFast(termObj, jsonFieldIsOnline)
		termCardPresent := parseJSONBoolFast(termObj, jsonFieldCardPresent)
		termKmFromHome := parseJSONFloat64Fast(termObj, jsonFieldKmFromHome)

		vec[7] = clampFloat32(float32(termKmFromHome) * v.invMaxKm)

		if termIsOnline {
			vec[9] = 1
		}
		if termCardPresent {
			vec[10] = 1
		}
	}

	return vec, nil
}
