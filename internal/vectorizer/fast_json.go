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

func parseJSONStringBytes(obj []byte, field []byte) []byte {
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

	quoteStart := -1
	for i, b := range rest {
		if b == '"' {
			quoteStart = i
			break
		}
		if b != ' ' && b != '\t' && b != '\n' && b != '\r' {
			return nil
		}
	}
	if quoteStart == -1 {
		return nil
	}

	for i := quoteStart + 1; i < len(rest); i++ {
		if rest[i] == '"' && rest[i-1] != '\\' {
			return rest[quoteStart+1 : i]
		}
	}
	return nil
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

func merchantIDInArray(obj []byte, field []byte, target []byte) bool {
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

	bracketStart := -1
	for i, b := range rest {
		if b == '[' {
			bracketStart = i
			break
		}
		if b != ' ' && b != '\t' && b != '\n' && b != '\r' {
			return false
		}
	}
	if bracketStart == -1 {
		return false
	}

	inStr := false
	var strStart int
	for i := bracketStart + 1; i < len(rest); i++ {
		b := rest[i]
		if b == '"' && (i == 0 || rest[i-1] != '\\') {
			if !inStr {
				inStr = true
				strStart = i + 1
			} else {
				if bytes.Equal(rest[strStart:i], target) {
					return true
				}
				inStr = false
			}
		} else if b == ']' && !inStr {
			return false
		}
	}
	return false
}

var errInvalidJSON = &invalidJSONError{}

type invalidJSONError struct{}

func (e *invalidJSONError) Error() string { return "invalid json" }

func parseHourAndWeekdayBytes(b []byte) (hour, weekday int) {
	if len(b) < 13 || b[10] != 'T' {
		return 0, 0
	}
	h := int(b[11]-'0')*10 + int(b[12]-'0')
	if len(b) < 10 {
		return h, 0
	}
	year := int(b[0]-'0')*1000 + int(b[1]-'0')*100 + int(b[2]-'0')*10 + int(b[3]-'0')
	month := int(b[5]-'0')*10 + int(b[6]-'0')
	day := int(b[8]-'0')*10 + int(b[9]-'0')
	t := [12]int{0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4}
	if month < 3 {
		year--
	}
	wd := (year + year/4 - year/100 + year/400 + t[month-1] + day) % 7
	return h, (wd + 6) % 7
}

func parseUnixSecondsBytes(b []byte) int64 {
	if len(b) < 19 || b[10] != 'T' {
		return 0
	}
	year := int64(b[0]-'0')*1000 + int64(b[1]-'0')*100 + int64(b[2]-'0')*10 + int64(b[3]-'0')
	month := int64(b[5]-'0')*10 + int64(b[6]-'0')
	day := int64(b[8]-'0')*10 + int64(b[9]-'0')
	hour := int64(b[11]-'0')*10 + int64(b[12]-'0')
	min := int64(b[14]-'0')*10 + int64(b[15]-'0')
	sec := int64(b[17]-'0')*10 + int64(b[18]-'0')

	y, m := year, month
	if m <= 2 {
		y--
		m += 12
	}
	days := y*365 + y/4 - y/100 + y/400 + (153*(m-3)+2)/5 + day - 719469
	return days*86400 + hour*3600 + min*60 + sec
}

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
	txRequestedAt := parseJSONStringBytes(txObj, jsonFieldRequestedAt)

	vec[0] = clampFloat32(float32(txAmount) * v.invMaxAmount)
	vec[1] = clampFloat32(float32(txInstallments) * v.invMaxInstall)

	custObj := findJSONObject(data, jsonFieldCust)
	if custObj != nil {
		custAvgAmount := parseJSONFloat64Fast(custObj, jsonFieldAvgAmount)
		custTxCount24h := parseJSONIntFast(custObj, jsonFieldTxCount24h)

		var amountVsAvg float64
		if custAvgAmount > 0 {
			amountVsAvg = txAmount / custAvgAmount
		}
		vec[2] = clampFloat32(float32(amountVsAvg) * v.invAmountRatio)
		vec[8] = clampFloat32(float32(custTxCount24h) * v.invMaxTxCount)

		merchObj := findJSONObject(data, jsonFieldMerch)
		if merchObj != nil {
			merchID := parseJSONStringBytes(merchObj, jsonFieldID)
			merchMCC := parseJSONStringBytes(merchObj, jsonFieldMCC)
			merchAvg := parseJSONFloat64Fast(merchObj, jsonFieldAvgAmount)

			if len(merchID) > 0 && !merchantIDInArray(custObj, jsonFieldKnownMerch, merchID) {
				vec[11] = 1
			}

			vec[12] = float32(v.mccRisk.Get(merchMCC))
			vec[13] = clampFloat32(float32(merchAvg) * v.invMaxMerchantAvg)
		}
	}

	hour, dayOfWeek := parseHourAndWeekdayBytes(txRequestedAt)
	vec[3] = float32(hour) / 23.0
	vec[4] = float32(dayOfWeek) / 6.0

	lastTxObj := findJSONObject(data, jsonFieldLastTx)
	if lastTxObj != nil {
		lastTxTime := parseJSONStringBytes(lastTxObj, jsonFieldTimestamp)
		lastTxKm := parseJSONFloat64Fast(lastTxObj, jsonFieldKmFromCurr)

		reqSec := parseUnixSecondsBytes(txRequestedAt)
		lastSec := parseUnixSecondsBytes(lastTxTime)
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
