package vectorizer

import (
	"bytes"

	"github.com/fabianoflorentino/fraudctl/internal/model"
)

// parsedTx holds all transaction fields extracted in a single forward scan.
type parsedTx struct {
	amount       float64
	installments int
	requestedAt  []byte
}

type parsedCust struct {
	avgAmount      float64
	txCount24h     int
	knownMerchants []byte // raw JSON array bytes
}

type parsedMerch struct {
	id        []byte
	mcc       []byte
	avgAmount float64
}

type parsedTerm struct {
	isOnline    bool
	cardPresent bool
	kmFromHome  float64
}

type parsedLastTx struct {
	timestamp     []byte
	kmFromCurrent float64
}

// parseTxFields extracts all transaction fields in a single forward scan.
func parseTxFields(tx []byte) parsedTx {
	var f parsedTx
	i := 0
	n := len(tx)
	for i < n && tx[i] != '{' {
		i++
	}
	if i >= n {
		return f
	}
	i++
	for i < n {
		for i < n && (tx[i] == ' ' || tx[i] == '\t' || tx[i] == '\n' || tx[i] == '\r' || tx[i] == ',') {
			i++
		}
		if i >= n || tx[i] == '}' {
			break
		}
		if tx[i] != '"' {
			i++
			continue
		}
		i++
		keyStart := i
		for i < n && tx[i] != '"' {
			if tx[i] == '\\' {
				i++
			}
			i++
		}
		key := tx[keyStart:i]
		i++
		for i < n && tx[i] != ':' {
			i++
		}
		i++
		for i < n && (tx[i] == ' ' || tx[i] == '\t' || tx[i] == '\n' || tx[i] == '\r') {
			i++
		}
		if i >= n {
			break
		}
		switch {
		case len(key) == 6 && key[0] == 'a':
			f.amount, i = parseFloatValueInline(tx, i, n)
		case len(key) == 12 && key[0] == 'i':
			f.installments, i = parseIntValueInline(tx, i, n)
		case len(key) == 12 && key[0] == 'r':
			f.requestedAt, i = parseStringValueInline(tx, i, n)
		default:
			i = skipValueInline(tx, i, n)
		}
	}
	return f
}

func parseCustFields(cust []byte) parsedCust {
	var f parsedCust
	i := 0
	n := len(cust)
	for i < n && cust[i] != '{' {
		i++
	}
	if i >= n {
		return f
	}
	i++
	for i < n {
		for i < n && (cust[i] == ' ' || cust[i] == '\t' || cust[i] == '\n' || cust[i] == '\r' || cust[i] == ',') {
			i++
		}
		if i >= n || cust[i] == '}' {
			break
		}
		if cust[i] != '"' {
			i++
			continue
		}
		i++
		keyStart := i
		for i < n && cust[i] != '"' {
			if cust[i] == '\\' {
				i++
			}
			i++
		}
		key := cust[keyStart:i]
		i++
		for i < n && cust[i] != ':' {
			i++
		}
		i++
		for i < n && (cust[i] == ' ' || cust[i] == '\t' || cust[i] == '\n' || cust[i] == '\r') {
			i++
		}
		if i >= n {
			break
		}
		switch {
		case len(key) == 10 && key[0] == 'a':
			f.avgAmount, i = parseFloatValueInline(cust, i, n)
		case len(key) == 12 && key[0] == 't':
			f.txCount24h, i = parseIntValueInline(cust, i, n)
		case len(key) == 15 && key[0] == 'k':
			f.knownMerchants, i = parseArrayValueInline(cust, i, n)
		default:
			i = skipValueInline(cust, i, n)
		}
	}
	return f
}

func parseMerchFields(merch []byte) parsedMerch {
	var f parsedMerch
	i := 0
	n := len(merch)
	for i < n && merch[i] != '{' {
		i++
	}
	if i >= n {
		return f
	}
	i++
	for i < n {
		for i < n && (merch[i] == ' ' || merch[i] == '\t' || merch[i] == '\n' || merch[i] == '\r' || merch[i] == ',') {
			i++
		}
		if i >= n || merch[i] == '}' {
			break
		}
		if merch[i] != '"' {
			i++
			continue
		}
		i++
		keyStart := i
		for i < n && merch[i] != '"' {
			if merch[i] == '\\' {
				i++
			}
			i++
		}
		key := merch[keyStart:i]
		i++
		for i < n && merch[i] != ':' {
			i++
		}
		i++
		for i < n && (merch[i] == ' ' || merch[i] == '\t' || merch[i] == '\n' || merch[i] == '\r') {
			i++
		}
		if i >= n {
			break
		}
		switch {
		case len(key) == 2 && key[0] == 'i':
			f.id, i = parseStringValueInline(merch, i, n)
		case len(key) == 3 && key[0] == 'm':
			f.mcc, i = parseStringValueInline(merch, i, n)
		case len(key) == 10 && key[0] == 'a':
			f.avgAmount, i = parseFloatValueInline(merch, i, n)
		default:
			i = skipValueInline(merch, i, n)
		}
	}
	return f
}

func parseTermFields(term []byte) parsedTerm {
	var f parsedTerm
	i := 0
	n := len(term)
	for i < n && term[i] != '{' {
		i++
	}
	if i >= n {
		return f
	}
	i++
	for i < n {
		for i < n && (term[i] == ' ' || term[i] == '\t' || term[i] == '\n' || term[i] == '\r' || term[i] == ',') {
			i++
		}
		if i >= n || term[i] == '}' {
			break
		}
		if term[i] != '"' {
			i++
			continue
		}
		i++
		keyStart := i
		for i < n && term[i] != '"' {
			if term[i] == '\\' {
				i++
			}
			i++
		}
		key := term[keyStart:i]
		i++
		for i < n && term[i] != ':' {
			i++
		}
		i++
		for i < n && (term[i] == ' ' || term[i] == '\t' || term[i] == '\n' || term[i] == '\r') {
			i++
		}
		if i >= n {
			break
		}
		switch {
		case len(key) == 9 && key[0] == 'i':
			f.isOnline, i = parseBoolValueInline(term, i, n)
		case len(key) == 12 && key[0] == 'c':
			f.cardPresent, i = parseBoolValueInline(term, i, n)
		case len(key) == 12 && key[0] == 'k':
			f.kmFromHome, i = parseFloatValueInline(term, i, n)
		default:
			i = skipValueInline(term, i, n)
		}
	}
	return f
}

func parseLastTxFields(ltx []byte) parsedLastTx {
	var f parsedLastTx
	i := 0
	n := len(ltx)
	for i < n && ltx[i] != '{' {
		i++
	}
	if i >= n {
		return f
	}
	i++
	for i < n {
		for i < n && (ltx[i] == ' ' || ltx[i] == '\t' || ltx[i] == '\n' || ltx[i] == '\r' || ltx[i] == ',') {
			i++
		}
		if i >= n || ltx[i] == '}' {
			break
		}
		if ltx[i] != '"' {
			i++
			continue
		}
		i++
		keyStart := i
		for i < n && ltx[i] != '"' {
			if ltx[i] == '\\' {
				i++
			}
			i++
		}
		key := ltx[keyStart:i]
		i++
		for i < n && ltx[i] != ':' {
			i++
		}
		i++
		for i < n && (ltx[i] == ' ' || ltx[i] == '\t' || ltx[i] == '\n' || ltx[i] == '\r') {
			i++
		}
		if i >= n {
			break
		}
		switch {
		case len(key) == 9 && key[0] == 't':
			f.timestamp, i = parseStringValueInline(ltx, i, n)
		case len(key) == 15 && key[0] == 'k':
			f.kmFromCurrent, i = parseFloatValueInline(ltx, i, n)
		default:
			i = skipValueInline(ltx, i, n)
		}
	}
	return f
}

// Inline value parsers — return (value, nextIndex) for single-pass scanning.
func parseFloatValueInline(data []byte, i, n int) (float64, int) {
	start := -1
	for i < n {
		b := data[i]
		if (b >= '0' && b <= '9') || b == '.' || b == '-' || b == '+' {
			start = i
			break
		}
		if b != ' ' && b != '\t' && b != '\n' && b != '\r' {
			return 0, i
		}
		i++
	}
	if start == -1 {
		return 0, i
	}
	end := start
	for i = start; i < n; i++ {
		b := data[i]
		if (b >= '0' && b <= '9') || b == '.' || b == 'e' || b == 'E' || b == '-' || b == '+' {
			end = i
		} else {
			break
		}
	}
	i = end + 1
	return parseFloatFast(data[start : end+1]), i
}

func parseIntValueInline(data []byte, i, n int) (int, int) {
	start := -1
	for i < n {
		b := data[i]
		if (b >= '0' && b <= '9') || b == '-' {
			start = i
			break
		}
		if b != ' ' && b != '\t' && b != '\n' && b != '\r' {
			return 0, i
		}
		i++
	}
	if start == -1 {
		return 0, i
	}
	end := start
	for i = start; i < n; i++ {
		b := data[i]
		if b >= '0' && b <= '9' {
			end = i
		} else {
			break
		}
	}
	i = end + 1
	return int(parseFloatFast(data[start : end+1])), i
}

func parseStringValueInline(data []byte, i, n int) ([]byte, int) {
	for i < n {
		if data[i] == '"' {
			i++
			start := i
			for i < n {
				if data[i] == '"' && data[i-1] != '\\' {
					return data[start:i], i + 1
				}
				i++
			}
			return nil, i
		}
		if data[i] != ' ' && data[i] != '\t' && data[i] != '\n' && data[i] != '\r' {
			return nil, i
		}
		i++
	}
	return nil, i
}

func parseBoolValueInline(data []byte, i, n int) (bool, int) {
	for i < n {
		b := data[i]
		if b == ' ' || b == '\t' || b == '\n' || b == '\r' {
			i++
			continue
		}
		if b == 't' && i+3 < n && data[i+1] == 'r' && data[i+2] == 'u' && data[i+3] == 'e' {
			return true, i + 4
		}
		return false, i
	}
	return false, i
}

func parseArrayValueInline(data []byte, i, n int) ([]byte, int) {
	for i < n {
		if data[i] == '[' {
			start := i
			depth := 1
			i++
			for i < n && depth > 0 {
				switch data[i] {
				case '[':
					depth++
				case ']':
					depth--
				case '"':
					i++
					for i < n {
						if data[i] == '\\' {
							i += 2
							continue
						}
						if data[i] == '"' {
							break
						}
						i++
					}
				}
				i++
			}
			return data[start : i-1], i
		}
		if data[i] != ' ' && data[i] != '\t' && data[i] != '\n' && data[i] != '\r' {
			return nil, i
		}
		i++
	}
	return nil, i
}

func skipValueInline(data []byte, i, n int) int {
	if i >= n {
		return i
	}
	switch data[i] {
	case '"':
		i++
		for i < n {
			if data[i] == '\\' {
				i += 2
				continue
			}
			if data[i] == '"' {
				i++
				break
			}
			i++
		}
	case 't':
		i += 4
	case 'f':
		i += 5
	case 'n':
		i += 4
	case '[':
		depth := 1
		i++
		for i < n && depth > 0 {
			switch data[i] {
			case '[':
				depth++
			case ']':
				depth--
			case '"':
				i++
				for i < n {
					if data[i] == '\\' {
						i += 2
						continue
					}
					if data[i] == '"' {
						break
					}
					i++
				}
			}
			i++
		}
	case '{':
		depth := 1
		i++
		for i < n && depth > 0 {
			switch data[i] {
			case '{':
				depth++
			case '}':
				depth--
			case '"':
				i++
				for i < n {
					if data[i] == '\\' {
						i += 2
						continue
					}
					if data[i] == '"' {
						break
					}
					i++
				}
			}
			i++
		}
	default:
		for i < n {
			b := data[i]
			if (b >= '0' && b <= '9') || b == '.' || b == '-' || b == '+' || b == 'e' || b == 'E' {
				i++
			} else {
				break
			}
		}
	}
	for i < n && (data[i] == ' ' || data[i] == '\t' || data[i] == '\n' || data[i] == '\r' || data[i] == ',') {
		i++
	}
	return i
}

func merchantIDInArrayRaw(arrBytes []byte, target []byte) bool {
	i := 0
	n := len(arrBytes)
	for i < n {
		if arrBytes[i] == '"' {
			i++
			start := i
			for i < n {
				if arrBytes[i] == '\\' {
					i += 2
					continue
				}
				if arrBytes[i] == '"' {
					break
				}
				i++
			}
			if bytes.Equal(arrBytes[start:i], target) {
				return true
			}
			i++
		} else {
			i++
		}
	}
	return false
}

// scanTopLevel finds all top-level JSON objects in ONE forward scan.
func scanTopLevel(data []byte) (tx, cust, merch, term, lastTx []byte) {
	i := 0
	n := len(data)

	for i < n && data[i] != '{' {
		i++
	}
	if i >= n {
		return
	}
	i++

	depth := 1
	for i < n && depth > 0 {
		switch data[i] {
		case '{':
			depth++
			i++
		case '}':
			depth--
			i++
		case '"':
			j := i + 1
			for j < n {
				if data[j] == '\\' {
					j += 2
					continue
				}
				if data[j] == '"' {
					break
				}
				j++
			}
			key := data[i+1 : j]
			i = j + 1

			if depth == 1 {
				for i < n && data[i] != ':' {
					i++
				}
				i++
				for i < n && (data[i] == ' ' || data[i] == '\t' || data[i] == '\n' || data[i] == '\r') {
					i++
				}
				if i >= n {
					return
				}

				if data[i] == '{' {
					objStart := i
					objDepth := 1
					i++
					for i < n && objDepth > 0 {
						switch data[i] {
						case '{':
							objDepth++
						case '}':
							objDepth--
						case '"':
							i++
							for i < n {
								if data[i] == '\\' {
									i += 2
									continue
								}
								if data[i] == '"' {
									break
								}
								i++
							}
						}
						i++
					}

					switch len(key) {
					case 11:
						if key[0] == 't' && key[1] == 'r' {
							tx = data[objStart:i]
						}
					case 8:
						if key[0] == 'c' {
							cust = data[objStart:i]
						} else if key[0] == 'm' {
							merch = data[objStart:i]
						} else if key[0] == 't' && key[1] == 'e' {
							term = data[objStart:i]
						}
					case 16:
						if key[0] == 'l' {
							lastTx = data[objStart:i]
						}
					}
				} else {
					skipValue(data, &i, n)
				}
			}
		default:
			i++
		}
	}
	return
}

func skipValue(data []byte, i *int, n int) {
	if *i >= n {
		return
	}
	switch data[*i] {
	case '"':
		*i++
		for *i < n {
			if data[*i] == '\\' {
				*i += 2
				continue
			}
			if data[*i] == '"' {
				break
			}
			*i++
		}
		*i++
	case 't':
		*i += 4
	case 'f':
		*i += 5
	case 'n':
		*i += 4
	case '[':
		arrDepth := 1
		*i++
		for *i < n && arrDepth > 0 {
			switch data[*i] {
			case '[':
				arrDepth++
			case ']':
				arrDepth--
			case '"':
				*i++
				for *i < n {
					if data[*i] == '\\' {
						*i += 2
						continue
					}
					if data[*i] == '"' {
						break
					}
					*i++
				}
			}
			*i++
		}
	default:
		for *i < n {
			c := data[*i]
			if (c >= '0' && c <= '9') || c == '.' || c == '-' || c == '+' || c == 'e' || c == 'E' {
				*i++
			} else {
				break
			}
		}
	}
	for *i < n && (data[*i] == ' ' || data[*i] == '\t' || data[*i] == '\n' || data[*i] == '\r') {
		*i++
	}
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

var errInvalidJSON = &invalidJSONError{}

type invalidJSONError struct{}

func (e *invalidJSONError) Error() string { return "invalid json" }

type parsedTimestamp struct {
	hour, weekday int
	unixSec       int64
}

func parseTimestampBytes(b []byte) (ts parsedTimestamp) {
	if len(b) < 19 || b[10] != 'T' {
		return
	}
	h := int(b[11]-'0')*10 + int(b[12]-'0')
	min := int64(b[14]-'0')*10 + int64(b[15]-'0')
	sec := int64(b[17]-'0')*10 + int64(b[18]-'0')

	year := int(b[0]-'0')*1000 + int(b[1]-'0')*100 + int(b[2]-'0')*10 + int(b[3]-'0')
	month := int(b[5]-'0')*10 + int(b[6]-'0')
	day := int(b[8]-'0')*10 + int(b[9]-'0')

	t := [12]int{0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4}
	mAdj := month
	yAdj := year
	if mAdj < 3 {
		yAdj--
	}
	wd := (yAdj + yAdj/4 - yAdj/100 + yAdj/400 + t[mAdj-1] + day) % 7

	iYear := int64(year)
	iMonth := int64(month)
	if iMonth <= 2 {
		iYear--
		iMonth += 12
	}
	days := iYear*365 + iYear/4 - iYear/100 + iYear/400 + (153*(iMonth-3)+2)/5 + int64(day) - 719469

	ts.hour = h
	ts.weekday = (wd + 6) % 7
	ts.unixSec = days*86400 + int64(h)*3600 + min*60 + sec
	return
}

func parseUnixSecondsBytes(b []byte) int64 {
	return parseTimestampBytes(b).unixSec
}

// VectorizeJSON parses a raw JSON request body and returns a 14-dimensional
// feature vector. It uses a manual byte-oriented parser to avoid the overhead
// of json.Unmarshal on the full request struct.
func (v *Vectorizer) VectorizeJSON(data []byte) (model.Vector14, error) {
	var vec model.Vector14

	if len(data) < 2 || data[0] != '{' {
		return vec, errInvalidJSON
	}

	// Single scan: find all 5 top-level objects at once
	txObj, custObj, merchObj, termObj, lastTxObj := scanTopLevel(data)
	if txObj == nil {
		return vec, nil
	}

	tx := parseTxFields(txObj)
	vec[0] = clampFloat32(float32(tx.amount) * v.invMaxAmount)
	vec[1] = clampFloat32(float32(tx.installments) * v.invMaxInstall)

	if custObj != nil {
		cust := parseCustFields(custObj)

		var amountVsAvg float64
		if cust.avgAmount > 0 {
			amountVsAvg = tx.amount / cust.avgAmount
		}
		vec[2] = clampFloat32(float32(amountVsAvg) * v.invAmountRatio)
		vec[8] = clampFloat32(float32(cust.txCount24h) * v.invMaxTxCount)

		if merchObj != nil {
			merch := parseMerchFields(merchObj)

			if len(merch.id) > 0 && len(cust.knownMerchants) > 0 {
				if !merchantIDInArrayRaw(cust.knownMerchants, merch.id) {
					vec[11] = 1
				}
			} else if len(merch.id) > 0 {
				vec[11] = 1
			}

			vec[12] = float32(v.mccRisk.Get(merch.mcc))
			vec[13] = clampFloat32(float32(merch.avgAmount) * v.invMaxMerchantAvg)
		}
	}

	reqTS := parseTimestampBytes(tx.requestedAt)
	vec[3] = float32(reqTS.hour) / 23.0
	vec[4] = float32(reqTS.weekday) / 6.0

	if lastTxObj != nil {
		ltx := parseLastTxFields(lastTxObj)
		lastSec := parseUnixSecondsBytes(ltx.timestamp)
		minutes := float32(reqTS.unixSec-lastSec) / 60.0
		vec[5] = clampFloat32(minutes * v.invMaxMinutes)
		vec[6] = clampFloat32(float32(ltx.kmFromCurrent) * v.invMaxKm)
	} else {
		vec[5] = -1
		vec[6] = -1
	}

	if termObj != nil {
		term := parseTermFields(termObj)
		vec[7] = clampFloat32(float32(term.kmFromHome) * v.invMaxKm)

		if term.isOnline {
			vec[9] = 1
		}
		if term.cardPresent {
			vec[10] = 1
		}
	}

	return vec, nil
}
