package handler

import (
	"bytes"

	"github.com/fabianoflorentino/fraudctl/internal/model"
)

func parseFraudRequest(body []byte) model.RawRequest {
	var r model.RawRequest

	var ctx string
	var merchantID []byte
	var knownMerchant bool

	pos := 0
	for pos < len(body) {
		quote := bytes.IndexByte(body[pos:], '"')
		if quote < 0 {
			break
		}
		pos += quote

		start := pos + 1
		end := start
		for end < len(body) && body[end] != '"' {
			end++
		}
		if end >= len(body) {
			break
		}

		after := end + 1
		for after < len(body) && body[after] <= ' ' {
			after++
		}

		if after < len(body) && body[after] == ':' {
			key := string(body[start:end])
			pos = after + 1

			for pos < len(body) && body[pos] <= ' ' {
				pos++
			}
			if pos >= len(body) {
				break
			}

			switch body[pos] {
			case '{':
				ctx = key
				pos++

			case '"':
				valStart := pos + 1
				valEnd := valStart
				for valEnd < len(body) && body[valEnd] != '"' {
					valEnd++
				}
				if valEnd >= len(body) {
					break
				}
				val := body[valStart:valEnd]

				switch ctx {
				case "transaction":
					if key == "requested_at" {
						r.RequestedAt = val
					}
				case "merchant":
					switch key {
					case "id":
						merchantID = val
					case "mcc":
						r.MerchantMCC = val
					}
				case "last_transaction":
					if key == "timestamp" {
						r.LastTimestamp = val
						r.HasLastTx = true
					}
				}
				pos = valEnd + 1

		case 't':
			if ctx == "terminal" {
				switch key {
				case "is_online":
					r.IsOnline = true
				case "card_present":
					r.CardPresent = true
				}
			}
			pos += 4

		case 'f':
			if ctx == "terminal" {
				switch key {
				case "is_online":
					r.IsOnline = false
				case "card_present":
					r.CardPresent = false
				}
			}
			pos += 5

			case 'n':
				if key == "last_transaction" || ctx == "last_transaction" {
					r.HasLastTx = false
				}
				pos += 4

			case '[':
				if ctx == "customer" && key == "known_merchants" && len(merchantID) > 0 {
					pos++
					r.KnownMerchant = scanMerchantArray(body, &pos, merchantID)
					// If found, scanMerchantArray already consumed the array
				} else {
					for pos < len(body) && body[pos] != ']' {
						pos++
					}
					pos++
				}

			default:
				valStart := pos
				for pos < len(body) && (body[pos] == '-' || body[pos] == '+' || body[pos] == '.' || (body[pos] >= '0' && body[pos] <= '9')) {
					pos++
				}
				num := body[valStart:pos]

				switch ctx {
				case "transaction":
					switch key {
					case "amount":
						r.Amount = parseFloat32(num)
					case "installments":
						r.Installments = uint8(parseUint(num))
					}
				case "customer":
					switch key {
					case "avg_amount":
						r.AvgAmount = parseFloat32(num)
					case "tx_count_24h":
						r.TxCount24h = uint16(parseUint(num))
					}
				case "merchant":
					if key == "avg_amount" {
						r.MerchantAvg = parseFloat32(num)
					}
				case "terminal":
					if key == "km_from_home" {
						r.KmFromHome = parseFloat32(num)
					}
				case "last_transaction":
					if key == "km_from_current" {
						r.LastKmFromCur = parseFloat32(num)
					}
				}
			}

		} else {
			pos = end + 1
		}
	}

	r.MerchantID = merchantID
	if len(merchantID) > 0 && !knownMerchant {
		r.KnownMerchant = checkKnownMerchant(body, merchantID)
	}
	return r
}

func scanMerchantArray(body []byte, pos *int, merchantID []byte) bool {
	for *pos < len(body) {
		for *pos < len(body) && body[*pos] != '"' {
			if body[*pos] == ']' {
				return false
			}
			(*pos)++
		}
		if *pos >= len(body) || body[*pos] != '"' {
			return false
		}
		(*pos)++
		elemStart := *pos
		for *pos < len(body) && body[*pos] != '"' {
			(*pos)++
		}
		if *pos >= len(body) {
			return false
		}
		elem := body[elemStart:*pos]
		if len(elem) == len(merchantID) && bytes.Equal(elem, merchantID) {
			// Skip to end of array
			for *pos < len(body) && body[*pos] != ']' {
				(*pos)++
			}
			if *pos < len(body) {
				(*pos)++
			}
			return true
		}
		(*pos)++
	}
	return false
}

func checkKnownMerchant(body []byte, merchantID []byte) bool {
	idx := bytes.Index(body, []byte(`"known_merchants":[`))
	if idx < 0 {
		return false
	}
	pos := idx + len(`"known_merchants":[`)
	for pos < len(body) {
		for pos < len(body) && body[pos] != '"' {
			if body[pos] == ']' {
				return false
			}
			pos++
		}
		if pos >= len(body) || body[pos] != '"' {
			return false
		}
		pos++
		elemStart := pos
		for pos < len(body) && body[pos] != '"' {
			pos++
		}
		if pos >= len(body) {
			return false
		}
		elem := body[elemStart:pos]
		if len(elem) == len(merchantID) && bytes.Equal(elem, merchantID) {
			return true
		}
		pos++
	}
	return false
}

func parseFloat32(buf []byte) float32 {
	if len(buf) == 0 {
		return 0
	}
	neg := buf[0] == '-'
	pos := 0
	if neg || buf[0] == '+' {
		pos++
	}
	var val float32
	var dec float32 = 1
	for pos < len(buf) && buf[pos] >= '0' && buf[pos] <= '9' {
		val = val*10 + float32(buf[pos]-'0')
		pos++
	}
	if pos < len(buf) && buf[pos] == '.' {
		pos++
		for pos < len(buf) && buf[pos] >= '0' && buf[pos] <= '9' {
			val = val*10 + float32(buf[pos]-'0')
			dec *= 10
			pos++
		}
	}
	if neg {
		return -val / dec
	}
	return val / dec
}

func parseUint(buf []byte) uint {
	if len(buf) == 0 {
		return 0
	}
	pos := 0
	if buf[0] == '+' {
		pos++
	}
	var val uint
	for pos < len(buf) && buf[pos] >= '0' && buf[pos] <= '9' {
		val = val*10 + uint(buf[pos]-'0')
		pos++
	}
	return val
}
