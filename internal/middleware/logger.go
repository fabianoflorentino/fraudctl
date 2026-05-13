// Package middleware provides low-overhead request telemetry for fraudctl.
//
// Design goals:
//   - Zero allocations and zero locks in the hot path (atomic counters only)
//   - Per-request log.Printf is intentionally removed from the hot path;
//     only outlier requests (latency > outlierThresholdUs) are captured
//   - Periodic aggregate stats printed every interval via StartReporter
//   - Border-zone tracking: fraudCount ∈ {2,3} with k=5 are the FP/FN candidates
//   - Outlier ring buffer: last N slow/anomalous requests captured lock-free
//     and drained by the reporter goroutine for diagnosis
package middleware

import (
	"fmt"
	"log"
	"sync/atomic"
	"time"
	"unsafe"
)

var telemetryEnabled uint32 = 1

// SetEnabled toggles telemetry collection globally.
func SetEnabled(enabled bool) {
	if enabled {
		atomic.StoreUint32(&telemetryEnabled, 1)
		return
	}
	atomic.StoreUint32(&telemetryEnabled, 0)
}

// IsEnabled returns whether telemetry collection is active.
func IsEnabled() bool {
	return atomic.LoadUint32(&telemetryEnabled) != 0
}

// outlierThresholdUs — requests above this value are captured in the ring buffer.
const outlierThresholdUs = 2000 // 2ms

// Latency histogram buckets (inclusive upper bound in microseconds).
var bucketBounds = [...]int64{250, 500, 750, 1000, 1500, 2000, 3000, 5000, 10000}

const numBuckets = len(bucketBounds) + 1 // +1 for >10ms

// Stats holds lock-free counters updated on every request.
type Stats struct {
	total        int64
	errors       int64
	fraud        int64
	legit        int64
	borderZone   int64 // fraudCount in {2,3} — ambiguous predictions near threshold
	parseErrors  int64 // VectorizeJSON failures
	totalLatency int64 // nanoseconds; used for mean
	buckets      [numBuckets]int64
	_            [8]int64 // cache-line padding
}

var global Stats

// outlierEntry holds data for a single slow request.
// Kept small (fits in a cache line) to minimise ring-buffer cost.
type outlierEntry struct {
	latencyUs  int64
	vecUs      int64
	knnUs      int64
	fraudCount int32
	approved   bool
	parseErr   bool
}

// outlierRing is a fixed-size lock-free ring buffer using a single atomic
// write index. Readers (reporter goroutine) only run every 10s — they don't
// need to coordinate with writers beyond the atomic index.
const ringSize = 64 // power of 2; wraps naturally with & mask

type outlierRing struct {
	entries [ringSize]outlierEntry
	head    int64 // monotonic write counter; index = head & (ringSize-1)
}

var ring outlierRing

// Record updates counters and, when the request is slow or anomalous,
// stores it in the outlier ring. No locks, no allocations.
func Record(duration, vecDuration, knnDuration time.Duration, fraudCount int, approved, parseErr bool) {
	if atomic.LoadUint32(&telemetryEnabled) == 0 {
		return
	}

	atomic.AddInt64(&global.total, 1)
	atomic.AddInt64(&global.totalLatency, int64(duration))

	us := duration.Microseconds()
	addToBucket(us)

	if parseErr {
		atomic.AddInt64(&global.parseErrors, 1)
		atomic.AddInt64(&global.errors, 1)
		// Parse errors are always captured as outliers.
		storeOutlier(us, vecDuration.Microseconds(), 0, fraudCount, approved, true)
		return
	}

	if approved {
		atomic.AddInt64(&global.legit, 1)
	} else {
		atomic.AddInt64(&global.fraud, 1)
	}

	border := fraudCount == 2 || fraudCount == 3
	if border {
		atomic.AddInt64(&global.borderZone, 1)
	}

	// Capture outliers: slow requests OR border-zone decisions.
	// Border-zone entries help correlate FP/FN with latency spikes.
	if us >= outlierThresholdUs || border {
		storeOutlier(us, vecDuration.Microseconds(), knnDuration.Microseconds(), fraudCount, approved, false)
	}
}

// storeOutlier writes an entry into the ring buffer atomically.
// Old entries are silently overwritten when the ring is full.
func storeOutlier(latUs, vecUs, knnUs int64, fraudCount int, approved, parseErr bool) {
	idx := atomic.AddInt64(&ring.head, 1) - 1
	slot := idx & (ringSize - 1)
	// Direct write — no CAS needed; the reporter only reads during quiet periods.
	ring.entries[slot] = outlierEntry{
		latencyUs:  latUs,
		vecUs:      vecUs,
		knnUs:      knnUs,
		fraudCount: int32(fraudCount),
		approved:   approved,
		parseErr:   parseErr,
	}
}

// addToBucket increments the histogram bucket for us microseconds.
func addToBucket(us int64) {
	for i, bound := range bucketBounds {
		if us <= bound {
			atomic.AddInt64((*int64)(unsafe.Pointer(&global.buckets[i])), 1)
			return
		}
	}
	atomic.AddInt64((*int64)(unsafe.Pointer(&global.buckets[numBuckets-1])), 1)
}

// StartReporter launches a background goroutine that prints aggregate stats
// and drains the outlier ring every interval. Call once from main.
func StartReporter(interval time.Duration) {
	if atomic.LoadUint32(&telemetryEnabled) == 0 {
		return
	}

	go func() {
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		var prevTotal, prevTime, prevHead int64
		prevTime = time.Now().UnixNano()

		for range ticker.C {
			printStats(&prevTotal, &prevTime)
			drainOutliers(&prevHead)
		}
	}()
}

func printStats(prevTotal, prevTime *int64) {
	now := time.Now().UnixNano()
	total := atomic.LoadInt64(&global.total)
	errors := atomic.LoadInt64(&global.errors)
	fraud := atomic.LoadInt64(&global.fraud)
	legit := atomic.LoadInt64(&global.legit)
	border := atomic.LoadInt64(&global.borderZone)
	parseErr := atomic.LoadInt64(&global.parseErrors)
	totalLat := atomic.LoadInt64(&global.totalLatency)

	delta := total - *prevTotal
	elapsed := float64(now-*prevTime) / 1e9
	rps := float64(delta) / elapsed

	var meanUs int64
	if total > 0 {
		meanUs = totalLat / total / 1e3
	}

	hist := buildHistogram()

	log.Printf(
		"STATS rps=%.1f total=%d fraud=%d legit=%d border_zone=%d parse_errors=%d http_errors=%d mean_us=%d | hist_us: %s",
		rps, total, fraud, legit, border, parseErr, errors-parseErr, meanUs, hist,
	)

	*prevTotal = total
	*prevTime = now
}

// drainOutliers prints any new entries in the ring since the last drain.
// Runs in the reporter goroutine — allocations are fine here.
func drainOutliers(prevHead *int64) {
	head := atomic.LoadInt64(&ring.head)
	newEntries := head - *prevHead
	if newEntries <= 0 {
		return
	}
	// Cap to ring size to avoid re-reading overwritten slots.
	if newEntries > ringSize {
		newEntries = ringSize
	}

	start := head - newEntries
	for i := int64(0); i < newEntries; i++ {
		slot := (start + i) & (ringSize - 1)
		e := ring.entries[slot] // snapshot
		log.Printf(
			"OUTLIER latency_us=%-6d vec_us=%-5d knn_us=%-5d fraud_count=%d approved=%-5v border=%-5v parse_err=%v",
			e.latencyUs, e.vecUs, e.knnUs, e.fraudCount, e.approved,
			e.fraudCount == 2 || e.fraudCount == 3, e.parseErr,
		)
	}

	*prevHead = head
}

func buildHistogram() string {
	labels := [...]string{"≤250", "≤500", "≤750", "≤1ms", "≤1.5ms", "≤2ms", "≤3ms", "≤5ms", "≤10ms", ">10ms"}
	var buf [256]byte
	b := buf[:0]
	for i := 0; i < numBuckets; i++ {
		count := atomic.LoadInt64((*int64)(unsafe.Pointer(&global.buckets[i])))
		if i > 0 {
			b = append(b, ' ')
		}
		b = fmt.Appendf(b, "%s:%d", labels[i], count)
	}
	return string(b)
}
