#include "simd_knn.h"
#include <immintrin.h>
#include <string.h>
#include <stdio.h>

#define DIM 14
#define BLOCK_SIZE 112 /* 14 dims * 8 vectors */
#define K_NEIGHBORS 5
#define PAD_VALUE 32767 /* INT16_MAX, outside real data range */

static inline void sort_topK(int32_t top_d[K_NEIGHBORS], uint8_t top_l[K_NEIGHBORS], int *worst_idx, int32_t new_d, uint8_t new_l) {
    if (new_d >= top_d[*worst_idx]) return;
    top_d[*worst_idx] = new_d;
    top_l[*worst_idx] = new_l;
    int wi = 0;
    int32_t wv = top_d[0];
    for (int j = 1; j < K_NEIGHBORS; j++) {
        if (top_d[j] > wv) { wv = top_d[j]; wi = j; }
    }
    *worst_idx = wi;
}

/* Find top-NPROBE nearest centroids using scalar (centroids are float32) */
static void find_top_centroids(
    const float *centroids,
    int nlist,
    const int16_t query[14],
    int nprobe,
    int *out_probes
) {
    if (nprobe > nlist) nprobe = nlist;
    if (nprobe < 1) nprobe = 1;

    float dists[2048]; /* max nlist supported */
    memset(dists, 0, nlist * sizeof(float));

    float qf[DIM];
    for (int d = 0; d < DIM; d++) {
        qf[d] = (float)query[d] / 10000.0f;
    }

    for (int d = 0; d < DIM; d++) {
        float qd = qf[d];
        for (int ci = 0; ci < nlist; ci++) {
            float diff = centroids[ci * DIM + d] - qd;
            dists[ci] += diff * diff;
        }
    }

    /* Select top-nprobe using a fixed-size heap (nprobe <= 2048) */
    float probe_dist[2048];
    for (int i = 0; i < nprobe; i++) { out_probes[i] = i; probe_dist[i] = dists[i]; }

    int worst = 0;
    for (int i = 1; i < nprobe; i++) {
        if (probe_dist[i] > probe_dist[worst]) worst = i;
    }

    for (int ci = nprobe; ci < nlist; ci++) {
        if (dists[ci] < probe_dist[worst]) {
            out_probes[worst] = ci;
            probe_dist[worst] = dists[ci];
            worst = 0;
            for (int j = 1; j < nprobe; j++) {
                if (probe_dist[j] > probe_dist[worst]) worst = j;
            }
        }
    }
}

/* Scan blocks using AVX2, processing 8 vectors per block */
static void scan_blocks_avx2(
    const int16_t *blocks,
    const uint8_t *labels,
    int start_block,
    int end_block,
    const __m256i q_vecs[DIM],
    int32_t top_d[K_NEIGHBORS],
    uint8_t top_l[K_NEIGHBORS],
    int *worst_idx
) {
    __m256i max_dist = _mm256_set1_epi32(INT32_MAX);

    for (int bi = start_block; bi < end_block; bi++) {
        const int16_t *block = blocks + bi * BLOCK_SIZE;

        /* Load first 8 int16s (dim 0) to check for padding */
        __m128i first_dim = _mm_loadu_si128((const __m128i *)block);
        __m128i padding_val = _mm_set1_epi16(PAD_VALUE);
        __m128i pad_mask128 = _mm_cmpeq_epi16(first_dim, padding_val);
        uint8_t pad_mask = (uint8_t)_mm_movemask_epi8(pad_mask128);
        /* Each int16 produces 1 bit in the movemask, but movemask works per-byte.
         * For int16, each element contributes 2 bits. We extract every other bit. */
        uint8_t pad_slots = 0;
        for (int s = 0; s < 8; s++) {
            if (pad_mask & (1 << (s * 2))) pad_slots |= (1 << s);
        }

        __m256i acc0 = _mm256_setzero_si256();
        __m256i acc1 = _mm256_setzero_si256();

        for (int d = 0; d < DIM; d += 2) {
            /* Load 8 int16 values for dimension d (128 bits = 8 x 16) */
            __m128i raw0 = _mm_loadu_si128((const __m128i *)(block + d * 8));
            /* Load 8 int16 values for dimension d+1 */
            __m128i raw1 = _mm_loadu_si128((const __m128i *)(block + (d + 1) * 8));

            /* Widen to 32-bit and subtract query */
            __m256i diff0 = _mm256_sub_epi32(_mm256_cvtepi16_epi32(raw0), q_vecs[d]);
            __m256i diff1 = _mm256_sub_epi32(_mm256_cvtepi16_epi32(raw1), q_vecs[d + 1]);

            /* Prevent int32 overflow: shift right by 2 before squaring.
             * Max diff 20000 >> 2 = 5000, 5000^2 * 14 = 350M < INT32_MAX.
             * Distance ordering preserved since x>y => x/4>y/4. */
            diff0 = _mm256_srai_epi32(diff0, 2);
            diff1 = _mm256_srai_epi32(diff1, 2);

            /* Square and accumulate */
            acc0 = _mm256_add_epi32(acc0, _mm256_mullo_epi32(diff0, diff0));
            acc1 = _mm256_add_epi32(acc1, _mm256_mullo_epi32(diff1, diff1));
        }

        /* Sum acc0 + acc1 */
        __m256i total = _mm256_add_epi32(acc0, acc1);

        /* Replace padded slots with INT32_MAX (scalar fallback for simplicity) */
        int32_t dists[8];
        _mm256_storeu_si256((__m256i *)dists, total);
        for (int s = 0; s < 8; s++) {
            if (pad_slots & (1 << s)) dists[s] = INT32_MAX;
        }

        /* Compare with current worst: closer = lanes where total < top[worst] */
        int32_t worst_val = top_d[*worst_idx];
        uint32_t mask = 0;
        for (int s = 0; s < 8; s++) {
            if (dists[s] < worst_val) mask |= (1u << s);
        }

        if (mask == 0) continue;

        const uint8_t *block_labels = labels + bi * 8;

        while (mask) {
            int slot = __builtin_ctz(mask);
            mask &= mask - 1;

            if (dists[slot] < top_d[*worst_idx]) {
                top_d[*worst_idx] = dists[slot];
                top_l[*worst_idx] = block_labels[slot];
                int wi = 0;
                int32_t wv = top_d[0];
                for (int j = 1; j < K_NEIGHBORS; j++) {
                    if (top_d[j] > wv) { wv = top_d[j]; wi = j; }
                }
                *worst_idx = wi;
            }
        }
    }
}

int knn_fraud_count_avx2(
    const int16_t *blocks,
    const uint8_t *labels,
    const uint32_t *offsets,
    const float *centroids,
    int k,
    int nprobe,
    const int16_t query[14],
    int *out_fraud_count
) {
    /* Prepare query vectors for AVX2: broadcast each dimension */
    __m256i q_vecs[DIM];
    for (int d = 0; d < DIM; d++) {
        q_vecs[d] = _mm256_set1_epi32((int32_t)query[d]);
    }

    /* Find nearest centroids */
    if (nprobe > k) nprobe = k; /* clamp to nlist (k param is nlist here) */
    int probes[2048];
    find_top_centroids(centroids, k, query, nprobe, probes);

    int total_blocks = 0;

    /* Top-K nearest neighbors */
    int32_t top_d[K_NEIGHBORS];
    uint8_t top_l[K_NEIGHBORS];
    for (int i = 0; i < K_NEIGHBORS; i++) { top_d[i] = INT32_MAX; top_l[i] = 0; }
    int worst_idx = 0;

    int last_ci = -1;
    for (int pi = 0; pi < nprobe; pi++) {
        int ci = probes[pi];
        if (ci == last_ci) continue;
        last_ci = ci;

        uint32_t start = offsets[ci];
        uint32_t end = offsets[ci + 1];
        total_blocks += (end - start);
        if (start == end) continue;

        scan_blocks_avx2(blocks, labels, start, end, q_vecs, top_d, top_l, &worst_idx);
    }

    /* Count fraud */
    int fraud = 0;
    for (int i = 0; i < K_NEIGHBORS; i++) {
        if (top_l[i] == 1) fraud++;
    }
    *out_fraud_count = fraud;
    return fraud;
}

/*
 * knn_fraud_count_retry: boundary-aware search with incremental retry.
 *
 * Phase 1: scan nprobe clusters, count fraud.
 * Phase 2: if fraud in [boundary_lo, boundary_hi], scan retry_extra more
 *          clusters using the SAME top-K heap (incremental, not restart).
 */
int knn_fraud_count_retry(
    const int16_t *blocks,
    const uint8_t *labels,
    const uint32_t *offsets,
    const float *centroids,
    int nlist,
    int nprobe,
    int retry_extra,
    int boundary_lo,
    int boundary_hi,
    const int16_t query[14],
    int *out_fraud_count
) {
    __m256i q_vecs[DIM];
    for (int d = 0; d < DIM; d++) {
        q_vecs[d] = _mm256_set1_epi32((int32_t)query[d]);
    }

    int total_nprobe = nprobe + retry_extra;
    if (total_nprobe > nlist) total_nprobe = nlist;
    if (nprobe > nlist) nprobe = nlist;

    /* Pre-compute all candidate centroids up to total_nprobe */
    int probes[2048];
    find_top_centroids(centroids, nlist, query, total_nprobe, probes);

    int32_t top_d[K_NEIGHBORS];
    uint8_t top_l[K_NEIGHBORS];
    for (int i = 0; i < K_NEIGHBORS; i++) { top_d[i] = INT32_MAX; top_l[i] = 0; }
    int worst_idx = 0;

    /* Phase 1: scan first nprobe clusters */
    for (int pi = 0; pi < nprobe; pi++) {
        int ci = probes[pi];
        uint32_t start = offsets[ci];
        uint32_t end = offsets[ci + 1];
        if (start == end) continue;
        scan_blocks_avx2(blocks, labels, start, end, q_vecs, top_d, top_l, &worst_idx);
    }

    int fraud = 0;
    for (int i = 0; i < K_NEIGHBORS; i++) {
        if (top_l[i] == 1) fraud++;
    }

    /* Phase 2: boundary retry — incrementally scan extra clusters */
    if (fraud >= boundary_lo && fraud <= boundary_hi && retry_extra > 0) {
        for (int pi = nprobe; pi < total_nprobe; pi++) {
            int ci = probes[pi];
            uint32_t start = offsets[ci];
            uint32_t end = offsets[ci + 1];
            if (start == end) continue;
            scan_blocks_avx2(blocks, labels, start, end, q_vecs, top_d, top_l, &worst_idx);
        }
        fraud = 0;
        for (int i = 0; i < K_NEIGHBORS; i++) {
            if (top_l[i] == 1) fraud++;
        }
    }

    *out_fraud_count = fraud;
    return fraud;
}
