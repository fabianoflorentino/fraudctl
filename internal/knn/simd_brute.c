#include "simd_brute.h"
#include <immintrin.h>
#include <string.h>

#define DIM 14
#define BLOCK 8
#define PAD 32767

static inline void update_topK(int32_t *top_d, uint8_t *top_l, int K, int32_t d, uint8_t l) {
    int worst = 0;
    int32_t wv = top_d[0];
    for (int j = 1; j < K; j++) {
        if (top_d[j] > wv) {
            wv = top_d[j];
            worst = j;
        }
    }
    if (d < wv) {
        top_d[worst] = d;
        top_l[worst] = l;
    }
}

int brute_fraud_count_avx2(
    const int16_t *data_soa,
    const uint8_t *labels,
    int N,
    const int16_t query[14],
    int K,
    int *out_fraud_count
) {
    /* Prepare query: broadcast each dimension to 8 int16 */
    __m128i q[DIM];
    for (int d = 0; d < DIM; d++) {
        q[d] = _mm_set1_epi16(query[d]);
    }

    int32_t top_d[32];
    uint8_t top_l[32];
    for (int i = 0; i < K; i++) {
        top_d[i] = 0x7FFFFFFF;
        top_l[i] = 0;
    }

    int blocks = (N + BLOCK - 1) / BLOCK;
    int32_t dists[8];
    uint8_t pad_count[8];

    for (int bi = 0; bi < blocks; bi++) {
        __m256i acc0 = _mm256_setzero_si256();
        __m256i acc1 = _mm256_setzero_si256();

        for (int d = 0; d < DIM; d += 2) {
            const int16_t *ptr0 = data_soa + d * N + bi * BLOCK;
            __m128i raw0 = _mm_loadu_si128((const __m128i *)ptr0);
            __m256i v0 = _mm256_cvtepi16_epi32(raw0);
            __m256i diff0 = _mm256_sub_epi32(v0, _mm256_cvtepi16_epi32(_mm_set1_epi16(query[d])));
            diff0 = _mm256_srai_epi32(diff0, 2);
            acc0 = _mm256_add_epi32(acc0, _mm256_mullo_epi32(diff0, diff0));

            if (d + 1 < DIM) {
                const int16_t *ptr1 = data_soa + (d + 1) * N + bi * BLOCK;
                __m128i raw1 = _mm_loadu_si128((const __m128i *)ptr1);
                __m256i v1 = _mm256_cvtepi16_epi32(raw1);
                __m256i diff1 = _mm256_sub_epi32(v1, _mm256_cvtepi16_epi32(_mm_set1_epi16(query[d + 1])));
                diff1 = _mm256_srai_epi32(diff1, 2);
                acc1 = _mm256_add_epi32(acc1, _mm256_mullo_epi32(diff1, diff1));
            }
        }

        /* Sum acc0 + acc1 */
        __m256i total = _mm256_add_epi32(acc0, acc1);
        _mm256_storeu_si256((__m256i *)dists, total);

        /* Check for padding */
        const int16_t *first_dim = data_soa + bi * BLOCK;
        __m128i pad_mask128 = _mm_cmpeq_epi16(_mm_loadu_si128((const __m128i *)first_dim), _mm_set1_epi16(PAD));
        uint8_t pm = (uint8_t)_mm_movemask_epi8(pad_mask128);
        for (int s = 0; s < 8; s++) {
            if (pm & (1 << (s * 2))) dists[s] = 0x7FFFFFFF;
        }

        /* Update top-K */
        int count = (bi + 1) * BLOCK > N ? N - bi * BLOCK : BLOCK;
        const uint8_t *lbl = labels + bi * BLOCK;

        for (int s = 0; s < count; s++) {
            if (dists[s] < top_d[0]) {
                int worst = 0;
                int32_t wv = top_d[0];
                for (int j = 1; j < K; j++) {
                    if (top_d[j] > wv) {
                        wv = top_d[j];
                        worst = j;
                    }
                }
                if (dists[s] < wv) {
                    top_d[worst] = dists[s];
                    top_l[worst] = lbl[s];
                }
            }
        }
    }

    int fraud = 0;
    for (int i = 0; i < K; i++) {
        fraud += top_l[i];
    }
    *out_fraud_count = fraud;
    return fraud;
}
