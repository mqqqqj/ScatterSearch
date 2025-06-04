#include "distance.h"
#include <immintrin.h>
#include <x86intrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <nmmintrin.h>

float distance_l2sqr(const float *a, const float *b, size_t size)
{
    float sum = 0.0f;
    for (size_t i = 0; i < size; i++)
    {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

float distance_l2sqr_avx(const float *a, const float *b, size_t size)
{
    float sum = 0.0f;
    size_t i = 0;

    for (; i + 8 <= size; i += 8)
    {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        __m256 square = _mm256_mul_ps(diff, diff);

        float temp[8];
        _mm256_storeu_ps(temp, square);
        sum += temp[0] + temp[1] + temp[2] + temp[3] +
               temp[4] + temp[5] + temp[6] + temp[7];
    }

    for (; i < size; i++)
    {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }

    return sum;
}

float distance_l2sqr_sse(const float *a, const float *b, size_t size)
{
    float sum = 0.0f;
    size_t i = 0;

    for (; i + 4 <= size; i += 4)
    {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        __m128 diff = _mm_sub_ps(va, vb);
        __m128 square = _mm_mul_ps(diff, diff);

        float temp[4];
        _mm_storeu_ps(temp, square);
        sum += temp[0] + temp[1] + temp[2] + temp[3];
    }

    for (; i < size; i++)
    {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }

    return sum;
}

float distance_ip(const float *a, const float *b, size_t size)
{
    float sum = 0.0f;
    for (size_t i = 0; i < size; i++)
    {
        sum += a[i] * b[i];
    }
    return -sum;
}

float distance_ip_avx(const float *a, const float *b, size_t size)
{
    float result = 0;
#define AVX_DOT(addr1, addr2, dest, tmp1, tmp2) \
    tmp1 = _mm256_loadu_ps(addr1);              \
    tmp2 = _mm256_loadu_ps(addr2);              \
    tmp1 = _mm256_mul_ps(tmp1, tmp2);           \
    dest = _mm256_add_ps(dest, tmp1);

    __m256 sum;
    __m256 sum_1;
    __m256 sum_2;
    __m256 l0, l1;
    __m256 r0, r1;
    unsigned D = (size + 7) & ~7U;
    unsigned DR = D % 16;
    unsigned DD = D - DR;
    const float *l = a;
    const float *r = b;
    const float *e_l = l + DD;
    const float *e_r = r + DD;
    float unpack[8] __attribute__((aligned(32))) = {0, 0, 0, 0, 0, 0, 0, 0};
    sum = _mm256_loadu_ps(unpack);
    // float unpack_2[8] __attribute__((aligned(32))) = {0, 0, 0, 0, 0, 0, 0, 0};
    // sum_1 = _mm256_loadu_ps(unpack);
    // sum_2 = _mm256_loadu_ps(unpack_2);
    if (DR)
    {
        AVX_DOT(e_l, e_r, sum, l0, r0);
    }

    for (unsigned i = 0; i < DD; i += 16, l += 16, r += 16)
    {
        AVX_DOT(l, r, sum, l0, r0);
        AVX_DOT(l + 8, r + 8, sum, l1, r1);
        // AVX_DOT(l, r, sum_1, l0, r0);
        // AVX_DOT(l + 8, r + 8, sum_2, l1, r1);
    }
    // sum = _mm256_add_ps(sum_1, sum_2);
    _mm256_storeu_ps(unpack, sum);
    result = unpack[0] + unpack[1] + unpack[2] + unpack[3] + unpack[4] + unpack[5] + unpack[6] + unpack[7];
    return -result;
}

float distance_ip_avx_simple(const float *a, const float *b, size_t size)
{
    float sum = 0.0f;
    size_t i = 0;

    for (; i + 8 <= size; i += 8)
    {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 prod = _mm256_mul_ps(va, vb);

        float temp[8];
        _mm256_storeu_ps(temp, prod);
        sum += temp[0] + temp[1] + temp[2] + temp[3] +
               temp[4] + temp[5] + temp[6] + temp[7];
    }

    for (; i < size; i++)
    {
        sum += a[i] * b[i];
    }

    return -sum;
}

float distance_ip_sse(const float *a, const float *b, size_t size)
{
    float sum = 0.0f;
    size_t i = 0;

    for (; i + 4 <= size; i += 4)
    {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        __m128 prod = _mm_mul_ps(va, vb);

        float temp[4];
        _mm_storeu_ps(temp, prod);
        sum += temp[0] + temp[1] + temp[2] + temp[3];
    }

    for (; i < size; i++)
    {
        sum += a[i] * b[i];
    }

    return -sum;
}
