#include "distance.h"
#include <immintrin.h>
#include <x86intrin.h>

float distance_l2sqr(const float *a, const float *b, size_t size)
{
    float result = 0;
#define AVX_L2SQR(addr1, addr2, dest, tmp1, tmp2) \
    tmp1 = _mm256_loadu_ps(addr1);                \
    tmp2 = _mm256_loadu_ps(addr2);                \
    tmp1 = _mm256_sub_ps(tmp1, tmp2);             \
    tmp1 = _mm256_mul_ps(tmp1, tmp1);             \
    dest = _mm256_add_ps(dest, tmp1);

    __m256 sum;
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
    if (DR)
    {
        AVX_L2SQR(e_l, e_r, sum, l0, r0);
    }

    for (unsigned i = 0; i < DD; i += 16, l += 16, r += 16)
    {
        AVX_L2SQR(l, r, sum, l0, r0);
        AVX_L2SQR(l + 8, r + 8, sum, l1, r1);
    }
    _mm256_storeu_ps(unpack, sum);
    result = unpack[0] + unpack[1] + unpack[2] + unpack[3] + unpack[4] + unpack[5] + unpack[6] + unpack[7];
    return result;
}

float distance_ip(const float *a, const float *b, size_t size)
{
    float result = 0;
#define AVX_DOT(addr1, addr2, dest, tmp1, tmp2) \
    tmp1 = _mm256_loadu_ps(addr1);              \
    tmp2 = _mm256_loadu_ps(addr2);              \
    tmp1 = _mm256_mul_ps(tmp1, tmp2);           \
    dest = _mm256_add_ps(dest, tmp1);

    __m256 sum;
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
    if (DR)
    {
        AVX_DOT(e_l, e_r, sum, l0, r0);
    }

    for (unsigned i = 0; i < DD; i += 16, l += 16, r += 16)
    {
        AVX_DOT(l, r, sum, l0, r0);
        AVX_DOT(l + 8, r + 8, sum, l1, r1);
    }
    _mm256_storeu_ps(unpack, sum);
    result = unpack[0] + unpack[1] + unpack[2] + unpack[3] + unpack[4] + unpack[5] + unpack[6] + unpack[7];
    return -result;
} 