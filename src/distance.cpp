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

float distance_l2sqr_sse(const float *a, const float *b, size_t size)
{
    float result = 0;
#define SSE_L2SQR(addr1, addr2, dest, tmp1, tmp2) \
    tmp1 = _mm_load_ps(addr1);                    \
    tmp2 = _mm_load_ps(addr2);                    \
    tmp1 = _mm_sub_ps(tmp1, tmp2);                \
    tmp1 = _mm_mul_ps(tmp1, tmp1);                \
    dest = _mm_add_ps(dest, tmp1);

    __m128 sum;
    __m128 l0, l1, l2, l3;
    __m128 r0, r1, r2, r3;
    unsigned D = (size + 3) & ~3U;
    unsigned DR = D % 16;
    unsigned DD = D - DR;
    const float *l = a;
    const float *r = b;
    const float *e_l = l + DD;
    const float *e_r = r + DD;
    float unpack[4] __attribute__((aligned(16))) = {0, 0, 0, 0};

    sum = _mm_load_ps(unpack);
    switch (DR)
    {
    case 12:
        SSE_L2SQR(e_l + 8, e_r + 8, sum, l2, r2);
    case 8:
        SSE_L2SQR(e_l + 4, e_r + 4, sum, l1, r1);
    case 4:
        SSE_L2SQR(e_l, e_r, sum, l0, r0);
    default:
        break;
    }
    for (unsigned i = 0; i < DD; i += 16, l += 16, r += 16)
    {
        SSE_L2SQR(l, r, sum, l0, r0);
        SSE_L2SQR(l + 4, r + 4, sum, l1, r1);
        SSE_L2SQR(l + 8, r + 8, sum, l2, r2);
        SSE_L2SQR(l + 12, r + 12, sum, l3, r3);
    }
    _mm_storeu_ps(unpack, sum);
    result += unpack[0] + unpack[1] + unpack[2] + unpack[3];

    return result;
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

float distance_ip_avx_simple(const float *a, const float *b, size_t size)
{
    float sum = 0.0f;
    size_t i = 0;
    __m256 sum_vec = _mm256_setzero_ps();
    float temp[8] __attribute__((aligned(32)));
    for (; i + 8 <= size; i += 8)
    {
        sum_vec = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), sum_vec);
    }
    _mm256_storeu_ps(temp, sum_vec);
    sum = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
    for (; i < size; i++)
    {
        sum += a[i] * b[i];
    }
    return -sum;
}

float distance_ip_avx_simple_unroll2(const float *a, const float *b, size_t size)
{
    float sum = 0.0f;
    size_t i = 0;
    __m256 sum_vec1 = _mm256_setzero_ps();
    __m256 sum_vec2 = _mm256_setzero_ps();
    float temp[8] __attribute__((aligned(32)));
    
    // 主循环：每次处理16个元素（2个AVX256寄存器）
    for (; i + 16 <= size; i += 16)
    {
        // 第一个寄存器：处理i到i+7的元素
        sum_vec1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), 
                                  _mm256_loadu_ps(b + i), 
                                  sum_vec1);
        
        // 第二个寄存器：处理i+8到i+15的元素
        sum_vec2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), 
                                  _mm256_loadu_ps(b + i + 8), 
                                  sum_vec2);
    }
    
    // 合并两个寄存器的结果
    sum_vec1 = _mm256_add_ps(sum_vec1, sum_vec2);
    
    // 处理剩余的元素（8个一组）
    for (; i + 8 <= size; i += 8)
    {
        sum_vec1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), 
                                  _mm256_loadu_ps(b + i), 
                                  sum_vec1);
    }
    
    // 将结果存储到临时数组并累加
    _mm256_storeu_ps(temp, sum_vec1);
    sum = temp[0] + temp[1] + temp[2] + temp[3] + 
          temp[4] + temp[5] + temp[6] + temp[7];
    
    // 处理最后剩余的元素
    for (; i < size; i++)
    {
        sum += a[i] * b[i];
    }
    
    return -sum;
}

float distance_ip_sse(const float *a, const float *b, size_t size)
{
    float result = 0;
#define SSE_DOT(addr1, addr2, dest, tmp1, tmp2) \
    tmp1 = _mm_loadu_ps(addr1);                 \
    tmp2 = _mm_loadu_ps(addr2);                 \
    tmp1 = _mm_mul_ps(tmp1, tmp2);              \
    dest = _mm_add_ps(dest, tmp1);

    __m128 sum;
    __m128 l0, l1;
    __m128 r0, r1;
    unsigned D = (size + 3) & ~3U; // 向上取整到4的倍数
    unsigned DR = D % 8;           // 处理8个float一组
    unsigned DD = D - DR;
    const float *l = a;
    const float *r = b;
    const float *e_l = l + DD;
    const float *e_r = r + DD;
    float unpack[4] __attribute__((aligned(16))) = {0, 0, 0, 0};

    sum = _mm_loadu_ps(unpack);
    if (DR)
    {
        SSE_DOT(e_l, e_r, sum, l0, r0);
    }

    for (unsigned i = 0; i < DD; i += 8, l += 8, r += 8)
    {
        SSE_DOT(l, r, sum, l0, r0);
        SSE_DOT(l + 4, r + 4, sum, l1, r1);
    }
    _mm_storeu_ps(unpack, sum);
    result = unpack[0] + unpack[1] + unpack[2] + unpack[3];
    return -result;
}

float distance_ip_sse_simple(const float *a, const float *b, size_t size)
{
    float sum = 0.0f;
    size_t i = 0;
    __m128 sum_vec = _mm_setzero_ps();
    float temp[4];

    for (; i + 4 <= size; i += 4)
    {
        sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(_mm_loadu_ps(a + i), _mm_loadu_ps(b + i)));
    }
    _mm_storeu_ps(temp, sum_vec);
    sum = temp[0] + temp[1] + temp[2] + temp[3];

    for (; i < size; i++)
    {
        sum += a[i] * b[i];
    }

    return -sum;
}