#ifndef DISTANCE_H
#define DISTANCE_H

#include <iostream>
float distance_l2sqr(const float *a, const float *b, size_t size);
float distance_l2sqr_avx(const float *a, const float *b, size_t size);
float distance_l2sqr_sse(const float *a, const float *b, size_t size);

float distance_ip(const float *a, const float *b, size_t size);
float distance_ip_avx(const float *a, const float *b, size_t size);
float distance_ip_avx_simple_unroll2(const float *a, const float *b, size_t size);
float distance_ip_sse(const float *a, const float *b, size_t size);
float distance_ip_avx_simple(const float *a, const float *b, size_t size);
float distance_ip_sse_simple(const float *a, const float *b, size_t size);
#endif