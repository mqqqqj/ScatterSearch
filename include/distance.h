#ifndef DISTANCE_H
#define DISTANCE_H
#include <immintrin.h>
#include <x86intrin.h>
#include <iostream>

float distance_l2sqr(const float *a, const float *b, size_t size);
float distance_ip(const float *a, const float *b, size_t size);

#endif