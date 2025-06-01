#pragma once
#ifndef UTIL_H
#define UTIL_H

#include <vector>
#include <cstring>

struct test_result
{
    unsigned L;
    float throughput;
    float latency;
    float recall;
    float std_recall;
    float p99_recall;
    float p95_recall;
};

void load_fvecs(char *filename, float *&data, unsigned &num, unsigned &dim);
void load_fbin(char *filename, float *&data, unsigned &num, unsigned &dim);
void load_groundtruth(char *filename, std::vector<std::vector<unsigned>> &groundtruth);
void save_results(const std::vector<test_result> &results, char *filename);

#endif