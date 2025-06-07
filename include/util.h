#pragma once
#ifndef UTIL_H
#define UTIL_H

#include <vector>
#include <cstring>

#define LOG

struct TestResult
{
    unsigned L;
    float throughput;
    float latency;
    float recall;
    float p99_recall;
    float p95_recall;
};

struct Neighbor
{
    unsigned id;
    float distance;
    bool unexplored;
    Neighbor() = default;
    Neighbor(unsigned id, float distance, bool unexplored) : id{id}, distance{distance}, unexplored(unexplored) {}
    inline bool operator<(const Neighbor &other) const
    {
        return distance < other.distance;
    }
};

void load_fvecs(char *filename, float *&data, unsigned &num, unsigned &dim);
void load_fbin(char *filename, float *&data, unsigned &num, unsigned &dim);
void load_groundtruth(char *filename, std::vector<std::vector<unsigned>> &groundtruth);
void save_results(const std::vector<TestResult> &results, char *filename);

inline int InsertIntoPool(Neighbor *addr, unsigned K, Neighbor nn)
{
    // find the location to insert
    int left = 0, right = K - 1;
    if (addr[left].distance > nn.distance)
    {
        memmove((char *)&addr[left + 1], &addr[left], K * sizeof(Neighbor));
        addr[left] = nn;
        return left;
    }
    if (addr[right].distance < nn.distance)
    {
        addr[K] = nn;
        return K;
    }
    while (left < right - 1)
    {
        int mid = (left + right) / 2;
        if (addr[mid].distance > nn.distance)
            right = mid;
        else
            left = mid;
    }
    // check equal ID
    while (left > 0)
    {
        if (addr[left].distance < nn.distance)
            break;
        if (addr[left].id == nn.id)
            return K + 1;
        left--;
    }
    if (addr[left].id == nn.id || addr[right].id == nn.id)
        return K + 1;
    memmove((char *)&addr[right + 1], &addr[right], (K - right) * sizeof(Neighbor));
    addr[right] = nn;
    return right;
}

#endif