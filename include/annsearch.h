#ifndef ANNSEARCH_H
#define ANNSEARCH_H

#include <cstddef>
#include <vector>
#include <queue>
#include <iostream>
#include <boost/dynamic_bitset.hpp>
#include <distance.h>

enum Metric
{
    L2,
    INNER_PRODUCT
};

struct CompareByFirst
{
    constexpr bool operator()(std::pair<float, unsigned> const &a,
                              std::pair<float, unsigned> const &b) const noexcept
    {
        return a.first < b.first;
    }
};

class ANNSearch
{
public:
    ANNSearch(unsigned dim, unsigned num, float *base, Metric m);
    void LoadGraph(const char *filename);
    void Search(const float *query, unsigned query_id, int K, int L, boost::dynamic_bitset<> &flags, std::vector<unsigned> &indices);

private:
    unsigned default_ep;
    size_t dimension;
    size_t base_num;
    float *base_data;
    std::vector<std::vector<unsigned>> graph;
    float (*distance_func)(const float *, const float *, size_t);
};
#endif
