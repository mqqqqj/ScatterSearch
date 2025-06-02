#ifndef ANNSEARCH_H
#define ANNSEARCH_H

#include <cstddef>
#include <vector>
#include <queue>
#include <iostream>
#include <atomic>
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

class ANNSearch
{
public:
    ANNSearch(unsigned dim, unsigned num, float *base, Metric m);
    void LoadGraph(const char *filename);
    void LoadGroundtruth(const char *filename);
    void Search(const float *query, unsigned query_id, int K, int L, boost::dynamic_bitset<> &flags, std::vector<unsigned> &indices);
    void MultiThreadSearch(const float *query, unsigned query_id, int K, int L, int num_threads, boost::dynamic_bitset<> &flags, std::vector<unsigned> &indices);
    void MultiThreadSearchArraySimulation(const float *query, unsigned query_id, int K, int L, int num_threads, boost::dynamic_bitset<> &flags, std::vector<unsigned> &indices);
    void MultiThreadSearchArraySimulationWithET(const float *query, unsigned query_id, int K, int L, int num_threads, boost::dynamic_bitset<> &flags, std::vector<unsigned> &indices);
    void SearchUntilBestThreadStop(const float *query, unsigned query_id, int K, int L, std::atomic<bool> &best_thread_finish, std::atomic<float> &best_dist, boost::dynamic_bitset<> &flags, std::vector<Neighbor> &neighbors);

private:
    double get_time_mark();
    inline int InsertIntoPool(Neighbor *addr, unsigned K, Neighbor nn);

private:
    unsigned default_ep;
    size_t dimension;
    size_t base_num;
    float *base_data;
    std::vector<std::vector<unsigned>> groundtruth;
    std::vector<std::vector<unsigned>> graph;
    float (*distance_func)(const float *, const float *, size_t);
};

#endif
