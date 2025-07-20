#ifndef ANNSEARCH_H
#define ANNSEARCH_H

#include <cstddef>
#include <vector>
#include <queue>
#include <iostream>
#include <atomic>
#include <boost/dynamic_bitset.hpp>
#include <distance.h>
#include <util.h>

#define AVX
// #define SSE

#define RECORD_DIST_COMPS

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
    ~ANNSearch();
    void LoadGraph(const char *filename);
    void LoadGroundtruth(const char *filename);
    void Search(const float *query, unsigned query_id, int K, int L, boost::dynamic_bitset<> &flags, std::vector<unsigned> &indices);
    void SearchArraySimulation(const float *query, unsigned query_id, int K, int L, boost::dynamic_bitset<> &flags, std::vector<unsigned> &indices);
    void SearchArraySimulationForPipeline(const float *query, unsigned query_id, int K, int L, boost::dynamic_bitset<> &flags, std::vector<Neighbor> &indices);
    void SearchArraySimulationForPipelineWithET(const float *query, unsigned query_id, int thread_id, int K, int L, boost::dynamic_bitset<> &flags, std::atomic<bool> &stop, std::vector<std::vector<Neighbor>> &retsets, std::vector<bool> &is_reach_20hop, std::atomic<float> &best_dist, std::atomic<int> &best_thread_id, std::vector<Neighbor> &indices);
    void MultiThreadSearch(const float *query, unsigned query_id, int K, int L, int num_threads, boost::dynamic_bitset<> &flags, std::vector<unsigned> &indices);
    void MultiThreadSearchArraySimulation(const float *query, unsigned query_id, int K, int L, int num_threads, boost::dynamic_bitset<> &flags, std::vector<unsigned> &indices);
    void MultiThreadSearchArraySimulationWithET(const float *query, unsigned query_id, int K, int L, int num_threads, boost::dynamic_bitset<> &flags, std::vector<unsigned> &indices);
    void MultiThreadSearchArraySimulationWithETTopM(const float *query, unsigned query_id, int K, int L, int num_threads, float percentage, boost::dynamic_bitset<> &flags, std::vector<unsigned> &indices);
    void SearchUntilBestThreadStop(const float *query, unsigned query_id, int K, int L, std::vector<std::vector<Neighbor>> &retsets, int *good_thread, bool *is_reach_100hop, std::atomic<bool> &best_thread_finish, std::atomic<float> &best_dist, boost::dynamic_bitset<> &flags, std::vector<Neighbor> &neighbors);

public:
    std::atomic<int> dist_comps;

private:
    double get_time_mark();
    void select_entry_points(int pool_size, int P, const float *query, std::vector<unsigned> &selected_eps);

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
