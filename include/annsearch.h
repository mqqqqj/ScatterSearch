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
#include <map>

#define AVX
// #define SSE

#define RECORD_DIST_COMPS
// #define COLLECT_VISITED_ID
// #define COLLECT_SEARCH_TREE
#define BREAKDOWN_ANALYSIS

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
    void SearchArraySimulationForPipeline(const float *query, unsigned query_id, int thread_id, int num_threads, int K, int L, boost::dynamic_bitset<> &flags, std::vector<Neighbor> &indices);
    void SearchArraySimulationForPipelineWithET(const float *query, unsigned query_id, int thread_id, int K, int L, boost::dynamic_bitset<> &flags, std::atomic<bool> &stop, std::vector<std::vector<Neighbor>> &retsets, std::vector<bool> &is_reach_20hop, std::atomic<float> &best_dist, std::atomic<int> &best_thread_id, int &local_ndc, int best_thread_ndc, std::vector<Neighbor> &indices);
    void MultiThreadSearch(const float *query, unsigned query_id, int K, int L, int num_threads, boost::dynamic_bitset<> &flags, std::vector<unsigned> &indices);
    void MultiThreadSearchArraySimulation(const float *query, unsigned query_id, int K, int L, int num_threads, boost::dynamic_bitset<> &flags, std::vector<unsigned> &indices);
    void MultiThreadSearchArraySimulationWithET(const float *query, unsigned query_id, int K, int L, int num_threads, boost::dynamic_bitset<> &flags, std::vector<unsigned> &indices);
    void MultiThreadSearchArraySimulationWithETTopM(const float *query, unsigned query_id, int K, int L, int num_threads, float percentage, boost::dynamic_bitset<> &flags, std::vector<unsigned> &indices);
    void SearchUntilBestThreadStop(const float *query, unsigned query_id, int K, int L, std::vector<std::vector<Neighbor>> &retsets, int *good_thread, bool *is_reach_100hop, std::atomic<bool> &best_thread_finish, std::atomic<float> &best_dist, boost::dynamic_bitset<> &flags, std::vector<Neighbor> &neighbors, int64_t &local_dist_comps);
    void EdgeWiseMultiThreadSearch(const float *query, unsigned query_id, int K, int L, int num_threads, boost::dynamic_bitset<> &flags, std::vector<unsigned> &indices);
    void ModifiedDeltaStepping(const float *query, unsigned query_id, int K, int L, int num_threads, boost::dynamic_bitset<> &flags, std::vector<unsigned> &indices);
    void MultiTurnSearch(const float *query, unsigned query_id, int K, int L, int num_turns, boost::dynamic_bitset<> &flags, std::vector<unsigned> &indices);

public:
    std::atomic<int> hop_count;
    float ub_ratio;
    int64_t dist_comps;
    int64_t max_dist_comps;
    double time_expand_;
    double time_merge_;
    double time_seq_;
    double time_total_;
#ifdef COLLECT_VISITED_ID
    std::vector<std::vector<unsigned>> visited_lists;
#endif
    std::vector<int> hop_find_first_knn;
    std::vector<int> hop_find_all_knn;
#ifdef COLLECT_SEARCH_TREE
    std::map<int, std::vector<std::pair<int, int>>> search_tree; // thread_id, arr:pair(current_node_id, father_node_id)
#endif
private:
    double get_time_mark();
    void select_entry_points(int pool_size, int P, const float *query, std::vector<unsigned> &selected_eps);

public:
    unsigned default_ep;
    size_t dimension;
    size_t base_num;
    float *base_data;
    std::vector<std::vector<unsigned>> groundtruth;
    std::vector<std::vector<unsigned>> graph;
    float (*distance_func)(const float *, const float *, size_t);
};

#endif
