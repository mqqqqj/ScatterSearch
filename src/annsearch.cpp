#include <annsearch.h>
#include <fstream>
#include <cstdlib>

ANNSearch::ANNSearch(unsigned dim, unsigned num, float *base, Metric m)
{
    dimension = dim;
    base_num = num;
    base_data = base;
    if (m == L2)
        distance_func = distance_l2sqr;
    else if (m == INNER_PRODUCT)
        distance_func = distance_ip;
    else
    {
        std::cerr << "Error: Unknown metric type" << std::endl;
        exit(1);
    }
    default_ep = 0;
}

void ANNSearch::LoadGraph(const char *filename)
{
    std::ifstream in(filename, std::ios::binary);
    unsigned width = 0;
    in.read((char *)&width, sizeof(unsigned)); // not used
    in.read((char *)&default_ep, sizeof(unsigned));
    while (!in.eof())
    {
        unsigned k;
        in.read((char *)&k, sizeof(unsigned));
        if (in.eof())
            break;
        std::vector<unsigned> tmp(k);
        in.read((char *)tmp.data(), k * sizeof(unsigned));
        graph.push_back(tmp);
    }
}

void ANNSearch::Search(const float *query, unsigned query_id, int K, int L, boost::dynamic_bitset<> &flags, std::vector<unsigned> &indices)
{
    std::priority_queue<std::pair<float, unsigned>, std::vector<std::pair<float, unsigned>>, CompareByFirst> top_candidates;
    std::priority_queue<std::pair<float, unsigned>, std::vector<std::pair<float, unsigned>>, CompareByFirst> candidate_set;
    unsigned ep = rand() % base_num;
    int prefill_num = 1;
    float lowerBound;
    for (int i = 0; i < prefill_num; i++)
    {
        unsigned id = rand() % base_num;
        while (flags[id])
            id = rand() % base_num;
        lowerBound = distance_func(base_data + dimension * id, query, dimension); // note that distance = -inner product, smaller is better
        candidate_set.emplace(-lowerBound, ep);
        flags[id] = true;
    }
    while (!candidate_set.empty())
    {
        std::pair<float, unsigned> current_node_pair = candidate_set.top();
        float candidate_dist = -current_node_pair.first;
        _mm_prefetch(base_data + dimension * graph[current_node_pair.second][0], _MM_HINT_T0);
        bool flag_stop_search = candidate_dist > lowerBound;
        if (flag_stop_search)
        {
            break;
        }
        candidate_set.pop();
        unsigned current_node_id = current_node_pair.second;
        size_t n_neighbor = graph[current_node_id].size();
        for (unsigned m = 0; m < graph[current_node_id].size(); ++m)
        {
            unsigned candidate_id = graph[current_node_id][m];
            _mm_prefetch(base_data + dimension * graph[current_node_pair.second][m + 1], _MM_HINT_T0);
            if (flags[candidate_id])
            {
                n_neighbor--;
                continue;
            }
            flags[candidate_id] = true;
            float dist =
                distance_func(query, base_data + dimension * candidate_id, dimension);
            bool flag_consider_candidate = top_candidates.size() < L || lowerBound > dist;
            if (flag_consider_candidate)
            {
                candidate_set.emplace(-dist, candidate_id);
                top_candidates.emplace(dist, candidate_id);
                _mm_prefetch(base_data + dimension * candidate_set.top().second, _MM_HINT_T0);
                bool flag_remove_extra = top_candidates.size() > L;
                while (flag_remove_extra)
                {
                    unsigned id = top_candidates.top().second;
                    top_candidates.pop();
                    flag_remove_extra = top_candidates.size() > L;
                }
                if (!top_candidates.empty())
                    lowerBound = top_candidates.top().first;
            }
        }
    }
    while (top_candidates.size() > K)
    {
        top_candidates.pop();
    }
    // std::cout << "id: " << query_id << " " << "find " << top_candidates.size() << std::endl;
    for (int j = 0; j < K; ++j)
    {
        float dist = top_candidates.top().first;
        unsigned id = top_candidates.top().second;
        indices[K - 1 - j] = id;
        top_candidates.pop();
    }
}