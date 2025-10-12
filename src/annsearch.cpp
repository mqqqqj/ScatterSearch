#include <annsearch.h>
#include <fstream>
#include <cstdlib>
#include <sys/time.h>
#include <omp.h>
#include <xmmintrin.h>
#include <cmath>

ANNSearch::ANNSearch(unsigned dim, unsigned num, float *base, Metric m)
{
    dist_comps = 0;
    max_dist_comps = 0;
    hop_count = 0;
    ub_ratio = 0;
    time_expand_ = 0;
    time_merge_ = 0;
    time_seq_ = 0;
    time_total_ = 0;
    dimension = dim;
    base_num = num;
    base_data = base;
    if (m == L2)
    {
#ifdef AVX
        distance_func = distance_l2sqr_avx;
        std::cout << "Dist function: L2SQR AVX" << std::endl;
#elif defined SSE
        distance_func = distance_l2sqr_sse;
        std::cout << "Dist function: L2SQR SSE" << std::endl;
#else
        distance_func = distance_l2sqr;
        std::cout << "Dist function: L2SQR naive" << std::endl;
#endif
    }
    else if (m == INNER_PRODUCT)
    {
#ifdef AVX
        distance_func = distance_ip_avx;
        // distance_func = distance_ip_avx_simple;
        std::cout << "Dist function: IP AVX" << std::endl;
#elif defined SSE
        distance_func = distance_ip_sse;
        // distance_func = distance_ip_sse_simple;
        std::cout << "Dist function: IP SSE" << std::endl;
#else
        distance_func = distance_ip;
        std::cout << "Dist function: IP naive" << std::endl;
#endif
    }
    else
    {
        std::cerr << "Error: Unknown metric type" << std::endl;
        exit(1);
    }
    default_ep = 0;
}

ANNSearch::~ANNSearch()
{
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

void ANNSearch::LoadGroundtruth(const char *filename)
{
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open())
    {
        std::cout << "open file error: " << filename << std::endl;
        exit(-1);
    }
    unsigned GK, nq;
    in.read((char *)&nq, sizeof(unsigned));
    in.read((char *)&GK, sizeof(unsigned));
    std::cout << "nq: " << nq << ", GK: " << GK << std::endl;
    for (unsigned i = 0; i < nq; i++)
    {
        std::vector<unsigned> result(GK);
        in.read((char *)result.data(), GK * sizeof(unsigned));
        groundtruth.push_back(result);
    }
    in.close();
}

double ANNSearch::get_time_mark()
{
    timeval t;
    gettimeofday(&t, nullptr);
    return t.tv_sec + t.tv_usec * 0.000001;
}

void ANNSearch::select_entry_points(int pool_size, int P, const float *query, std::vector<unsigned> &selected_eps)
{
    std::vector<unsigned> ep_pool(pool_size);
    for (int i = 0; i < pool_size; i++)
    {
        ep_pool[i] = rand() % base_num;
    }
    // --- Stage 1: Parallel Direction Vector Calculation ---
    // This vector will store the normalized direction vectors (p_i - q)
    float direction_vectors[pool_size][dimension];

#pragma omp parallel for num_threads(P)
    for (size_t i = 0; i < pool_size; ++i)
    {
        float *direction = direction_vectors[i];
        // Calculate direction vector: p_i - q
        for (int d = 0; d < dimension; ++d)
        {
            direction[d] = base_data[dimension * ep_pool[i] + d] - query[d];
        }
        // const float *base = &base_data[dimension * ep_pool[i]]; // 基向量起始地址
        // const float *q = query;                                 // 查询向量

        // int d = 0;
        // // 处理能被16整除的部分 (AVX512一次处理16个float)
        // const int simd_step = 16;
        // const int simd_count = dimension / simd_step;

        // for (int j = 0; j < simd_count; ++j, d += simd_step)
        // {
        //     // 加载基向量的16个元素
        //     __m512 base_vec = _mm512_loadu_ps(&base[d]);
        //     // 加载查询向量的16个元素
        //     __m512 query_vec = _mm512_loadu_ps(&q[d]);
        //     // 计算: base_vec - query_vec
        //     __m512 diff_vec = _mm512_sub_ps(base_vec, query_vec);
        //     // 存储结果到direction
        //     _mm512_storeu_ps(&direction[d], diff_vec);
        // }

        // // 处理剩余的元素 (不足16个的部分)
        // for (; d < dimension; ++d)
        // {
        //     direction[d] = base[d] - q[d];
        // }
        // Normalize the direction vector
        float norm_sq = -distance_func(direction, direction, dimension);
        if (norm_sq > 1e-9f)
        {
            float inv_norm = 1.0f / std::sqrt(norm_sq);
            for (int d = 0; d < dimension; ++d)
            {
                direction[d] *= inv_norm;
            }
        }
    }

    // --- Stage 2: Greedy Diversity Selection ---
    std::vector<bool> is_selected(pool_size, false);

    // A. Select the first point
    selected_eps.push_back(ep_pool[0]);
    is_selected[0] = true;

    // B. Iteratively select the remaining P-1 points
    for (int i = 1; i < P; ++i)
    {
        int best_candidate_idx = -1;
        float min_max_similarity = 2.0f; // Max similarity is 1.0, so 2.0 is a safe initial value

        // Find the candidate that is least similar to any already selected point
        for (size_t j = 0; j < pool_size; ++j)
        {
            if (is_selected[j])
                continue;

            float max_sim_with_selected = -2.0f; // Min similarity is -1.0
            // Find the similarity to the "closest" (in angle) already selected point
            for (unsigned sel_ep : selected_eps)
            {
                // We need to find the original index of sel_ep to get its direction vector
                auto it = std::find(ep_pool.begin(), ep_pool.end(), sel_ep);
                int sel_idx = std::distance(ep_pool.begin(), it);

                float sim = -distance_func(direction_vectors[j], direction_vectors[sel_idx], dimension);
                if (sim > max_sim_with_selected)
                {
                    max_sim_with_selected = sim;
                }
            }

            // We want to minimize the maximum similarity, which is equivalent to
            // maximizing the minimum angular distance.
            if (max_sim_with_selected < min_max_similarity)
            {
                min_max_similarity = max_sim_with_selected;
                best_candidate_idx = j;
            }
        }

        if (best_candidate_idx != -1)
        {
            selected_eps.push_back(ep_pool[best_candidate_idx]);
            is_selected[best_candidate_idx] = true;
        }
        else
        {
            // No more candidates to select, break early
            break;
        }
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
    for (int j = 0; j < K; ++j)
    {
        float dist = top_candidates.top().first;
        unsigned id = top_candidates.top().second;
        indices[K - 1 - j] = id;
        top_candidates.pop();
    }
}

void ANNSearch::SearchArraySimulation(const float *query, unsigned query_id, int K, int L, boost::dynamic_bitset<> &flags, std::vector<unsigned> &indices)
{
    // int ep = rand() % base_num;
    int ep = default_ep;
    std::vector<unsigned> init_ids(L);
    unsigned tmp_l = 0;
    for (; tmp_l < L && tmp_l < graph[ep].size(); tmp_l++)
    {
        init_ids[tmp_l] = graph[ep][tmp_l];
        flags[init_ids[tmp_l]] = true;
    }
    std::vector<Neighbor> retset(L + 1);
    while (tmp_l < K)
    {
        unsigned id = rand() % base_num;
        if (flags[id])
        {
            // id++;
            continue;
        }
        flags[id] = true;
        init_ids[tmp_l] = id;
        tmp_l++;
        // id++;
    }
    for (unsigned j = 0; j < tmp_l; j++)
    {
        unsigned id = init_ids[j];
        _mm_prefetch(base_data + dimension * id, _MM_HINT_T0);
        float dist = distance_func(base_data + dimension * id, query, dimension);
        dist_comps++;
        retset[j] = Neighbor(id, dist, true);
    }
    std::sort(retset.begin(), retset.begin() + tmp_l); // sort the retset by distance in ascending order
    int k = 0;
    int hop = 0;
    while (k < (int)L)
    {
        int nk = L;
        if (retset[k].unexplored)
        {
            retset[k].unexplored = false;
            unsigned n = retset[k].id;
            _mm_prefetch(graph[n].data(), _MM_HINT_T0);
            for (unsigned m = 0; m < graph[n].size(); ++m)
            {
                unsigned id = graph[n][m];
                if (flags[id])
                    continue;
                flags[id] = true;
                if (m + 1 < graph[n].size())
                {
                    _mm_prefetch(base_data + dimension * graph[n][m + 1], _MM_HINT_T0);
                }
                float dist = distance_func(query, base_data + dimension * id, dimension);
                dist_comps++;
                if (dist >= retset[tmp_l - 1].distance)
                    continue;
                Neighbor nn(id, dist, true);
                int r = InsertIntoPool(retset.data(), tmp_l, nn);
                if (tmp_l < L)
                    tmp_l++;
                if (r < nk)
                    nk = r;
            }
            hop++;
        }
        if (nk <= k)
            k = nk;
        else
            ++k;
    }
    for (size_t i = 0; i < K; i++)
    {
        indices[i] = retset[i].id;
    }
    flags.reset();
}

void ANNSearch::SearchArraySimulationForPipeline(const float *query, unsigned query_id, int thread_id, int num_threads, int K, int L, boost::dynamic_bitset<> &flags, std::vector<Neighbor> &indices)
{
    // int ep = rand() % base_num;
    std::vector<unsigned> init_ids(L);
    unsigned tmp_l = 0;
    for (int j = 0; j < graph[default_ep].size(); j++)
    {
        if (j % num_threads == thread_id)
        {
            init_ids[tmp_l] = graph[default_ep][j];
            flags[init_ids[tmp_l]] = true;
            tmp_l++;
        }
    }
    std::vector<Neighbor> retset(L + 1);
    for (unsigned j = 0; j < tmp_l; j++)
    {
        unsigned id = init_ids[j];
        _mm_prefetch(base_data + dimension * id, _MM_HINT_T0);
        float dist = distance_func(base_data + dimension * id, query, dimension);
        retset[j] = Neighbor(id, dist, true);
    }
    std::sort(retset.begin(), retset.begin() + tmp_l); // sort the retset by distance in ascending order
    int k = 0;
    while (k < (int)L)
    {
        int nk = L;
        if (retset[k].unexplored)
        {
            retset[k].unexplored = false;
            unsigned n = retset[k].id;
            _mm_prefetch(graph[n].data(), _MM_HINT_T0);
            for (unsigned m = 0; m < graph[n].size(); ++m)
            {
                unsigned id = graph[n][m];
                if (m + 1 < graph[n].size())
                {
                    _mm_prefetch(base_data + dimension * graph[n][m + 1], _MM_HINT_T0);
                }
                if (flags[id])
                    continue;
                flags[id] = true;
                float dist = distance_func(query, base_data + dimension * id, dimension);
                if (dist >= retset[tmp_l - 1].distance)
                    continue;
                Neighbor nn(id, dist, true);
                int r = InsertIntoPool(retset.data(), tmp_l, nn);
                if (tmp_l < L)
                    tmp_l++;
                if (r < nk)
                    nk = r;
            }
        }
        if (nk <= k)
            k = nk;
        else
            ++k;
    }
    indices.resize(K);
    for (size_t i = 0; i < K; i++)
    {
        indices[i] = retset[i];
    }
}

void ANNSearch::SearchArraySimulationForPipelineWithET(const float *query, unsigned query_id, int thread_id, int K, int L,
                                                       boost::dynamic_bitset<> &flags, std::atomic<bool> &stop,
                                                       std::vector<std::vector<Neighbor>> &retsets, std::vector<bool> &is_reach_100hop,
                                                       std::atomic<float> &best_dist, std::atomic<int> &best_thread_id,
                                                       int &local_ndc, int best_thread_ndc, std::vector<Neighbor> &indices)
{
    // std::vector<Neighbor> retset(L + 1);
    std::vector<Neighbor> &retset = retsets[thread_id];
    unsigned tmp_l = 0;
    int num_threads = retsets.size();
    // for (int i = 0; i < num_threads; i++)
    // {
    //     if (is_reach_20hop[i])
    //     {
    //         int idx = rand() % 20;
    //         retset[tmp_l] = retsets[i][idx];
    //         tmp_l++;
    //     }
    // }
    for (int j = 0; j < graph[default_ep].size(); j++)
    {
        if (j % num_threads == thread_id)
        {
            int id = graph[default_ep][j];
            float dist = distance_func(base_data + dimension * graph[default_ep][j], query, dimension);
            local_ndc++;
            retset[tmp_l] = Neighbor(id, dist, true);
            flags[id] = true;
            tmp_l++;
        }
    }
    while (tmp_l < K)
    {
        int id = rand() % base_num;
        while (flags[id])
            id = rand() % base_num;
        float dist = distance_func(base_data + dimension * id, query, dimension);
        local_ndc++;
        retset[tmp_l] = Neighbor(id, dist, true);
        tmp_l++;
    }
    std::sort(retset.begin(), retset.begin() + tmp_l); // sort the retset by distance in ascending order
    int k = 0;
    int hop = 0;
    bool need_identify = true;
    while (k < (int)L)
    {
        if (best_thread_ndc != 0 && local_ndc >= best_thread_ndc)
            break;
        if (hop == 50)
        {
            is_reach_100hop[thread_id] = true;
            if (best_dist > retset[K - 1].distance)
            {
                best_dist = retset[K - 1].distance;
                best_thread_id = thread_id;
            }
        }
        if (need_identify == true && hop > 50 && best_dist < retset[0].distance)
        {
            need_identify = false;
            break;
        }
        int nk = L;
        if (retset[k].unexplored)
        {
            retset[k].unexplored = false;
            unsigned n = retset[k].id;
            _mm_prefetch(graph[n].data(), _MM_HINT_T0);
            for (unsigned m = 0; m < graph[n].size(); ++m)
            {
                unsigned id = graph[n][m];
                if (m + 1 < graph[n].size())
                {
                    _mm_prefetch(base_data + dimension * graph[n][m + 1], _MM_HINT_T0);
                }
                if (flags[id])
                    continue;
                flags[id] = true;
                float dist = distance_func(query, base_data + dimension * id, dimension);
                local_ndc++;
                if (dist >= retset[tmp_l - 1].distance)
                    continue;
                Neighbor nn(id, dist, true);
                int r = InsertIntoPool(retset.data(), tmp_l, nn);
                if (tmp_l < L)
                    tmp_l++;
                if (r < nk)
                    nk = r;
            }
            hop++;
        }
        if (nk <= k)
            k = nk;
        else
            ++k;
    }
    indices.resize(K);
    for (size_t i = 0; i < K; i++)
    {
        indices[i] = retset[i];
    }
}

void ANNSearch::MultiThreadSearch(const float *query, unsigned query_id, int K, int L, int num_threads, boost::dynamic_bitset<> &flags, std::vector<unsigned> &indices)
{
    std::vector<std::vector<Neighbor>> retsets(num_threads);
    int ndc_thread[num_threads];
    int finish_num = 0;

#pragma omp parallel num_threads(num_threads)
    {
        int i = omp_get_thread_num();
        std::priority_queue<std::pair<float, unsigned>, std::vector<std::pair<float, unsigned>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<float, unsigned>, std::vector<std::pair<float, unsigned>>, CompareByFirst> candidate_set;
        int ep = rand() % base_num;
        int hop = 0;
        while (flags[ep])
            ep = rand() % base_num;
        float lowerBound = distance_func(base_data + dimension * ep, query, dimension); // note that distance = -inner product, smaller is better
        ndc_thread[i] = 1;
        candidate_set.emplace(-lowerBound, ep);
        flags[ep] = true;
        while (top_candidates.size() < K)
        {
            int id = rand() % base_num;
            while (flags[id])
                id = rand() % base_num;
            float dist = distance_func(base_data + dimension * id, query, dimension); // note that distance = -inner product, smaller is better
            ndc_thread[i] += 1;
            candidate_set.emplace(-dist, id);
            top_candidates.emplace(dist, id);
            flags[id] = true;
        }
        while (!candidate_set.empty())
        {
            // if (finish_num >= num_threads / 2)
            //     break;
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
            for (unsigned m = 0; m < graph[current_node_id].size(); ++m)
            {
                unsigned candidate_id = graph[current_node_id][m];
                _mm_prefetch(base_data + dimension * graph[current_node_pair.second][m + 1], _MM_HINT_T0);
                if (flags[candidate_id])
                    continue;
                flags[candidate_id] = true;
                float dist =
                    distance_func(query, base_data + dimension * candidate_id, dimension);
                ndc_thread[i] += 1;
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
            hop++;
        }
        if (top_candidates.size() < K)
            exit(1);
        while (top_candidates.size() > K)
        {
            top_candidates.pop();
        }
        retsets[i].resize(K);
        int idx = K - 1;
        while (top_candidates.size())
        {
            auto node = top_candidates.top();
            retsets[i][idx] = Neighbor(node.second, node.first, false);
            idx--;
            top_candidates.pop();
        }
        finish_num++;
    }
    int master = 0;
    for (int i = master + 1; i < num_threads; i++)
    {
        for (unsigned j = 0; j < K; j++)
        {
            InsertIntoPool(retsets[master].data(), K, retsets[i][j]);
        }
    }
    for (size_t i = 0; i < K; i++)
    {
        indices[i] = retsets[master][i].id;
    }
}
void ANNSearch::MultiThreadSearchArraySimulation(const float *query, unsigned query_id, int K, int L, int num_threads, boost::dynamic_bitset<> &flags, std::vector<unsigned> &indices)
{
#ifdef BREAKDOWN_ANALYSIS
    time_seq_ -= get_time_mark();
#endif
    std::vector<std::vector<Neighbor>> retsets(num_threads);
    int64_t dist_comps_per_thread[num_threads];
#ifdef BREAKDOWN_ANALYSIS
    time_seq_ += get_time_mark();
#endif
#ifdef BREAKDOWN_ANALYSIS
    time_expand_ -= get_time_mark();
#endif
#pragma omp parallel num_threads(num_threads)
    {
        int i = omp_get_thread_num();
        int64_t local_dist_comps = 0;
        std::vector<unsigned> init_ids(L);
        unsigned tmp_l = 0;
        retsets[i].resize(L + 1);
        // int ep = rand() % base_num;
        for (int j = 0; j < graph[default_ep].size(); j++)
        {
            if (j % num_threads == i)
            {
                init_ids[tmp_l] = graph[default_ep][j];
                flags[init_ids[tmp_l]] = true;
                tmp_l++;
            }
        }
        // for (int j = 0; j < graph[ep].size(); j++)
        // {
        //     init_ids[tmp_l] = graph[ep][j];
        //     flags[init_ids[tmp_l]] = true;
        //     tmp_l++;
        // }
        for (unsigned j = 0; j < tmp_l; j++)
        {
            unsigned id = init_ids[j];
            _mm_prefetch(base_data + dimension * id, _MM_HINT_T0);
            float dist = distance_func(base_data + dimension * id, query, dimension);
#ifdef COLLECT_VISITED_ID
            visited_lists[i].push_back(id);
#endif
#ifdef RECORD_DIST_COMPS
            local_dist_comps++;
#endif
            retsets[i][j] = Neighbor(id, dist, true);
        }
        std::sort(retsets[i].begin(), retsets[i].begin() + tmp_l); // sort the retset by distance in ascending order
        int k = 0;
        int hop = 0;
        while (k < (int)L)
        {
            int nk = L;
            int min_r = L;
            if (retsets[i][k].unexplored)
            {
                retsets[i][k].unexplored = false;
                unsigned n = retsets[i][k].id;
                _mm_prefetch(graph[n].data(), _MM_HINT_T0);
                for (unsigned m = 0; m < graph[n].size(); ++m)
                {
                    unsigned id = graph[n][m];
                    if (m + 1 < graph[n].size())
                    {
                        _mm_prefetch(base_data + dimension * graph[n][m + 1], _MM_HINT_T0);
                    }
                    if (flags[id])
                        continue;
                    flags[id] = true;
                    float dist = distance_func(query, base_data + dimension * id, dimension);
#ifdef COLLECT_VISITED_ID
                    visited_lists[i].push_back(id);
#endif
#ifdef RECORD_DIST_COMPS
                    local_dist_comps++;
#endif
                    if (dist >= retsets[i][tmp_l - 1].distance)
                        continue;
                    Neighbor nn(id, dist, true);
                    int r = InsertIntoPool(retsets[i].data(), tmp_l, nn);
                    if (tmp_l < L)
                        tmp_l++;
                    if (r < nk)
                        nk = r;
                }
                hop++;
            }
            if (nk <= k)
                k = nk;
            else
                ++k;
        }
        hop_count += hop;
        dist_comps_per_thread[i] = local_dist_comps;
    }
#ifdef BREAKDOWN_ANALYSIS
    time_expand_ += get_time_mark();
    time_merge_ -= get_time_mark();
#endif
    for (int i = 1; i < num_threads; i++)
    {
        for (size_t j = 0; j < K; j++)
        {
            int pos = InsertIntoPool(retsets[0].data(), K, retsets[i][j]);
            if (pos == K)
                break;
        }
    }
#ifdef BREAKDOWN_ANALYSIS
    time_merge_ += get_time_mark();
    time_seq_ -= get_time_mark();
#endif
    for (size_t i = 0; i < K; i++)
    {
        indices[i] = retsets[0][i].id;
    }
    flags.reset();
#ifdef RECORD_DIST_COMPS
    float mincomps = 1000000, maxcomps = 0;
    for (int i = 0; i < num_threads; i++)
    {
        dist_comps += dist_comps_per_thread[i];
        if (dist_comps_per_thread[i] < mincomps)
            mincomps = dist_comps_per_thread[i];
        if (dist_comps_per_thread[i] > maxcomps)
            maxcomps = dist_comps_per_thread[i];
    }
    max_dist_comps += maxcomps;
    ub_ratio += maxcomps / mincomps;
#endif
#ifdef BREAKDOWN_ANALYSIS
    time_seq_ += get_time_mark();
#endif
}

void ANNSearch::MultiThreadSearchArraySimulationWithET(const float *query, unsigned query_id, int K, int L, int num_threads, boost::dynamic_bitset<> &flags, std::vector<unsigned> &indices)
{
#ifdef BREAKDOWN_ANALYSIS
    time_seq_ -= get_time_mark();
#endif

    std::vector<std::vector<Neighbor>> retsets(num_threads);
    std::vector<std::vector<Neighbor>> new_retsets(num_threads);
    int best_thread_id = -1;
    std::atomic<int> decide_num;
    decide_num = 0;
    std::atomic<float> best_dist;
    best_dist = 1000;
    std::atomic<bool> best_thread_finish;
    best_thread_finish = false;
    int good_thread[num_threads];
    memset(good_thread, 0, sizeof(int) * num_threads);
    bool is_reach_100hop[num_threads];
    memset(is_reach_100hop, 0, sizeof(bool) * num_threads);
    int64_t dist_comps_per_thread[num_threads];
    // std::vector<unsigned> ep_list;
    // select_entry_points(30, num_threads, query, ep_list);
    int election_hop = 50;
    if (L < 50)
    {
        election_hop = L;
    }
#ifdef BREAKDOWN_ANALYSIS
    time_seq_ += get_time_mark();
    time_expand_ -= get_time_mark();
#endif
#pragma omp parallel num_threads(num_threads)
    {
        int i = omp_get_thread_num();
        int hop = 0;
        int64_t local_dist_comps = 0;
        std::vector<unsigned> init_ids(L);
        bool need_identify = true;
        unsigned tmp_l = 0;
        retsets[i].resize(L + 1);
        for (int j = 0; j < graph[default_ep].size(); j++)
        {
            if (j % num_threads == i)
            {
                init_ids[tmp_l] = graph[default_ep][j];
                flags[init_ids[tmp_l]] = true;
                tmp_l++;
            }
        }
        // int ep = ep_list[i];
        // while (tmp_l < graph[ep].size())
        // {
        //     init_ids[tmp_l] = graph[ep][tmp_l];
        //     flags[init_ids[tmp_l]] = true;
        //     tmp_l++;
        // }
        for (unsigned j = 0; j < tmp_l; j++)
        {
            unsigned id = init_ids[j];
            float dist = distance_func(base_data + dimension * id, query, dimension);
#ifdef COLLECT_SEARCH_TREE
            search_tree[i].push_back(std::make_pair(id, default_ep));
#endif
#ifdef RECORD_DIST_COMPS
            local_dist_comps++;
#endif
            retsets[i][j] = Neighbor(id, dist, true);
        }
        std::sort(retsets[i].begin(), retsets[i].begin() + tmp_l); // sort the retset by distance in ascending order
        int k = 0;
        while (k < (int)L)
        {
            int nk = L;
            if (best_thread_finish)
                break;
            if (hop == election_hop)
            {
                decide_num++;
                is_reach_100hop[i] = true;
                if (best_dist > retsets[i][K - 1].distance)
                {
                    best_dist = retsets[i][K - 1].distance;
                    best_thread_id = i;
                }
            }
            if (need_identify && decide_num == num_threads)
            {
                need_identify = false;
                if (best_dist < retsets[i][0].distance)
                {
                    good_thread[i] = -1;
                    break;
                }
                else if (best_dist > retsets[i][0].distance)
                {
                    good_thread[i] = 1;
                }
            }
            if (retsets[i][k].unexplored)
            {
                retsets[i][k].unexplored = false;
                unsigned n = retsets[i][k].id;
                for (unsigned m = 0; m < graph[n].size(); ++m)
                {
                    unsigned id = graph[n][m];
                    if (flags[id])
                        continue;
                    flags[id] = true;
                    float dist = distance_func(query, base_data + dimension * id, dimension);
#ifdef COLLECT_SEARCH_TREE
                    search_tree[i].push_back(std::make_pair(id, n));
#endif
#ifdef RECORD_DIST_COMPS
                    local_dist_comps++;
#endif
                    if (dist >= retsets[i][tmp_l - 1].distance)
                        continue;
                    Neighbor nn(id, dist, true);
                    int r = InsertIntoPool(retsets[i].data(), tmp_l, nn);
                    if (tmp_l < L)
                        tmp_l++;
                    if (r < nk)
                        nk = r;
                }
                hop++;
            }
            if (nk <= k)
                k = nk;
            else
                ++k;
        }
        // if (best_thread_id == i)
        // {
        //     best_thread_finish = true;
        // }
        if (good_thread[i] == 1)
        {
            best_thread_finish = true;
        }
        else
        {
            if (best_thread_finish == false)
            {
                std::vector<Neighbor> new_retset(L + 1);
                SearchUntilBestThreadStop(query, query_id, K, L, retsets, good_thread, is_reach_100hop, best_thread_finish, best_dist, flags, retsets[i], tmp_l, local_dist_comps);
                new_retsets[i] = new_retset;
            }
        }
        dist_comps_per_thread[i] = local_dist_comps;
    }
#ifdef BREAKDOWN_ANALYSIS
    time_expand_ += get_time_mark();
    time_merge_ -= get_time_mark();
#endif
    for (int i = 0; i < num_threads; i++)
    {
        if (i != best_thread_id)
            for (size_t j = 0; j < K; j++)
            {
                int pos = InsertIntoPool(retsets[best_thread_id].data(), K, retsets[i][j]);
                if (pos == K)
                    break;
            }
    }
    // for (size_t i = 0; i < new_retsets.size(); i++)
    // {
    //     if (new_retsets[i].size())
    //         for (int j = 0; j < K; j++)
    //         {
    //             int pos = InsertIntoPool(retsets[best_thread_id].data(), K, new_retsets[i][j]);
    //             if (pos == K)
    //                 break;
    //         }
    // }
#ifdef BREAKDOWN_ANALYSIS
    time_merge_ += get_time_mark();
#endif
// for (int i = 0; i < num_threads; i++)
// {
//     std::cout << i << "," << good_thread[i] << std::endl;
// }
// std::cout << best_thread_id << std::endl;
#ifdef BREAKDOWN_ANALYSIS
    time_seq_ -= get_time_mark();
#endif
    for (size_t i = 0; i < K; i++)
    {
        indices[i] = retsets[best_thread_id][i].id;
    }
    flags.reset();
#ifdef RECORD_DIST_COMPS
    float mincomps = 1000000, maxcomps = 0;
    for (int i = 0; i < num_threads; i++)
    {
        dist_comps += dist_comps_per_thread[i];
        if (dist_comps_per_thread[i] < mincomps)
            mincomps = dist_comps_per_thread[i];
        if (dist_comps_per_thread[i] > maxcomps)
            maxcomps = dist_comps_per_thread[i];
    }
    max_dist_comps += maxcomps;
    ub_ratio += maxcomps / mincomps;
#endif
#ifdef BREAKDOWN_ANALYSIS
    time_seq_ += get_time_mark();
#endif
}

void ANNSearch::MultiThreadSearchArraySimulationWithETTopM(const float *query, unsigned query_id, int K, int L, int num_threads, float percentage, boost::dynamic_bitset<> &flags, std::vector<unsigned> &indices)
{
    std::vector<std::vector<Neighbor>> retsets(num_threads);
    std::vector<std::vector<Neighbor>> new_retsets(num_threads);
    std::atomic<int> decide_num;
    decide_num = 0;
    std::atomic<bool> should_finish;
    should_finish = false;
    int good_thread_num = std::ceil(num_threads * percentage);
    std::atomic<int> good_thread_finish_num;
    good_thread_finish_num = 0;
    bool is_reach_100hop[num_threads];
    memset(is_reach_100hop, 0, sizeof(bool) * num_threads);
    std::vector<std::pair<int, float>> tid_dist_pair(num_threads);
    unsigned retset_len[num_threads];
    std::atomic<int> best_thread_id;
    best_thread_id = -1;
    std::vector<unsigned> ep_list;
    std::atomic<int> finish_num;
    finish_num = 0;
    // select_entry_points(30, num_threads, query, ep_list);
#pragma omp parallel num_threads(num_threads)
    {
        int i = omp_get_thread_num();
        int ep = rand() % base_num;
        // int ep = ep_list[i];
        int hop = 0;
        std::vector<unsigned> init_ids(L);
        bool need_rank = true;
        int my_rank = 100;
        std::vector<std::pair<int, float>> tid_dist_pair_copy;
        unsigned &tmp_l = retset_len[i];
        tmp_l = 0;
        while (tmp_l < K)
        {
            int id = rand() % base_num;
            if (flags[id])
                continue;
            init_ids[tmp_l] = id;
            tmp_l++;
        }
        retsets[i].resize(L + 1);
        for (unsigned j = 0; j < tmp_l; j++)
        {
            unsigned id = init_ids[j];
            float dist = distance_func(base_data + dimension * id, query, dimension);
            retsets[i][j] = Neighbor(id, dist, true);
        }
        std::sort(retsets[i].begin(), retsets[i].begin() + tmp_l); // sort the retset by distance in ascending order
        int k = 0;
        while (k < (int)L)
        {
            int nk = L;
            if (should_finish)
                break;
            if (hop == 20)
            {
                tid_dist_pair[i] = std::pair<int, float>(i, retsets[i][0].distance);
                is_reach_100hop[i] = true;
                decide_num++;
            }
            if (need_rank && decide_num == num_threads)
            {
                need_rank = false;
                tid_dist_pair_copy = tid_dist_pair; // 拷贝内容
                std::sort(tid_dist_pair_copy.begin(), tid_dist_pair_copy.end(), [](const std::pair<int, float> &a, const std::pair<int, float> &b)
                          { return a.second < b.second; });
                if (best_thread_id != -1 && best_thread_id != tid_dist_pair_copy[0].first)
                {
                    std::cout << best_thread_id << " not equal to " << tid_dist_pair_copy[0].first << std::endl;
                }
                best_thread_id = tid_dist_pair_copy[0].first;
                for (int j = 0; j < num_threads; j++)
                {
                    if (tid_dist_pair_copy[j].first == i)
                    {
                        my_rank = j;
                        break;
                    }
                }
                if (tid_dist_pair_copy[0].second < 1.4 * retsets[i][0].distance)
                {
                    break;
                }
            }
            if (retsets[i][k].unexplored)
            {
                retsets[i][k].unexplored = false;
                unsigned n = retsets[i][k].id;
                for (unsigned m = 0; m < graph[n].size(); ++m)
                {
                    unsigned id = graph[n][m];
                    if (flags[id])
                        continue;
                    flags[id] = true;
                    float dist = distance_func(query, base_data + dimension * id, dimension);
                    if (dist >= retsets[i][tmp_l - 1].distance)
                        continue;
                    Neighbor nn(id, dist, true);
                    int r = InsertIntoPool(retsets[i].data(), tmp_l, nn);
                    if (tmp_l < L)
                        tmp_l++;
                    if (r < nk)
                        nk = r;
                }
                hop++;
            }
            if (nk <= k)
                k = nk;
            else
                ++k;
        }
        finish_num++;
        // if (my_rank == 0)
        // {
        //     if (best_thread_id != i)
        //         std::cout << "2" << std::endl;
        //     should_finish = true;
        // }
        if (best_thread_id == i)
        {
            // std::cout << "finish num: " << finish_num << std::endl;
            should_finish = true;
            if (decide_num < num_threads)
            {
                std::cout << "1" << std::endl;
            }
        }
    }
    int master_id = -1;
    for (int i = 0; i < num_threads; i++)
    {
        if (retset_len[i] >= K)
        {
            master_id = i;
            break;
        }
    }
    for (int i = 0; i < num_threads; i++)
    {
        if (i != master_id)
            for (size_t j = 0; j < std::min((unsigned)K, retset_len[i]); j++)
            {
                InsertIntoPool(retsets[master_id].data(), K, retsets[i][j]);
            }
    }
    for (size_t i = 0; i < K; i++)
    {
        indices[i] = retsets[master_id][i].id;
    }
}

void ANNSearch::SearchUntilBestThreadStop(const float *query, unsigned query_id, int K, int L, std::vector<std::vector<Neighbor>> &main_retsets, int *good_thread, bool *is_reach_100hop, std::atomic<bool> &best_thread_finish, std::atomic<float> &best_dist, boost::dynamic_bitset<> &flags, std::vector<Neighbor> &retset, unsigned &tmp_l, int64_t &local_dist_comps)
{
    std::vector<int> retset_size;
    int good_id_num = 0;
    int thread_num = main_retsets.size();
    int i = rand() % thread_num;
    if (is_reach_100hop[i] && good_thread[i] == 1)
    {
        for (int idx = L - 1; idx; idx--)
        {
            if (main_retsets[i][idx].unexplored == true)
            {
                InsertIntoPool(retset.data(), tmp_l, main_retsets[i][idx]);
                good_id_num++;
            }
            if (good_id_num > 10)
            {
                break;
            }
        }
    }
    int k = 0;
    int hop = 0;
    while ((k < (int)L))
    {
        int nk = L;
        if (best_thread_finish)
            break;
        if (retset[k].unexplored)
        {
            retset[k].unexplored = false;
            unsigned n = retset[k].id;
            for (unsigned m = 0; m < graph[n].size(); ++m)
            {
                unsigned id = graph[n][m];
                if (flags[id])
                    continue;
                flags[id] = true;
                float dist = distance_func(base_data + dimension * id, query, dimension);
#ifdef RECORD_DIST_COMPS
                local_dist_comps++;
#endif
                if (dist >= retset[tmp_l - 1].distance)
                    continue;
                Neighbor nn(id, dist, true);
                int r = InsertIntoPool(retset.data(), tmp_l, nn);
                if (tmp_l < L)
                    tmp_l++;
                if (r < nk)
                    nk = r;
            }
            hop++;
        }
        if (nk <= k)
            k = nk;
        else
            ++k;
    }
}

void ANNSearch::EdgeWiseMultiThreadSearch(const float *query, unsigned query_id, int K, int L, int num_threads, boost::dynamic_bitset<> &flags, std::vector<unsigned> &indices)
{
#ifdef BREAKDOWN_ANALYSIS
    time_seq_ -= get_time_mark();
#endif
    int ep = default_ep;
    std::vector<unsigned> init_ids(L);
    unsigned tmp_l = 0;
    for (; tmp_l < L && tmp_l < graph[ep].size(); tmp_l++)
    {
        init_ids[tmp_l] = graph[ep][tmp_l];
        flags[init_ids[tmp_l]] = true;
    }
    while (tmp_l < K)
    {
        int id = rand() % base_num;
        if (flags[id])
            continue;
        init_ids[tmp_l] = id;
        flags[id] = true;
        tmp_l++;
    }
    std::vector<Neighbor> retset(L + 1);
    for (unsigned j = 0; j < tmp_l; j++)
    {
        unsigned id = init_ids[j];
        _mm_prefetch(base_data + dimension * id, _MM_HINT_T0);
        float dist = distance_func(base_data + dimension * id, query, dimension);
#ifdef RECORD_DIST_COMPS
        dist_comps++;
        max_dist_comps++;
#endif
        retset[j] = Neighbor(id, dist, true);
    }
    std::sort(retset.begin(), retset.begin() + tmp_l); // sort the retset by distance in ascending order
    int k = 0;
    int hop = 0;
    std::vector<std::vector<Neighbor>> local_candidates_per_thread(num_threads);
    std::vector<int64_t> dist_comps_per_thread(num_threads);
    for (int i = 0; i < num_threads; i++)
        dist_comps_per_thread[i] = 0;
#ifdef BREAKDOWN_ANALYSIS
    time_seq_ += get_time_mark();
    time_expand_ -= get_time_mark();
#endif
    while (k < (int)L)
    {
        int nk = L;
        if (retset[k].unexplored)
        {
            retset[k].unexplored = false;
            unsigned n = retset[k].id;
            _mm_prefetch(graph[n].data(), _MM_HINT_T0);
#pragma omp parallel for num_threads(num_threads)
            for (unsigned m = 0; m < graph[n].size(); ++m)
            {
                int tid = omp_get_thread_num();
                unsigned id = graph[n][m];
                if (flags[id])
                    continue;
                flags[id] = true;
                if (m + 1 < graph[n].size())
                {
                    _mm_prefetch(base_data + dimension * graph[n][m + 1], _MM_HINT_T0);
                }
                float dist = distance_func(query, base_data + dimension * id, dimension);
#ifdef RECORD_DIST_COMPS
                dist_comps_per_thread[tid]++;
#endif
                if (dist >= retset[tmp_l - 1].distance)
                    continue;
                Neighbor nn(id, dist, true);

                local_candidates_per_thread[tid].push_back(nn);
            }
#ifdef BREAKDOWN_ANALYSIS
            time_merge_ -= get_time_mark();
#endif
            for (int tid = 0; tid < num_threads; ++tid)
            {
                auto &local_candidates = local_candidates_per_thread[tid];
                for (auto &nn : local_candidates)
                {
                    int r = InsertIntoPool(retset.data(), tmp_l, nn);
                    if (r < nk)
                        nk = r;
                    if (tmp_l < L)
                        tmp_l++;
                }
                local_candidates.clear(); // 清空供下一轮使用
            }
            hop++;
#ifdef BREAKDOWN_ANALYSIS
            time_merge_ += get_time_mark();
#endif
        }
        if (nk <= k)
            k = nk;
        else
            ++k;
    }
#ifdef BREAKDOWN_ANALYSIS
    time_expand_ += get_time_mark();
    time_seq_ -= get_time_mark();
#endif
    for (size_t i = 0; i < K; i++)
    {
        indices[i] = retset[i].id;
    }
    flags.reset();
    float mincomps = 1000000, maxcomps = 0;
    for (int i = 0; i < num_threads; i++)
    {
        dist_comps += dist_comps_per_thread[i];
        if (dist_comps_per_thread[i] < mincomps)
            mincomps = dist_comps_per_thread[i];
        if (dist_comps_per_thread[i] > maxcomps)
            maxcomps = dist_comps_per_thread[i];
    }
    max_dist_comps += maxcomps;
    ub_ratio += maxcomps / mincomps;
    hop *= num_threads;
    hop_count += hop;
#ifdef BREAKDOWN_ANALYSIS
    time_seq_ += get_time_mark();
#endif
}

void ANNSearch::ModifiedDeltaStepping(const float *query, unsigned query_id, int K, int L, int num_threads, boost::dynamic_bitset<> &flags, std::vector<unsigned> &indices)
{
#ifdef BREAKDOWN_ANALYSIS
    time_seq_ -= get_time_mark();
#endif
    int ep = default_ep;
    std::vector<unsigned> init_ids;
    for (int i = 0; i < L && i < graph[ep].size(); ++i)
    {
        unsigned id = graph[ep][i];
        if (flags.test(id))
            continue;
        flags.set(id);
        init_ids.push_back(id);
    }
    std::vector<Neighbor> retset;
    for (unsigned id : init_ids)
    {
        _mm_prefetch(base_data + dimension * id, _MM_HINT_T0);
        float dist = distance_func(base_data + dimension * id, query, dimension);
#ifdef RECORD_DIST_COMPS
        dist_comps++;
        max_dist_comps++;
#endif
        retset.emplace_back(id, dist, true);
    }
    std::sort(retset.begin(), retset.end());
    if (retset.size() < L)
        retset.resize(L);
    int current_size = static_cast<int>(init_ids.size());
    bool has_unexplored = true;
    std::vector<std::vector<Neighbor>> local_candidates(num_threads);
    std::vector<int> thread_ids(num_threads);
    std::vector<int64_t> dist_comps_per_thread(num_threads);
    for (int i = 0; i < num_threads; i++)
        dist_comps_per_thread[i] = 0;
#ifdef BREAKDOWN_ANALYSIS
    time_seq_ += get_time_mark();
#endif
    while (has_unexplored)
    {
#ifdef BREAKDOWN_ANALYSIS
        time_seq_ -= get_time_mark();
#endif
        has_unexplored = false;
        thread_ids.clear();
        for (int i = 0; i < L && i < current_size; ++i)
        {
            if (retset[i].unexplored)
            {
                thread_ids.push_back(i);
                if (thread_ids.size() >= static_cast<size_t>(num_threads))
                    break;
            }
        }
        if (thread_ids.empty())
            break;
        int batch_size = thread_ids.size();
        for (auto &buf : local_candidates)
            buf.clear();
#ifdef BREAKDOWN_ANALYSIS
        time_seq_ += get_time_mark();
        time_expand_ -= get_time_mark();
#endif
#pragma omp parallel for num_threads(num_threads)
        for (int b = 0; b < batch_size; ++b)
        {
            int tid = omp_get_thread_num();
            int k = thread_ids[b];
            unsigned n = retset[k].id;
            _mm_prefetch(graph[n].data(), _MM_HINT_T0);
            for (unsigned m = 0; m < graph[n].size(); ++m)
            {
                unsigned id = graph[n][m];
                if (flags[id])
                    continue;
                flags[id] = true;
                if (m + 1 < graph[n].size())
                {
                    _mm_prefetch(base_data + dimension * graph[n][m + 1], _MM_HINT_T0);
                }
                float dist = distance_func(query, base_data + dimension * id, dimension);
#ifdef RECORD_DIST_COMPS
                dist_comps_per_thread[tid]++;
#endif
                if (dist >= retset[L - 1].distance && current_size >= L)
                    continue;
                local_candidates[tid].emplace_back(id, dist, true);
            }
        }
#ifdef BREAKDOWN_ANALYSIS
        time_expand_ += get_time_mark();
        time_merge_ -= get_time_mark();
#endif
        int nk = L;
        for (int tid = 0; tid < num_threads; ++tid)
        {
            for (auto &nn : local_candidates[tid])
            {
                int r = InsertIntoPool(retset.data(), current_size, nn);
                if (r < nk)
                    nk = r;
                if (current_size < L)
                    ++current_size;
            }
        }
        for (int idx : thread_ids)
        {
            retset[idx].unexplored = false;
        }
        for (int i = 0; i < L && i < current_size; ++i)
        {
            if (retset[i].unexplored)
            {
                has_unexplored = true;
                break;
            }
        }
#ifdef BREAKDOWN_ANALYSIS
        time_merge_ += get_time_mark();
#endif
    }
#ifdef BREAKDOWN_ANALYSIS
    time_seq_ -= get_time_mark();
#endif
    for (int i = 0; i < K && i < retset.size(); ++i)
    {
        indices[i] = retset[i].id;
    }
    float mincomps = 1000000, maxcomps = 0;
    for (int i = 0; i < num_threads; i++)
    {
        dist_comps += dist_comps_per_thread[i];
        if (dist_comps_per_thread[i] < mincomps)
            mincomps = dist_comps_per_thread[i];
        if (dist_comps_per_thread[i] > maxcomps)
            maxcomps = dist_comps_per_thread[i];
    }
    max_dist_comps += maxcomps;
    ub_ratio += maxcomps / mincomps;
    flags.reset();
#ifdef BREAKDOWN_ANALYSIS
    time_seq_ += get_time_mark();
#endif
}

void ANNSearch::MultiTurnSearch(const float *query, unsigned query_id, int K, int L, int num_turns, boost::dynamic_bitset<> &flags, std::vector<unsigned> &indices)
{
    std::vector<std::vector<Neighbor>> retsets(num_turns);
    for (int i = 0; i < num_turns; i++)
    {
        std::vector<unsigned> init_ids(L);
        unsigned tmp_l = 0;
        retsets[i].resize(L + 1);
        // int ep = rand() % base_num;
        for (int j = 0; j < graph[default_ep].size(); j++)
        {
            if (j % num_turns == i)
            {
                init_ids[tmp_l] = graph[default_ep][j];
                flags[init_ids[tmp_l]] = true;
                tmp_l++;
            }
        }
        for (unsigned j = 0; j < tmp_l; j++)
        {
            unsigned id = init_ids[j];
            _mm_prefetch(base_data + dimension * id, _MM_HINT_T0);
            float dist = distance_func(base_data + dimension * id, query, dimension);
#ifdef COLLECT_VISITED_ID
            visited_lists[i].push_back(id);
#endif
#ifdef RECORD_DIST_COMPS
            dist_comps++;
#endif
            retsets[i][j] = Neighbor(id, dist, true);
        }
        std::sort(retsets[i].begin(), retsets[i].begin() + tmp_l); // sort the retset by distance in ascending order
        int k = 0;
        int hop = 0;
        while (k < (int)L)
        {
            int nk = L;
            if (retsets[i][k].unexplored)
            {
                retsets[i][k].unexplored = false;
                unsigned n = retsets[i][k].id;
                _mm_prefetch(graph[n].data(), _MM_HINT_T0);
                for (unsigned m = 0; m < graph[n].size(); ++m)
                {
                    unsigned id = graph[n][m];
                    if (m + 1 < graph[n].size())
                    {
                        _mm_prefetch(base_data + dimension * graph[n][m + 1], _MM_HINT_T0);
                    }
                    if (flags[id])
                        continue;
                    flags[id] = true;
                    float dist = distance_func(query, base_data + dimension * id, dimension);
#ifdef COLLECT_VISITED_ID
                    visited_lists[i].push_back(id);
#endif
#ifdef RECORD_DIST_COMPS
                    dist_comps++;
#endif
                    if (dist >= retsets[i][tmp_l - 1].distance)
                        continue;
                    Neighbor nn(id, dist, true);
                    int r = InsertIntoPool(retsets[i].data(), tmp_l, nn);
                    if (tmp_l < L)
                        tmp_l++;
                    if (r < nk)
                        nk = r;
                }
                hop++;
            }
            if (nk <= k)
                k = nk;
            else
                ++k;
        }
        hop_count += hop;
    }
    for (int i = 1; i < num_turns; i++)
    {
        for (size_t j = 0; j < K; j++)
        {
            int pos = InsertIntoPool(retsets[0].data(), K, retsets[i][j]);
            if (pos == K)
                break;
        }
    }
    for (size_t i = 0; i < K; i++)
    {
        indices[i] = retsets[0][i].id;
    }
    flags.reset();
}