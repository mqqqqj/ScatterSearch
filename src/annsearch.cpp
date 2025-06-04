#include <annsearch.h>
#include <fstream>
#include <cstdlib>
#include <sys/time.h>
#include <omp.h>
#include <xmmintrin.h> // 添加 SSE 头文件，包含 _mm_prefetch 和 _MM_HINT_T0

#define BREAKDOWN_PRINT
#define AVX
// #define SSE
ANNSearch::ANNSearch(unsigned dim, unsigned num, float *base, Metric m)
{
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
        // distance_func = distance_ip_sse;
        distance_func = distance_ip_sse_simple;
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
    int ep = rand() % base_num;
    std::vector<unsigned> init_ids(L);
    unsigned tmp_l = 0;
    for (; tmp_l < L && tmp_l < graph[ep].size(); tmp_l++)
    {
        init_ids[tmp_l] = graph[ep][tmp_l];
        flags[init_ids[tmp_l]] = true;
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
    for (size_t i = 0; i < K; i++)
    {
        indices[i] = retset[i].id;
    }
}

void ANNSearch::SearchArraySimulationForPipeline(const float *query, unsigned query_id, int K, int L, boost::dynamic_bitset<> &flags, std::vector<Neighbor> &indices)
{
    int ep = rand() % base_num;
    std::vector<unsigned> init_ids(L);
    unsigned tmp_l = 0;
    for (; tmp_l < L && tmp_l < graph[ep].size(); tmp_l++)
    {
        init_ids[tmp_l] = graph[ep][tmp_l];
        flags[init_ids[tmp_l]] = true;
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

void ANNSearch::SearchArraySimulationForPipelineWithET(const float *query, unsigned query_id, int thread_id, int K, int L, boost::dynamic_bitset<> &flags,
                                                       std::atomic<bool> &stop, std::atomic<float> &best_dist, std::atomic<int> &best_thread_id,
                                                       std::vector<Neighbor> &indices)
{
    std::vector<Neighbor> retset(L + 1);
    unsigned tmp_l = 0;
    while (tmp_l < K)
    {
        int id = rand() % base_num;
        while (flags[id])
            id = rand() % base_num;
        float dist = distance_func(base_data + dimension * id, query, dimension);
        retset[tmp_l] = Neighbor(id, dist, true);
        tmp_l++;
    }
    std::sort(retset.begin(), retset.begin() + tmp_l); // sort the retset by distance in ascending order
    int k = 0;
    int hop = 0;
    while (k < (int)L)
    {
        if (stop)
            break;
        if (hop == 10)
        {
            if (best_dist > retset[0].distance)
            {
                best_dist = retset[0].distance;
                best_thread_id = thread_id;
            }
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
    std::vector<std::vector<Neighbor>> retsets(num_threads);
    int finish_num = 0;
#pragma omp parallel num_threads(num_threads)
    {
        int i = omp_get_thread_num();
        int ep = rand() % base_num;
        std::vector<unsigned> init_ids(L);
        unsigned tmp_l = 0;
        for (; tmp_l < L && tmp_l < graph[ep].size(); tmp_l++)
        {
            init_ids[tmp_l] = graph[ep][tmp_l];
            flags[init_ids[tmp_l]] = true;
        }
        retsets[i].resize(L + 1);
        for (unsigned j = 0; j < tmp_l; j++)
        {
            unsigned id = init_ids[j];
            _mm_prefetch(base_data + dimension * id, _MM_HINT_T0);
            float dist = distance_func(base_data + dimension * id, query, dimension);
            retsets[i][j] = Neighbor(id, dist, true);
        }
        std::sort(retsets[i].begin(), retsets[i].begin() + tmp_l); // sort the retset by distance in ascending order
        int k = 0;
        int hop = 0;
        while (k < (int)L) // && hop < L
        {
            int nk = L;
            // if (finish_num >= num_threads / 2)
            //     break;
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
    }
    for (int i = 1; i < num_threads; i++)
    {
        for (size_t j = 0; j < K; j++)
        {
            InsertIntoPool(retsets[0].data(), K, retsets[i][j]);
        }
    }
    for (size_t i = 0; i < K; i++)
    {
        indices[i] = retsets[0][i].id;
    }
}

void ANNSearch::MultiThreadSearchArraySimulationWithET(const float *query, unsigned query_id, int K, int L, int num_threads, boost::dynamic_bitset<> &flags, std::vector<unsigned> &indices)
{
    std::vector<std::vector<Neighbor>> retsets(num_threads);
    std::vector<std::vector<Neighbor>> new_retsets(num_threads);
    int finish_num = 0;
    int best_thread_id = -1;
    std::atomic<int> decide_num;
    decide_num = 0;
    std::atomic<float> best_dist;
    best_dist = 1000;
    std::atomic<bool> best_thread_finish;
    best_thread_finish = false;
    bool good_thread[num_threads];
    memset(good_thread, 0, sizeof(bool) * num_threads);
    std::atomic<int> good_thread_num;
    good_thread_num = 0;
    std::atomic<int> good_thread_finish_num;
    good_thread_finish_num = 0;
#pragma omp parallel num_threads(num_threads)
    {
        int i = omp_get_thread_num();
        int ep = rand() % base_num;
        int hop = 0;
        std::vector<unsigned> init_ids(L);
        bool need_identify = true;
        unsigned tmp_l = 0;
        for (; tmp_l < L && tmp_l < graph[ep].size(); tmp_l++)
        {
            init_ids[tmp_l] = graph[ep][tmp_l];
            flags[init_ids[tmp_l]] = true;
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
            if (best_thread_finish)
                break;
            if (hop == 10)
            {
                decide_num++;
                if (best_dist > retsets[i][0].distance)
                {
                    best_dist = retsets[i][0].distance;
                    best_thread_id = i;
                }
            }
            if (need_identify && decide_num == num_threads)
            {
                need_identify = false;
                if (best_dist < 1.1 * retsets[i][0].distance)
                {
                    // bad search, start a new one
                    std::vector<Neighbor> new_retset;
                    SearchUntilBestThreadStop(query, query_id, K, L, best_thread_finish, best_dist, flags, new_retset);
                    new_retsets[i] = new_retset;
                    break;
                }
                if (best_dist > 1.02 * retsets[i][0].distance)
                {
                    good_thread_num++;
                    good_thread[i] = true;
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
        // if (good_thread[i])
        // {
        //     good_thread_finish_num++;
        // }
        // if (good_thread_finish_num == good_thread_num)
        //     best_thread_finish = true;
        if (i == best_thread_id)
            best_thread_finish = true;
        else
        {
            if (best_thread_finish == false)
            {
                std::vector<Neighbor> new_retset;
                SearchUntilBestThreadStop(query, query_id, K, L, best_thread_finish, best_dist, flags, new_retset);
                new_retsets[i] = new_retset;
            }
        }
    }
    for (int i = 1; i < num_threads; i++)
    {
        for (size_t j = 0; j < K; j++)
        {
            InsertIntoPool(retsets[0].data(), K, retsets[i][j]);
        }
    }
    for (size_t i = 0; i < new_retsets.size(); i++)
    {
        if (new_retsets[i].size())
            for (int j = 0; j < K; j++)
            {
                InsertIntoPool(retsets[0].data(), K, new_retsets[i][j]);
            }
    }
    for (size_t i = 0; i < K; i++)
    {
        indices[i] = retsets[0][i].id;
    }
}

void ANNSearch::SearchUntilBestThreadStop(const float *query, unsigned query_id, int K, int L, std::atomic<bool> &best_thread_finish, std::atomic<float> &best_dist, boost::dynamic_bitset<> &flags, std::vector<Neighbor> &neighbors)
{
    std::vector<std::vector<Neighbor>> retsets;
    std::vector<int> retset_size;
    int current_turn = 0;
    while (current_turn < 10) // true
    {
        if (best_thread_finish)
        {
            break;
        }
        std::vector<Neighbor> retset(L + 1);
        retsets.push_back(retset);
        retset_size.push_back(0);
        int &tmp_l = retset_size[current_turn];
        int ep = rand() % base_num;
        for (; tmp_l < L && tmp_l < graph[ep].size(); tmp_l++)
        {
            unsigned id = graph[ep][tmp_l];
            if (flags[id])
                continue;
            flags[id] = true;
            float dist = distance_func(base_data + dimension * id, query, dimension);
            retsets[current_turn][tmp_l] = Neighbor(id, dist, true);
        }
        std::sort(retsets[current_turn].begin(), retsets[current_turn].begin() + tmp_l);
        int k = 0;
        int hop = 0;
        while ((k < (int)L))
        {
            int nk = L;
            if (best_thread_finish)
                break;
            // if(hop == 10 && best_dist < 1.1 * retsets[current_turn][0].distance)
            // {
            //     break;
            // }
            if (retsets[current_turn][k].unexplored)
            {
                retsets[current_turn][k].unexplored = false;
                unsigned n = retsets[current_turn][k].id;
                for (unsigned m = 0; m < graph[n].size(); ++m)
                {
                    unsigned id = graph[n][m];
                    if (flags[id])
                        continue;
                    flags[id] = true;
                    float dist = distance_func(base_data + dimension * id, query, dimension);
                    if (dist >= retsets[current_turn][tmp_l - 1].distance)
                        continue;
                    Neighbor nn(id, dist, true);
                    int r = InsertIntoPool(retsets[current_turn].data(), tmp_l, nn);
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
        // evaluate recall@100
        // int correct = 0;
        // for (int i = 0; i < K; i++)
        // {
        //     for (int j = 0; j < K; j++)
        //     {
        //         if (retsets[current_turn][i].id == groundtruth[query_id][j])
        //         {
        //             correct++;
        //             break;
        //         }
        //     }
        // }
        // std::cout << "query " << query_id << ", turn " << current_turn << ", recall: " << (float)correct / K << std::endl;
        current_turn++;
    }
    int master = -1;
    for (size_t i = 0; i < retsets.size(); i++)
    {
        if (retset_size[i] >= K)
            master = i;
    }
    if (master != -1)
    {
        for (size_t i = 0; i < retsets.size(); i++)
        {
            for (size_t j = 0; j < retset_size[i]; j++)
            {
                InsertIntoPool(retsets[master].data(), K, retsets[i][j]);
            }
        }
        neighbors.resize(K);
        for (size_t i = 0; i < K; i++)
        {
            neighbors[i] = retsets[master][i];
        }
    }
}