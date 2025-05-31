#include <annsearch.h>
#include <fstream>
#include <cstdlib>
#include <sys/time.h>
#include <omp.h>
#include <util.h>

#define BREAKDOWN_PRINT

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

double ANNSearch::get_time_mark()
{
    timeval t;
    gettimeofday(&t, nullptr);
    return t.tv_sec + t.tv_usec * 0.000001;
}

inline int ANNSearch::InsertIntoPool(Neighbor *addr, unsigned K, Neighbor nn)
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
#pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < num_threads; i++)
    {
        int ep = rand() % base_num;
        std::vector<unsigned> init_ids(L);
        unsigned tmp_l = 0;
        for (; tmp_l < L && tmp_l < graph[ep].size(); tmp_l++)
        {
            init_ids[tmp_l] = graph[ep][tmp_l];
            flags[init_ids[tmp_l]] = true;
        }

        while (tmp_l < L)
        {
            unsigned id = rand() % base_num;
            if (flags[id])
                continue;
            flags[id] = true;
            init_ids[tmp_l] = id;
            tmp_l++;
        }
        retsets[i].resize(L + 1);
        for (unsigned j = 0; j < init_ids.size(); j++)
        {
            unsigned id = init_ids[j];
            float dist = distance_func(base_data + dimension * id, query, dimension);
            retsets[i][j] = Neighbor(id, dist, true);
        }
        std::sort(retsets[i].begin(), retsets[i].begin() + L); // sort the retset by distance in ascending order
        int k = 0;
        while (k < (int)L)
        {
            int nk = L;
            // if(finish_num >= num_threads / 2)
            //   break;
            if (retsets[i][k].unexplored)
            {
                retsets[i][k].unexplored = false;
                unsigned n = retsets[i][k].id; // current node
                for (unsigned m = 0; m < graph[n].size(); ++m)
                {
                    unsigned id = graph[n][m];
                    if (flags[id])
                        continue;
                    flags[id] = 1;
                    float dist = distance_func(query, base_data + dimension * id, dimension);
                    if (dist >= retsets[i][L - 1].distance)
                        continue;
                    Neighbor nn(id, dist, true);
                    int r = InsertIntoPool(retsets[i].data(), L, nn);
                    if (r < nk)
                        nk = r;
                }
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
