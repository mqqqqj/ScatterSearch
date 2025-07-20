#include <annsearch.h>
#include <util.h>
#include <map>
#include <sys/resource.h>
#include <chrono>
#include <numeric>
#include <unordered_set>

int main(int argc, char **argv)
{
    // 禁用core文件生成
    struct rlimit limit;
    limit.rlim_cur = 0;
    limit.rlim_max = 0;
    setrlimit(RLIMIT_CORE, &limit);
    srand(time(0));
    if (argc != 9)
    {
        std::cout << argv[0]
                  << " data_file query_file nsg_path search_L search_K num_threads query_id gt_path"
                  << std::endl;
        exit(-1);
    }
    float *data_load = nullptr;
    unsigned points_num, dim;
    load_fbin(argv[1], data_load, points_num, dim);
    float *query_load = NULL;
    unsigned query_num, query_dim;
    load_fbin(argv[2], query_load, query_num, query_dim);
    assert(dim == query_dim);
    int L = atoi(argv[4]);
    int K = atoi(argv[5]);
    int num_threads = atoi(argv[6]);
    int query_id = atoi(argv[7]);
    std::vector<std::vector<unsigned>> groundtruth;
    load_groundtruth(argv[8], groundtruth);

    std::cout << "Groundtruth loaded" << std::endl;
    if (L < K)
    {
        std::cout << "search_L cannot be smaller than search_K!" << std::endl;
        exit(-1);
    }
    ANNSearch engine(dim, points_num, data_load, INNER_PRODUCT);
    engine.LoadGraph(argv[3]);
    engine.LoadGroundtruth(argv[8]);
    boost::dynamic_bitset<> flags{points_num, 0};
    std::vector<unsigned> tmp(K);

    engine.MultiThreadSearchArraySimulation(query_load + (size_t)query_id * dim, query_id, K, L, num_threads, flags, tmp);
    // engine.MultiThreadSearchArraySimulationWithET(query_load + (size_t)query_id * dim, query_id, K, L, num_threads, flags, tmp);

    // 基于tmp和groundtruth计算recall
    std::unordered_set<unsigned> gt_set(groundtruth[query_id].begin(), groundtruth[query_id].end());
    unsigned hit = 0;
    for (unsigned id : tmp)
    {
        if (gt_set.count(id))
            hit++;
    }
    float recall = static_cast<float>(hit) / K;
    std::cout << "Recall: " << recall << std::endl;
    std::cout << "Computations: " << engine.dist_comps << std::endl;
    return 0;
}