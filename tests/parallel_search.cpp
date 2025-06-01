#include <annsearch.h>
#include <util.h>
#include <map>
#include <sys/resource.h>
#include <chrono>
#include <numeric>

int main(int argc, char **argv)
{
    // 禁用core文件生成
    struct rlimit limit;
    limit.rlim_cur = 0;
    limit.rlim_max = 0;
    setrlimit(RLIMIT_CORE, &limit);
    srand(time(0));
    if (argc != 8)
    {
        std::cout << argv[0]
                  << " data_file query_file nsg_path search_L search_K num_threads gt_path"
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
    std::vector<std::vector<unsigned>> groundtruth;
    load_groundtruth(argv[7], groundtruth);
    if (query_num > 10000)
        query_num = 10000;
    std::cout << "Groundtruth loaded" << std::endl;
    if (L < K)
    {
        std::cout << "search_L cannot be smaller than search_K!" << std::endl;
        exit(-1);
    }
    ANNSearch engine(dim, points_num, data_load, INNER_PRODUCT);
    engine.LoadGraph(argv[3]);
    boost::dynamic_bitset<> flags{points_num, 0};
    std::vector<std::vector<unsigned>> res(query_num);
    std::vector<float> latency_list(query_num); // 单位：毫秒
    for (unsigned i = 0; i < query_num; i++)
    {
        std::vector<unsigned> tmp(K);
        auto start_time = std::chrono::high_resolution_clock::now();
        // engine.Search(query_load + (size_t)i * dim, i, K, L, flags, tmp);
        // engine.MultiThreadSearch(query_load + (size_t)i * dim, i, K, L, num_threads, flags, tmp);
        // engine.MultiThreadSearchArraySimulation(query_load + (size_t)i * dim, i, K, L, num_threads, flags, tmp);
        engine.MultiThreadSearchArraySimulationWithET(query_load + (size_t)i * dim, i, K, L, num_threads, flags, tmp);
        flags.reset();
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        latency_list[i] = duration.count() / 1000.0f; // 转换为毫秒
        res[i] = tmp;
        if (i % 1000 == 999)
        {
            std::cout << "query " << i << " done" << std::endl;
        }
    }
    // 计算平均latency
    float accumulate_latency = std::accumulate(latency_list.begin(), latency_list.end(), 0.0f);
    float avg_latency = accumulate_latency / latency_list.size();
    std::cout << "avg_latency: " << avg_latency << " ms" << std::endl;
    std::vector<float> recalls(query_num);
    for (unsigned i = 0; i < query_num; i++)
    {
        int correct = 0;
        for (unsigned j = 0; j < K; j++)
        {
            for (unsigned g = 0; g < K; g++)
            {
                if (res[i][j] == groundtruth[i][g])
                {
                    correct++;
                    break;
                }
            }
        }
        recalls[i] = (float)correct / K;
    }
    float accumulate_recall = std::accumulate(recalls.begin(), recalls.end(), 0.0f);
    float avg_recall = accumulate_recall / recalls.size();
    std::cout << "avg_recall: " << avg_recall << std::endl;
    std::sort(recalls.begin(), recalls.end());
    float p95_recall = recalls[recalls.size() * 0.05];
    std::cout << "p95_recall: " << p95_recall << std::endl;
    float p99_recall = recalls[recalls.size() * 0.01];
    std::cout << "p99_recall: " << p99_recall << std::endl;
    return 0;
}