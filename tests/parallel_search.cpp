#include <annsearch.h>
#include <util.h>
#include <map>
#include <sys/resource.h>
#include <chrono>
#include <numeric>
#include <sstream>

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
                  << " data_file query_file nsg_path search_L_list search_K num_threads gt_path dataset_name"
                  << std::endl;
        std::cout << "search_L_list format: L1,L2,L3,... (comma separated values)" << std::endl;
        exit(-1);
    }
    float *data_load = nullptr;
    unsigned points_num, dim;
    load_fbin(argv[1], data_load, points_num, dim);
    float *query_load = NULL;
    unsigned query_num, query_dim;
    load_fbin(argv[2], query_load, query_num, query_dim);
    assert(dim == query_dim);

    // 解析L_list
    std::vector<int> L_list;
    std::string L_str = argv[4];
    std::stringstream ss(L_str);
    std::string L_val;
    while (std::getline(ss, L_val, ','))
    {
        L_list.push_back(std::stoi(L_val));
    }

    int K = atoi(argv[5]);
    int num_threads = atoi(argv[6]);
    std::vector<std::vector<unsigned>> groundtruth;
    load_groundtruth(argv[7], groundtruth);
    std::string dataset_name = argv[8];
    if (query_num > 10000)
        query_num = 10000;
    query_num = 1000;
    std::cout << "Groundtruth loaded" << std::endl;

    // 检查所有L值是否合法
    for (int L : L_list)
    {
        if (L < K)
        {
            std::cout << "search_L cannot be smaller than search_K!" << std::endl;
            exit(-1);
        }
    }

    ANNSearch engine(dim, points_num, data_load, INNER_PRODUCT);
    engine.LoadGraph(argv[3]);
    engine.LoadGroundtruth(argv[7]);
    std::cout << "L,Throughput,latency,recall,p95recall,p99recall,total_dist_comps,max_dist_comps,hops,t_expand(s.),t_merge(s.),t_seq(s.),t_p_expand(%),t_p_merge(%),t_p_seq(%)" << std::endl;
    std::vector<TestResult> test_results;
    // 对每个L值进行搜索
    for (int L : L_list)
    {
        boost::dynamic_bitset<> flags{points_num, 0};
        std::vector<std::vector<unsigned>> res(query_num);
        std::vector<float> latency_list(query_num); // 单位：毫秒
        auto s = std::chrono::high_resolution_clock::now();
        for (unsigned i = 0; i < query_num; i++)
        {
            std::vector<unsigned> tmp(K);
            auto start_time = std::chrono::high_resolution_clock::now();
            // engine.MultiThreadSearchArraySimulation(query_load + (size_t)i * dim, i, K, L, num_threads, flags, tmp);
            engine.MultiThreadSearchArraySimulationWithET(query_load + (size_t)i * dim, i, K, L, num_threads, flags, tmp);
            // engine.EdgeWiseMultiThreadSearch(query_load + (size_t)i * dim, i, K, L, num_threads, flags, tmp);
            // engine.ModifiedDeltaStepping(query_load + (size_t)i * dim, i, K, L, num_threads, flags, tmp);
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

            latency_list[i] = duration.count() / 1000.0f; // 转换为毫秒
            res[i] = tmp;
        }
        auto e = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = e - s;
        float qps = query_num / diff.count();
        float accumulate_latency = std::accumulate(latency_list.begin(), latency_list.end(), 0.0f);
        float avg_latency = accumulate_latency / latency_list.size();
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
        std::sort(recalls.begin(), recalls.end());
        engine.ub_ratio /= query_num;
        // std::cout << "unbalance ratio: " << engine.ub_ratio << std::endl;
        engine.ub_ratio = 0;
        TestResult tr{L, qps, avg_latency, avg_recall, recalls[recalls.size() * 0.05], recalls[recalls.size() * 0.01], (float)engine.dist_comps / query_num, (float)engine.hop_count / (query_num * num_threads)};

        test_results.push_back(tr);
        std::cout << tr.L << "," << tr.throughput << "," << tr.latency << "," << tr.recall << "," << tr.p95_recall << "," << tr.p99_recall << "," << tr.dist_comps << "," << (float)engine.max_dist_comps / query_num << "," << tr.hops << "," << engine.time_expand_ << "," << engine.time_merge_ << "," << engine.time_seq_ << "," << 100000 * engine.time_expand_ / accumulate_latency << "," << 100000 * engine.time_merge_ / accumulate_latency << "," << 100000 * engine.time_seq_ / accumulate_latency << std::endl;
        engine.dist_comps = 0;
        engine.max_dist_comps = 0;
        engine.hop_count = 0;
        engine.time_expand_ = 0;
        engine.time_merge_ = 0;
        engine.time_seq_ = 0;
    }
    std::string save_path = "./results/" + dataset_name + "_parallel_" + std::to_string(num_threads) + "t.csv";
    // save_results(test_results, save_path);
    return 0;
}