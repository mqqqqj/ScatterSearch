#include <annsearch.h>
#include <util.h>
#include <map>
#include <sys/resource.h>
#include <chrono>
#include <numeric>
#include <ThreadPool.h>

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
    // query_num = 100;
    std::cout << "Groundtruth loaded" << std::endl;
    if (L < K)
    {
        std::cout << "search_L cannot be smaller than search_K!" << std::endl;
        exit(-1);
    }
    ANNSearch engine(dim, points_num, data_load, INNER_PRODUCT);
    engine.LoadGraph(argv[3]);
    engine.LoadGroundtruth(argv[7]);
    std::vector<float> latency_list(query_num); // 单位：毫秒
    ThreadPool pool(num_threads);
    std::vector<std::future<void>> futures;
    std::vector<std::vector<std::vector<Neighbor>>> res(query_num, std::vector<std::vector<Neighbor>>(num_threads));
    std::vector<std::vector<std::chrono::high_resolution_clock::time_point>> query_start_times(query_num, std::vector<std::chrono::high_resolution_clock::time_point>(num_threads));
    std::vector<std::vector<std::chrono::high_resolution_clock::time_point>> query_end_times(query_num, std::vector<std::chrono::high_resolution_clock::time_point>(num_threads));
    // for early termination
    int master_thread[query_num];
    memset(master_thread, -1, sizeof(int) * query_num);
    std::atomic<int> finish_num[query_num];
    int best_thread_ndc[query_num];
    memset(best_thread_ndc, 0, sizeof(int) * query_num);
    std::atomic<float> best_dist[query_num];
    std::atomic<int> best_thread_id[query_num];
    std::atomic<bool> best_thread_finish[query_num];
    for (unsigned i = 0; i < query_num; i++)
    {
        finish_num[i] = 0;
        best_dist[i] = 1000;
        best_thread_id[i] = -1;
        best_thread_finish[i] = false;
    }
    auto s = std::chrono::high_resolution_clock::now();
    int flag_pool_size = 20;
    std::vector<boost::dynamic_bitset<>> flags(flag_pool_size);
    for (unsigned i = 0; i < flag_pool_size; i++)
    {
        flags[i] = boost::dynamic_bitset<>(points_num, 0);
    }
    for (unsigned i = 0; i < query_num; i++)
    {
        float *query_ptr = query_load + (size_t)i * dim;
        for (int j = 0; j < num_threads; j++)
        {
            futures.push_back(pool.enqueue([&, i, j, query_ptr]()
                                           {
                    query_start_times[i][j] = std::chrono::high_resolution_clock::now();
                    int flag_idx = i % flag_pool_size;
                    engine.SearchArraySimulationForPipelineWithET(query_load + (size_t)i * dim, i, j, K, L, flags[flag_idx], best_thread_finish[i], best_dist[i], best_thread_id[i], res[i][j]);
                    finish_num[i] ++;
                    // if(finish_num[i] >= num_threads / 2)
                    //     best_thread_finish[i] = true;
                    if(best_thread_id[i] == j)
                        best_thread_finish[i] = true;
                    if(finish_num[i] == num_threads)
                    {
                        flags[flag_idx].reset();
                        // merge result
                        int master = -1;
                        for(unsigned t = 0; t < num_threads; t++)
                            if(res[i][t].size())
                                master = t;
                        master_thread[i] = master;
                        for (unsigned t = 0; t < num_threads; t++)
                        {
                            if(res[i][t].size() && t != master)
                                for (unsigned k = 0; k < K; k++)
                                {
                                    InsertIntoPool(res[i][master].data(), K, res[i][t][k]);
                                }
                        }
                    }
                    query_end_times[i][j] = std::chrono::high_resolution_clock::now(); }));
        }
    }
    std::cout << "Waiting for all threads to finish" << std::endl;
    for (size_t i = 0; i < futures.size(); i++)
    {
        futures[i].get();
    }
    std::cout << "All threads finished" << std::endl;
    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    std::cout << "Throughput(QPS): " << query_num / diff.count() << std::endl;
    std::cout << "Query time(s): " << diff.count() << std::endl;
    // 计算每个查询的延迟
    for (unsigned i = 0; i < query_num; i++)
    {
        auto earliest_start = *std::min_element(query_start_times[i].begin(), query_start_times[i].end());
        auto latest_end = *std::max_element(query_end_times[i].begin(), query_end_times[i].end());
        latency_list[i] = std::chrono::duration_cast<std::chrono::microseconds>(
                              latest_end - earliest_start)
                              .count() /
                          1000.0f;
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
                if (res[i][master_thread[i]][j].id == groundtruth[i][g])
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