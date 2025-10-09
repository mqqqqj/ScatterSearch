#include <annsearch.h>
#include <util.h>
#include <map>
#include <sys/resource.h>
#include <chrono>
#include <numeric>
#include <ThreadPool.h>
#include <sstream>
#include <queue>              // 新增：引入队列
#include <mutex>              // 新增：引入互斥锁
#include <condition_variable> // 新增：引入条件变量
#include <deque>              // 新增：引入双端队列
#include <thread>             // 新增：引入线程
#include <atomic>             // 新增：引入原子操作

int main(int argc, char **argv)
{
    // 禁用core文件生成
    struct rlimit limit;
    limit.rlim_cur = 0;
    limit.rlim_max = 0;
    setrlimit(RLIMIT_CORE, &limit);
    srand(time(0));
    if (argc != 10)
    {
        std::cout << argv[0]
                  << " data_file query_file nsg_path search_L_list search_K num_threads gt_path dataset_name request_rate"
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
    float request_rate = atof(argv[9]); // 每秒查询数
    std::vector<std::vector<unsigned>> groundtruth;
    load_groundtruth(argv[7], groundtruth);
    std::string dataset_name = argv[8];
    if (query_num > 1000)
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
    std::vector<TestResult> test_results;
    std::cout << "L,Throughput,latency,recall,p95recall,p99recall" << std::endl;

    for (int L : L_list)
    {
        // 新增：请求队列和相关同步工具
        std::deque<std::pair<unsigned, float *>> request_queue; // 使用 std::deque 替代 std::queue
        std::mutex queue_mutex;
        std::condition_variable queue_cv;
        bool done = false;
        std::vector<std::chrono::high_resolution_clock::time_point> query_receive_time(query_num);
        std::vector<std::vector<std::chrono::high_resolution_clock::time_point>> query_search_start_times(query_num, std::vector<std::chrono::high_resolution_clock::time_point>(num_threads));
        std::vector<std::vector<std::chrono::high_resolution_clock::time_point>> query_search_end_times(query_num, std::vector<std::chrono::high_resolution_clock::time_point>(num_threads));
        unsigned batch_size = 1; // batch_size可根据需求调整
        std::thread request_generator([&]()
                                      {
            unsigned i = 0;
            while (i < query_num)
            {
                std::vector<std::pair<unsigned, float *>> batch;
                for (unsigned j = 0; j < batch_size && i < query_num; ++j, ++i)
                {
                    float *query_ptr = query_load + (size_t)i * dim;
                    batch.push_back({i, query_ptr});
                }

                {
                    std::unique_lock<std::mutex> lock(queue_mutex);
                    for (auto &item : batch)
                    {
                        request_queue.push_back(item); // 使用 push_back
                        query_receive_time[item.first] = std::chrono::high_resolution_clock::now();
                    }
                    queue_cv.notify_all(); // 通知所有等待线程有新任务
                }

                if (i < query_num)
                {
                    std::this_thread::sleep_for(std::chrono::duration<float, std::milli>(1000.0 / request_rate));
                }
            }
            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                done = true;
                queue_cv.notify_all();
            } });
        std::vector<float> latency_list(query_num); // 单位：毫秒
        std::vector<float> queueing_time_list(query_num);
        std::vector<float> processing_time_list(query_num);
        ThreadPool pool(num_threads);
        std::vector<std::future<void>> futures;
        std::vector<std::vector<std::vector<Neighbor>>> res(query_num, std::vector<std::vector<Neighbor>>(num_threads));
        // for early termination
        int master_thread[query_num];
        memset(master_thread, -1, sizeof(int) * query_num);
        std::atomic<int> finish_num[query_num];
        int best_thread_ndc[query_num];
        memset(best_thread_ndc, 0, sizeof(int) * query_num);
        std::atomic<float> best_dist[query_num];
        std::atomic<int> best_thread_id[query_num];
        std::atomic<bool> best_thread_finish[query_num];
        std::vector<int> best_thread_finish_order(query_num);
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
        std::vector<std::vector<std::vector<Neighbor>>> retsets(flag_pool_size, std::vector<std::vector<Neighbor>>(num_threads, std::vector<Neighbor>(L + 1)));
        std::vector<std::vector<bool>> is_reach_20_hop(flag_pool_size, std::vector<bool>(num_threads));
        while (true)
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            queue_cv.wait(lock, [&]()
                          { return !request_queue.empty() || done; });
            if (request_queue.empty() && done)
                break;
            auto item = request_queue.front(); // 使用 front
            unsigned query_id = item.first;
            float *query_ptr = item.second;
            request_queue.pop_front(); // 使用 pop_front
            lock.unlock();
            for (int j = 0; j < num_threads; j++)
            {
                futures.push_back(pool.enqueue([&, query_id, j, query_ptr]()
                                               {
                                                unsigned i = query_id;
                        query_search_start_times[i][j] = std::chrono::high_resolution_clock::now();
                        int flag_idx = i % flag_pool_size;
                        int local_ndc = 0;
                        engine.SearchArraySimulationForPipelineWithET(query_load + (size_t)i * dim, i, j, K, L, flags[flag_idx], best_thread_finish[i], retsets[flag_idx], is_reach_20_hop[flag_idx], best_dist[i], best_thread_id[i], local_ndc, best_thread_ndc[i], res[i][j]);
                        finish_num[i] ++;
                        // if(finish_num[i] >= num_threads / 2)
                        //     best_thread_finish[i] = true;
                        if(best_thread_id[i] == j)
                        {
                            best_thread_finish[i] = true;
                            best_thread_ndc[i] = local_ndc;
                            best_thread_finish_order[i] = finish_num[i];
                        }
                        if(finish_num[i] == num_threads)
                        {
                            flags[flag_idx].reset();
                            // 把retsets[flag_idx]中的元素清空
                            for(int t = 0; t < num_threads; t++) {
                                retsets[flag_idx][t].clear();
                                retsets[flag_idx][t].resize(L + 1);
                                is_reach_20_hop[flag_idx][num_threads] = false;
                            }
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
                        query_search_end_times[i][j] = std::chrono::high_resolution_clock::now(); }));
            }
        }

        // std::cout << "Waiting for all threads to finish" << std::endl;
        for (size_t i = 0; i < futures.size(); i++)
        {
            futures[i].get();
        }
        // std::cout << "All threads finished" << std::endl;

        // 等待请求生成线程完成
        request_generator.join();

        auto e = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = e - s;
        std::vector<float> start_time_diffs(query_num);
        for (unsigned i = 0; i < query_num; i++)
        {
            auto earliest_start = *std::min_element(query_search_start_times[i].begin(), query_search_start_times[i].end());
            auto latest_start = *std::max_element(query_search_start_times[i].begin(), query_search_start_times[i].end());
            auto latest_end = *std::max_element(query_search_end_times[i].begin(), query_search_end_times[i].end());
            start_time_diffs[i] = std::chrono::duration_cast<std::chrono::microseconds>(
                                      latest_start - earliest_start)
                                      .count() /
                                  1000.0f;
            processing_time_list[i] = std::chrono::duration_cast<std::chrono::microseconds>(
                                          latest_end - earliest_start)
                                          .count() /
                                      1000.0f;
            queueing_time_list[i] = std::chrono::duration_cast<std::chrono::microseconds>(
                                        earliest_start - query_receive_time[i])
                                        .count() /
                                    1000.0f;
            latency_list[i] = std::chrono::duration_cast<std::chrono::microseconds>(
                                  latest_end - query_receive_time[i])
                                  .count() /
                              1000.0f;
        }
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
        std::sort(recalls.begin(), recalls.end());
        float p95_recall = recalls[recalls.size() * 0.05];
        float p99_recall = recalls[recalls.size() * 0.01];
        float qps = query_num / diff.count();
        TestResult tr{L, qps, avg_latency, avg_recall, p95_recall, p99_recall};
        test_results.push_back(tr);
        std::cout << tr.L << "," << tr.throughput << "," << tr.latency << "," << tr.recall << "," << tr.p95_recall << "," << tr.p99_recall << std::endl;
    }
    // std::string save_path = "./results/" + dataset_name + "_online_pipeline_" + std::to_string(num_threads) + "t.csv";
    // save_results(test_results, save_path);
    return 0;
}