#include <annsearch.h>
#include <util.h>
#include <map>
#include <sys/resource.h>
#include <chrono>
#include <numeric>
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
        std::cout << "request_rate: queries per second" << std::endl;
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
    if (query_num > 10000)
        query_num = 10000;
    // query_num = 100;
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
    // 对每个L值进行搜索
    for (int L : L_list)
    {
        // 新增：请求队列和相关同步工具
        std::deque<std::pair<unsigned, float *>> request_queue; // 使用 std::deque 替代 std::queue
        std::mutex queue_mutex;
        std::condition_variable queue_cv;
        bool done = false;
        std::vector<std::chrono::high_resolution_clock::time_point> query_receive_time(query_num);
        std::vector<std::chrono::high_resolution_clock::time_point> query_start_times(query_num);
        std::vector<std::chrono::high_resolution_clock::time_point> query_end_times(query_num);
        unsigned batch_size = 1; // batch_size可根据需求调整

        // 查询发送线程
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

        boost::dynamic_bitset<> flags{points_num, 0};
        std::vector<std::vector<unsigned>> res(query_num);
        std::vector<float> latency_list(query_num); // 单位：毫秒

        auto s = std::chrono::high_resolution_clock::now();

        // 处理线程
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

            // 记录查询开始时间
            query_start_times[query_id] = std::chrono::high_resolution_clock::now();

            std::vector<unsigned> tmp(K);
            // engine.MultiThreadSearchArraySimulation(query_ptr, query_id, K, L, num_threads, flags, tmp);
            engine.MultiThreadSearchArraySimulationWithET(query_ptr, query_id, K, L, num_threads, flags, tmp);
            flags.reset();

            // 记录查询结束时间
            query_end_times[query_id] = std::chrono::high_resolution_clock::now();

            // 计算延迟：从入队时间到完成时间
            auto latency = std::chrono::duration_cast<std::chrono::microseconds>(
                               query_end_times[query_id] - query_receive_time[query_id])
                               .count() /
                           1000.0f;
            latency_list[query_id] = latency;

            res[query_id] = tmp;

            if (query_id % 1000 == 999)
            {
                std::cout << "query " << query_id << " done" << std::endl;
            }
        }

        // 等待请求生成线程完成
        request_generator.join();

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
        TestResult tr{L, qps, avg_latency, avg_recall, recalls[recalls.size() * 0.05], recalls[recalls.size() * 0.01]};
        test_results.push_back(tr);
        std::cout << "L,Throughput,latency,recall,p95recall,p99recall" << std::endl;
        std::cout << tr.L << "," << tr.throughput << "," << tr.latency << "," << tr.recall << "," << tr.p95_recall << "," << tr.p99_recall << std::endl;
    }
    std::string save_path = "./results/" + dataset_name + "_online_parallel_" + std::to_string(num_threads) + "t.csv";
    save_results(test_results, save_path);
    return 0;
}