#include <annsearch.h>
#include <util.h>
#include <map>
#include <sys/resource.h>
#include <chrono>
#include <numeric>
#include <pthread.h>
#include <semaphore.h>
#include <mutex>
#include <set>
#include <sstream>

// 线程参数结构体
struct ThreadParam
{
    int thread_id;
    ANNSearch *engine;
    const float *query_load;
    unsigned query_num;
    int K;
    int L;
    unsigned dim;
    boost::dynamic_bitset<> *flags;
    std::vector<std::vector<std::vector<Neighbor>>> *results;
    std::vector<float> *latency_list;
    pthread_barrier_t *barrier;
};

// 线程函数
void *search_thread(void *arg)
{
    ThreadParam *param = static_cast<ThreadParam *>(arg);
    int i = param->thread_id;
    unsigned query_id = 0;
    while (true)
    {
        // 检查是否所有查询都已完成
        if (query_id >= param->query_num)
        {
            break;
        }

        // 记录开始时间
        auto start_time = std::chrono::high_resolution_clock::now();

        // 执行搜索
        std::vector<Neighbor> tmp(param->K);
        param->engine->SearchArraySimulationForPipeline(
            param->query_load + param->dim * query_id,
            query_id,
            param->K,
            param->L,
            *param->flags,
            tmp);

        // 记录结束时间
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

        // 保存结果
        (*param->latency_list)[query_id] = duration.count() / 1000.0f;
        (*param->results)[query_id][param->thread_id] = tmp;

        // 等待所有线程完成当前查询
        pthread_barrier_wait(param->barrier);
        query_id++;
        // 只有第一个线程重置 flags 并增加 current_query
        if (param->thread_id == 0)
        {
            param->flags->reset();
            if (query_id % 1000 == 999)
            {
                std::cout << "query " << query_id << " done" << std::endl;
            }
        }
    }

    return nullptr;
}

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
        std::cout << argv[0] << " data_file query_file nsg_path search_L_list search_K num_threads gt_path" << std::endl;
        std::cout << "search_L_list format: L1,L2,L3,... (comma separated values)" << std::endl;
        exit(-1);
    }

    // 加载数据
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

    if (query_num > 10000)
        query_num = 10000;

    std::vector<std::vector<unsigned>> groundtruth;
    load_groundtruth(argv[7], groundtruth);
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

    // 初始化搜索引擎
    ANNSearch engine(dim, points_num, data_load, INNER_PRODUCT);
    engine.LoadGraph(argv[3]);
    engine.LoadGroundtruth(argv[7]);

    std::vector<TestResult> test_results;
    // 对每个L值进行搜索
    for (int L : L_list)
    {
        // 初始化共享数据结构
        boost::dynamic_bitset<> flags{points_num, 0};
        std::vector<std::vector<std::vector<Neighbor>>> res(query_num, std::vector<std::vector<Neighbor>>(num_threads));
        std::vector<float> latency_list(query_num);
        std::atomic<unsigned> current_query{0};
        std::mutex query_mutex;

        // 初始化线程同步屏障
        pthread_barrier_t barrier;
        pthread_barrier_init(&barrier, nullptr, num_threads);

        // 创建线程参数
        std::vector<ThreadParam> thread_params(num_threads);
        std::vector<pthread_t> threads(num_threads);

        // 初始化线程参数
        for (int i = 0; i < num_threads; i++)
        {
            thread_params[i] = {
                i,             // thread_id
                &engine,       // engine
                query_load,    // query_load
                query_num,     // query_num
                K,             // K
                L,             // L
                dim,           // dim
                &flags,        // flags
                &res,          // results
                &latency_list, // latency_list
                &barrier,      // barrier
            };
        }

        // 记录开始时间
        auto s = std::chrono::high_resolution_clock::now();

        // 创建线程
        for (int i = 0; i < num_threads; i++)
        {
            pthread_create(&threads[i], nullptr, search_thread, &thread_params[i]);
        }

        // 等待所有线程完成
        for (int i = 0; i < num_threads; i++)
        {
            pthread_join(threads[i], nullptr);
        }

        // 合并每个查询的结果
        std::vector<std::vector<unsigned>> final_results(query_num);
        for (unsigned i = 0; i < query_num; i++)
        {
            // 找到主线程（有结果的线程）
            int master = -1;
            for (unsigned t = 0; t < num_threads; t++)
            {
                if (res[i][t].size())
                {
                    master = t;
                    break;
                }
            }

            if (master != -1)
            {
                final_results[i] = std::vector<unsigned>(K);
                // 合并其他线程的结果到主线程
                for (unsigned t = 0; t < num_threads; t++)
                {
                    if (t != master && res[i][t].size())
                    {
                        for (unsigned k = 0; k < K; k++)
                        {
                            InsertIntoPool(res[i][master].data(), K, res[i][t][k]);
                        }
                    }
                }
                // 保存最终结果
                for (int j = 0; j < K; j++)
                {
                    final_results[i][j] = res[i][master][j].id;
                }
            }
        }

        // 记录结束时间
        auto e = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = e - s;
        float qps = query_num / diff.count();

        // 清理资源
        pthread_barrier_destroy(&barrier);

        // 计算平均latency
        float accumulate_latency = std::accumulate(latency_list.begin(), latency_list.end(), 0.0f);
        float avg_latency = accumulate_latency / latency_list.size();

        // 计算合并后的召回率
        std::vector<float> recalls(query_num);
        for (unsigned i = 0; i < query_num; i++)
        {
            int correct = 0;
            for (unsigned j = 0; j < K; j++)
            {
                for (unsigned g = 0; g < K; g++)
                {
                    if (final_results[i][j] == groundtruth[i][g])
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
    std::string save_path = "./parallel_results/pthread_" + std::to_string(num_threads) + "t.csv";
    save_results(test_results, save_path);
    return 0;
}