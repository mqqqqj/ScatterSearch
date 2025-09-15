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
#include <condition_variable>
#include <thread>

struct SearchContext
{
    const float *query;
    int K, L, dimension;
    const std::vector<std::vector<unsigned>> *graph;
    const float *base_data;
    unsigned default_ep;
    float (*distance_func)(const float *, const float *, size_t);
    boost::dynamic_bitset<> *global_flags;
    std::vector<unsigned> result_indices; // 输出结果
    std::vector<std::vector<Neighbor>> retsets;
    std::vector<unsigned> retset_sizes;
    std::atomic<bool> sync_triggered;
    std::atomic<int> finished_threads;
    int num_threads;
    std::mutex mtx;
    std::condition_variable cv;
    bool merge_completed;
    // 构造函数初始化
    SearchContext(
        const float *q, int k, int l,
        ANNSearch *engine,
        boost::dynamic_bitset<> *flags,
        int n_threads)
        : query(q), K(k), L(l), dimension(engine->dimension), graph(&engine->graph), base_data(engine->base_data),
          default_ep(engine->default_ep), distance_func(engine->distance_func), global_flags(flags),
          sync_triggered(false), finished_threads(0), num_threads(n_threads),
          merge_completed(false)
    {
        retsets.resize(n_threads);
        retset_sizes.resize(n_threads, 0);
        result_indices.resize(K);
    }
};

struct ThreadArg
{
    SearchContext *ctx;
    int tid;
};
void *search_thread_worker(void *arg)
{
    ThreadArg *t_arg = static_cast<ThreadArg *>(arg);
    SearchContext *ctx = t_arg->ctx;
    const int tid = t_arg->tid;
    std::vector<Neighbor> &retset = ctx->retsets[tid];
    retset.resize(ctx->L + 1);
    unsigned &tmp_l = ctx->retset_sizes[tid];
    tmp_l = 0;
    // Step 1: 初始化种子节点
    const std::vector<unsigned> &ep_neighbors = (*ctx->graph)[ctx->default_ep];
    for (size_t j = 0; j < ep_neighbors.size(); ++j)
    {
        if (j % ctx->num_threads == tid)
        {
            unsigned id = ep_neighbors[j];
            if (ctx->global_flags->test(id))
                continue;
            ctx->global_flags->set(id);
            float dist = ctx->distance_func(
                ctx->base_data + ctx->dimension * id,
                ctx->query,
                ctx->dimension);
            retset[tmp_l++] = Neighbor(id, dist, true);
        }
    }
    std::sort(retset.begin(), retset.begin() + tmp_l);
    // Step 2: 主搜索循环（带取消检查）
    int k = 0;
    while (k < ctx->L && tmp_l < ctx->L + 1)
    {
        if (ctx->sync_triggered.load(std::memory_order_acquire))
        {
            break; // 被别人触发同步了，立即退出
        }
        if (k < tmp_l && retset[k].unexplored)
        {
            retset[k].unexplored = false;
            unsigned n = retset[k].id;
            _mm_prefetch((*ctx->graph)[n].data(), _MM_HINT_T0);
            for (size_t m = 0; m < (*ctx->graph)[n].size(); ++m)
            {
                if (ctx->sync_triggered.load(std::memory_order_acquire))
                {
                    break; // 取消扩展邻接点
                }
                unsigned id = (*ctx->graph)[n][m];
                if (m + 1 < (*ctx->graph)[n].size())
                {
                    _mm_prefetch(ctx->base_data + ctx->dimension * (*ctx->graph)[n][m + 1], _MM_HINT_T0);
                }
                if (ctx->global_flags->test(id))
                    continue;
                ctx->global_flags->set(id);
                float dist = ctx->distance_func(ctx->query, ctx->base_data + ctx->dimension * id, ctx->dimension);
                if (dist >= retset[tmp_l - 1].distance)
                    continue;
                Neighbor nn(id, dist, true);
                int pos = InsertIntoPool(retset.data(), tmp_l, nn);
                if (tmp_l < ctx->L)
                    tmp_l++;
            }
        }
        ++k;
    }
    // 当前线程工作结束（完成或被中断）
    // Step 3: 尝试成为“同步发起者”
    if (!ctx->sync_triggered.exchange(true, std::memory_order_acq_rel))
    {
        // ✅ 我是第一个触发同步的线程 → 执行归并
        // 合并所有线程的结果到线程 0 的 retset
        for (int i = 1; i < ctx->num_threads; ++i)
        {
            for (size_t j = 0; j < ctx->retset_sizes[i]; ++j)
            {
                const Neighbor &nb = ctx->retsets[i][j];
                int pos = InsertIntoPool(ctx->retsets[0].data(), ctx->retset_sizes[0], nb);
                if (ctx->retset_sizes[0] < ctx->L)
                    ctx->retset_sizes[0]++;
                if (pos == ctx->L)
                    break;
            }
        }
        // 提取前 K 个
        std::partial_sort(
            ctx->retsets[0].begin(),
            ctx->retsets[0].begin() + ctx->K,
            ctx->retsets[0].begin() + ctx->retset_sizes[0]);
        for (int i = 0; i < ctx->K; ++i)
        {
            ctx->result_indices[i] = ctx->retsets[0][i].id;
        }
        // 设置合并完成标志
        {
            std::lock_guard<std::mutex> lock(ctx->mtx);
            ctx->merge_completed = true;
        }
        ctx->cv.notify_all(); // 唤醒所有等待线程
    }
    // Step 4: 所有线程等待同步完成（确保 result_indices 被写入）
    {
        std::unique_lock<std::mutex> lock(ctx->mtx);
        ctx->cv.wait(lock, [&]
                     { return ctx->merge_completed; });
    }
    return nullptr;
}

void MultiThreadSearchArraySimulationSync(
    const float *query,
    unsigned query_id, // 可能未使用
    int K,
    int L,
    int num_threads,
    boost::dynamic_bitset<> &flags,
    std::vector<unsigned> &indices, ANNSearch *engine)
{
    // 创建上下文
    SearchContext ctx(
        query, K, L, engine, &flags, num_threads);
    // 分配线程参数
    std::vector<pthread_t> threads(num_threads);
    std::vector<ThreadArg> args(num_threads);
    for (int i = 0; i < num_threads; ++i)
    {
        args[i] = {&ctx, i};
        pthread_create(&threads[i], nullptr, search_thread_worker, &args[i]);
    }
    // 等待所有线程完成
    for (int i = 0; i < num_threads; ++i)
    {
        pthread_join(threads[i], nullptr);
    }
    // 将结果拷贝出来
    indices = ctx.result_indices;
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
        std::vector<std::vector<unsigned>> res(query_num);
        std::vector<float> latency_list(query_num); // 单位：毫秒
        auto s = std::chrono::high_resolution_clock::now();
        for (unsigned i = 0; i < query_num; i++)
        {
            std::vector<unsigned> tmp(K);
            auto start_time = std::chrono::high_resolution_clock::now();
            MultiThreadSearchArraySimulationSync(query_load + (size_t)i * dim, i, K, L, num_threads, flags, tmp, &engine);
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
        TestResult tr{L, qps, avg_latency, avg_recall, recalls[recalls.size() * 0.05], recalls[recalls.size() * 0.01], (float)engine.dist_comps / query_num, (float)engine.hop_count / (query_num * num_threads)};
        engine.dist_comps = 0;
        engine.hop_count = 0;
        test_results.push_back(tr);
        std::cout << "L,Throughput,latency,recall,p95recall,p99recall,dist_comps,hops" << std::endl;
        std::cout << tr.L << "," << tr.throughput << "," << tr.latency << "," << tr.recall << "," << tr.p95_recall << "," << tr.p99_recall << "," << tr.dist_comps << "," << tr.hops << std::endl;
    }
    return 0;
}