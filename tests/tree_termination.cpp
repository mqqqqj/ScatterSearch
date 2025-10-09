#include <annsearch.h>
#include <util.h>
#include <map>
#include <sys/resource.h>
#include <pthread.h>
#include <vector>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <cmath>
#include <thread>
#include <chrono>
#include <queue>

// 全局互斥锁，用于保护标准输出
std::mutex cout_mutex;

// ./build/tests/tree_termination /SSD/LAION/LAION_base_imgemb_10M.fbin \
/SSD/LAION/LAION_test_query_textemb_50k.fbin \
/SSD/models/nsg/laion.L2000.R64.C2000.nsg \
200 100 128 /SSD/LAION/gt.test50K.bin

struct WorkItem
{
    int start_query_idx;
    int num_tasks;
};

// 线程参数结构体
struct ThreadParam
{
    int thread_id;
    int num_total_threads;
    ANNSearch *engine;
    const float *query_load;
    int K;
    int L;
    unsigned dim;
    unsigned points_num;

    // 共享资源
    std::vector<std::queue<WorkItem>> *work_queues;
    std::vector<std::mutex> *queue_mutexes;
    std::vector<std::condition_variable> *cvs;
    std::vector<int> *parents;
    std::vector<std::atomic<int>> *credits;
    std::vector<std::vector<Neighbor>> *results;
    std::atomic<bool> *task_done;
};

void distribute_and_execute(ThreadParam *param, WorkItem work);

void *search_thread_wrapper(void *arg)
{
    ThreadParam *param = static_cast<ThreadParam *>(arg);
    int thread_id = param->thread_id;

    std::unique_lock<std::mutex> lock(param->queue_mutexes->at(thread_id));
    param->cvs->at(thread_id).wait(lock, [&]
                                   { return !param->work_queues->at(thread_id).empty(); });

    WorkItem work = param->work_queues->at(thread_id).front();
    param->work_queues->at(thread_id).pop();
    lock.unlock();

    distribute_and_execute(param, work);
    return nullptr;
}

void distribute_and_execute(ThreadParam *param, WorkItem work)
{
    int thread_id = param->thread_id;
    int start_idx = work.start_query_idx;
    int num_tasks = work.num_tasks;

    if (num_tasks > 1)
    {
        int half_tasks = num_tasks / 2;
        int child_id = thread_id + half_tasks;

        param->parents->at(child_id) = thread_id;

        {
            std::lock_guard<std::mutex> guard(cout_mutex);
            std::cout << "[Assign] Thread " << thread_id << " assigns " << half_tasks << " tasks (queries " << start_idx + half_tasks << " to " << start_idx + num_tasks - 1 << ") to Thread " << child_id << "." << std::endl;
        }

        WorkItem child_work = {start_idx + half_tasks, half_tasks};
        {
            std::lock_guard<std::mutex> child_lock(param->queue_mutexes->at(child_id));
            param->work_queues->at(child_id).push(child_work);
        }
        param->cvs->at(child_id).notify_one();

        // 递归处理自己的任务
        WorkItem my_work = {start_idx, half_tasks};
        distribute_and_execute(param, my_work);

        // 自己这边的子树完成后，等待子节点那边的子树完成
        while (param->credits->at(thread_id).load() < half_tasks)
        {
            std::this_thread::yield();
        }

        // 将收到的所有权重归还给父节点
        int parent_id = param->parents->at(thread_id);
        if (parent_id != -1)
        {
            {
                std::lock_guard<std::mutex> guard(cout_mutex);
                std::cout << "[Merge] Thread " << thread_id << " (internal) received all child weights. Returning " << num_tasks << " weights to Parent " << parent_id << "." << std::endl;
            }
            param->credits->at(parent_id).fetch_add(num_tasks);
        }
        return;
    }

    // Base case: num_tasks == 1, 执行搜索任务
    boost::dynamic_bitset<> flags(param->points_num, 0);
    param->engine->SearchArraySimulationForPipeline(
        param->query_load + param->dim * start_idx,
        start_idx,
        thread_id,
        param->num_total_threads,
        param->K,
        param->L,
        flags,
        param->results->at(start_idx));

    param->task_done[thread_id] = true;

    // 归还自己的权重 (1)
    int parent_id = param->parents->at(thread_id);
    if (parent_id != -1)
    {
        {
            std::lock_guard<std::mutex> guard(cout_mutex);
            std::cout << "[Merge] Thread " << thread_id << " (leaf) finished query " << start_idx << ". Returning 1 weight to Parent " << parent_id << "." << std::endl;
        }
        param->credits->at(parent_id).fetch_add(1);
    }
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
        std::cout << argv[0] << " data_file query_file nsg_path search_L search_K num_threads gt_path" << std::endl;
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

    if (num_threads <= 0 || (num_threads & (num_threads - 1)) != 0)
    {
        std::cout << "Number of threads must be a power of 2." << std::endl;
        exit(-1);
    }
    if (query_num < num_threads)
    {
        std::cout << "Not enough queries in the query file. Need at least " << num_threads << std::endl;
        exit(-1);
    }

    std::vector<std::vector<unsigned>> groundtruth;
    load_groundtruth(argv[7], groundtruth);
    std::cout << "Groundtruth loaded" << std::endl;

    if (L < K)
    {
        std::cout << "search_L cannot be smaller than search_K!" << std::endl;
        exit(-1);
    }

    ANNSearch engine(dim, points_num, data_load, INNER_PRODUCT);
    engine.LoadGraph(argv[3]);
    engine.LoadGroundtruth(argv[7]);

    // ---- New termination algorithm implementation ----

    std::vector<std::queue<WorkItem>> work_queues(num_threads);
    std::vector<std::mutex> queue_mutexes(num_threads);
    std::vector<std::condition_variable> cvs(num_threads);
    std::vector<int> parents(num_threads, -1);
    std::vector<std::atomic<int>> credits(num_threads);
    for (int i = 0; i < num_threads; ++i)
        credits[i] = 0;

    std::vector<std::vector<Neighbor>> results(num_threads, std::vector<Neighbor>(K));
    std::atomic<bool> *task_done = new std::atomic<bool>[num_threads];
    for (int i = 0; i < num_threads; ++i)
        task_done[i] = false;

    std::vector<ThreadParam> thread_params(num_threads);
    std::vector<pthread_t> threads(num_threads);

    for (int i = 0; i < num_threads; i++)
    {
        thread_params[i] = {
            i, num_threads, &engine, query_load, K, L, dim, points_num,
            &work_queues, &queue_mutexes, &cvs, &parents, &credits, &results, task_done};
    }

    for (int i = 0; i < num_threads; i++)
    {
        pthread_create(&threads[i], nullptr, search_thread_wrapper, &thread_params[i]);
    }

    // Kick off the process
    {
        std::lock_guard<std::mutex> lock(queue_mutexes[0]);
        work_queues[0].push({0, num_threads});
    }
    cvs[0].notify_one();

    // Wait for termination
    while (!task_done[0].load() || credits[0].load() < num_threads - 1)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::cout << "Termination condition met." << std::endl;

    // Join threads
    for (int i = 0; i < num_threads; i++)
    {
        pthread_join(threads[i], nullptr);
    }

    delete[] task_done;

    std::cout << "All searches are complete." << std::endl;

    return 0;
}