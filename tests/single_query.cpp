#include <annsearch.h>
#include <util.h>
#include <map>
#include <sys/resource.h>
#include <chrono>
#include <numeric>
#include <unordered_set>
#include <fstream>

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
    float avg_recall = 0;
    float recall_list[100];
#ifdef COLLECT_SEARCH_TREE
    for (int i = 0; i < num_threads; ++i)
    {
        engine.search_tree[i] = {}; // 显式初始化每个 worker_id
    }
#endif
#ifdef COLLECT_VISITED_ID
    engine.visited_lists.clear();
    engine.visited_lists.resize(num_threads);
#endif
    boost::dynamic_bitset<> flags{points_num, 0};
    std::vector<unsigned> tmp(K);
    engine.MultiThreadSearchArraySimulationWithET(query_load + (size_t)query_id * dim, query_id, K, L, num_threads, flags, tmp);
    // engine.MultiThreadSearchArraySimulation(query_load + (size_t)query_id * dim, query_id, K, L, num_threads, flags, tmp);
    // 基于tmp和groundtruth计算recall
    std::unordered_set<unsigned> gt_set(groundtruth[query_id].begin(), groundtruth[query_id].begin() + K);
    unsigned hit = 0;
    for (unsigned id : tmp)
    {
        if (gt_set.count(id))
            hit++;
    }
    float recall = static_cast<float>(hit) / K;
    avg_recall += recall;
    recall_list[query_id] = recall;
#ifdef COLLECT_SEARCH_TREE
    std::string tree_filename = "/home/mqj/proj/search_tree_vis/output/laion/noet_query" + std::to_string(query_id) + "_" + std::to_string(num_threads) + "threads_recall_" + std::to_string(recall) + ".txt";
    std::ofstream tree_file(tree_filename);
    for (int i = 0; i < num_threads; i++)
    {
        std::cout << "thread calculate num: " << engine.search_tree[i].size() << std::endl;
        for (size_t j = 0; j < engine.search_tree[i].size(); j++)
        {
            tree_file << i << "," << engine.search_tree[i][j].first << "," << engine.search_tree[i][j].second << std::endl;
        }
    }
    tree_file.close();
#endif
#ifdef COLLECT_VISITED_ID
    std::unordered_set<unsigned> visited_node_set;
    for (int t = 0; t < num_threads; t++)
    {
        std::string filename = "/home/mqj/proj/demos/t-sne/laion10k_qid" + std::to_string(query_id) + "_nosync_thread_" + std::to_string(t) + "_visited_ids.txt";
        std::ofstream outfile(filename);
        if (outfile.is_open())
        {
            for (const auto &id : engine.visited_lists[t])
            {
                outfile << id << std::endl;
            }
            outfile.close();
            std::cout << "已成功将 " << engine.visited_lists[t].size() << " 个访问节点ID写入文件: " << filename << std::endl;
        }
        else
        {
            std::cerr << "无法打开文件进行写入: " << filename << std::endl;
        }
    }
#endif
    std::cout << "平均召回率 " << avg_recall << std::endl;
    std::cout << "平均计算量 " << (float)engine.dist_comps << std::endl;

    // 将recall_list写入文件
    // std::string recall_filename = "/home/mqj/proj/tmp_results/laion_nosync_" + std::to_string(num_threads) + "threads_recall.txt";
    // std::ofstream recall_file(recall_filename);
    // if (recall_file.is_open())
    // {
    //     for (int i = 0; i < 100; i++)
    //     {
    //         recall_file << recall_list[i] << std::endl;
    //     }
    //     recall_file.close();
    //     std::cout << "已成功将100个query的recall值写入文件: " << recall_filename << std::endl;
    // }
    // else
    // {
    //     std::cerr << "无法打开recall文件进行写入: " << recall_filename << std::endl;
    // }

    return 0;
}