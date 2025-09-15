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
    int query_id;
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
    float avg_recall = 0;
    query_num = 100;
    float recall_list[query_num];
    for (query_id = 0; query_id < query_num; query_id++)
    {
        engine.dist_comps = 0;
#ifdef COLLECT_VISITED_ID
        engine.visited_lists.clear();
        engine.visited_lists.resize(num_threads);
#endif
        boost::dynamic_bitset<> flags{points_num, 0};
        std::vector<unsigned> tmp(K);
        engine.SearchArraySimulation(query_load + (size_t)query_id * dim, query_id, K, L, flags, tmp);

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
    }
    std::cout << "平均召回率 " << avg_recall / query_num << std::endl;
    std::cout << engine.hop_find_first_knn.size() << std::endl;
    std::cout << engine.hop_find_all_knn.size() << std::endl;
    float ratio = 0;
    for (size_t i = 0; i < engine.hop_find_all_knn.size(); i++)
    {
        ratio += (float)engine.hop_find_first_knn[i] / engine.hop_find_all_knn[i];
    }
    ratio /= query_num;
    std::cout << ratio << std::endl;
    // Extract dataset name from data_file path
    std::string data_file_path(argv[1]);
    size_t last_slash_pos = data_file_path.find_last_of('/');
    std::string data_file_name = (last_slash_pos != std::string::npos) ? data_file_path.substr(last_slash_pos + 1) : data_file_path;

    // Remove file extension if present
    size_t last_dot_pos = data_file_name.find_last_of('.');
    if (last_dot_pos != std::string::npos)
    {
        data_file_name = data_file_name.substr(0, last_dot_pos);
    }

    // Create output file name with dataset name
    std::string output_file_name = "./results/knn_hop_data_" + data_file_name + ".txt";
    std::ofstream output_file(output_file_name);

    if (output_file.is_open())
    {
        // Write data in two columns: first_knn hop and all_knn hop
        size_t min_size = std::min(engine.hop_find_first_knn.size(), engine.hop_find_all_knn.size());
        for (size_t i = 0; i < min_size; ++i)
        {
            output_file << engine.hop_find_first_knn[i] << " " << engine.hop_find_all_knn[i] << std::endl;
        }
        output_file.close();
        std::cout << "Hop data written to " << output_file_name << std::endl;
    }
    else
    {
        std::cerr << "Unable to open file " << output_file_name << " for writing." << std::endl;
    }

    return 0;
}
