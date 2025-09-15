#include <util.h>
#include <fstream>
#include <iostream>
#include <cstdlib>

void load_fvecs(char *filename, float *&data, unsigned &num,
                unsigned &dim)
{ // load data with sift10K pattern
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open())
    {
        std::cout << "open file error: " << filename << std::endl;
        exit(-1);
    }
    in.read((char *)&dim, 4);
#ifdef LOG
    std::cout << "data dimension: " << dim << std::endl;
#endif
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;
    num = (unsigned)(fsize / (dim + 1) / 4);
    data = new float[(size_t)num * (size_t)dim];
    // 使用 aligned_alloc 分配32字节对齐的内存
    // data = (float *)aligned_alloc(32, num * dim * sizeof(float));
    if (!data)
    {
        std::cout << "Memory allocation failed" << std::endl;
        exit(-1);
    }
    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num; i++)
    {
        in.seekg(4, std::ios::cur);
        in.read((char *)(data + i * dim), dim * 4);
    }
    in.close();
}

void load_fbin(char *filename, float *&data, unsigned &num, unsigned &dim)
{
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open())
    {
        std::cout << "open file error: " << filename << std::endl;
        exit(-1);
    }
    in.read((char *)&num, 4);
    in.read((char *)&dim, 4);
#ifdef LOG
    std::cout << "data num: " << num << ", data dimension: " << dim << std::endl;
#endif
    float *aligned_data = new float[(size_t)num * (size_t)dim + 8];
    // data = (float *)((char *)aligned_data + 1); // unalign 先转为char*偏移，再转回float*
    data = aligned_data;
    if (!data)
    {
        std::cout << "Memory allocation failed" << std::endl;
        exit(-1);
    }

    for (size_t i = 0; i < num; i++)
    {
        in.read((char *)(data + i * dim), dim * 4);
    }
    in.close();
#ifdef LOG
    std::cout << "load fbin done: " << filename << std::endl;
#endif
}

void load_groundtruth(char *filename, std::vector<std::vector<unsigned>> &groundtruth)
{
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open())
    {
        std::cout << "open file error: " << filename << std::endl;
        exit(-1);
    }
    unsigned GK, nq;
    in.read((char *)&nq, sizeof(unsigned));
    in.read((char *)&GK, sizeof(unsigned));
#ifdef LOG
    std::cout << "nq: " << nq << ", GK: " << GK << std::endl;
#endif
    for (unsigned i = 0; i < nq; i++)
    {
        std::vector<unsigned> result(GK);
        in.read((char *)result.data(), GK * sizeof(unsigned));
        groundtruth.push_back(result);
    }
    in.close();
}

void save_results(const std::vector<TestResult> &results, std::string filename)
{
    std::ofstream file(filename, std::ios::app);
    file << "L,Throughput,latency,recall,p95recall,p99recall,dist_comps,hops" << std::endl;
    for (unsigned i = 0; i < results.size(); i++)
    {
        file << results[i].L << "," << results[i].throughput << ","
             << results[i].latency << "," << results[i].recall << ","
             << results[i].p95_recall << "," << results[i].p99_recall << ","
             << results[i].dist_comps << "," << results[i].hops << std::endl;
    }
    file.close();
}