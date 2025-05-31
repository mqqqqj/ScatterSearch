#include <util.h>
#include <fstream>
#include <iostream>

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
    std::cout << "data dimension: " << dim << std::endl;
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;
    num = (unsigned)(fsize / (dim + 1) / 4);
    data = new float[num * dim * sizeof(float)];

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
    std::cout << "data num: " << num << ", data dimension: " << dim << std::endl;
    data = new float[(size_t)num * (size_t)dim];
    for (size_t i = 0; i < num; i++)
    {
        in.read((char *)(data + i * dim), dim * 4);
    }
    in.close();
    std::cout << "load fbin done: " << filename << std::endl;
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
    std::cout << "nq: " << nq << ", GK: " << GK << std::endl;
    for (unsigned i = 0; i < nq; i++)
    {
        std::vector<unsigned> result(GK);
        in.read((char *)result.data(), GK * sizeof(unsigned));
        groundtruth.push_back(result);
    }
    in.close();
}

void save_result(char *filename, std::vector<std::vector<unsigned>> &results)
{
    std::ofstream out(filename, std::ios::binary | std::ios::out);

    for (unsigned i = 0; i < results.size(); i++)
    {
        unsigned GK = (unsigned)results[i].size();
        out.write((char *)&GK, sizeof(unsigned));
        out.write((char *)results[i].data(), GK * sizeof(unsigned));
    }
    out.close();
}