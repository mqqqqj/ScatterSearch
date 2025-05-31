#pragma once
#include <vector>
#include <cstring>

#ifndef UTIL_H
#define UTIL_H



void load_fvecs(char *filename, float *&data, unsigned &num, unsigned &dim);
void load_fbin(char *filename, float *&data, unsigned &num, unsigned &dim);
void load_groundtruth(char *filename, std::vector<std::vector<unsigned>> &groundtruth);
void save_result(char *filename, std::vector<std::vector<unsigned>> &results);

#endif