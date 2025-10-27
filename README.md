## Introduction

ScatterSearch is a high-performance intra-query parallel graph-based ANNS algorithm.

## Prerequisites
+ AVX2
+ G++ 13.2.0 with OpenMP
+ CMAKE 3.9+
+ BOOST 1.55+

## Dataset

Make sure you have the base and query dataset in the *.fbin format. 

You can download the Glove200 dataset at [this link](https://nlp.stanford.edu/projects/glove/).

## Quick start
### Compile

```
mkdir build && cd build
cmake ..
make -j
```


### Run
Remember to replace the file paths in the script with your actual paths.
```
bash scripts/parallel_search.sh
```
