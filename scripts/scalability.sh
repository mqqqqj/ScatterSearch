cd build && make -j && cd ..

thread_num=32

L_list=(160 170)

L_str=$(IFS=,; echo "${L_list[*]}")
taskset -c 0-17,36-53 ./build/tests/parallel_search /SSD/Glove200/base.1m.fbin \
/SSD/Glove200/query.fbin \
/SSD/models/nsg/glove200.L2000.R64.C200.nsg \
"$L_str" 100 "$thread_num" /SSD/Glove200/gt.query.top100.bin "glove" | tee -a /home/mqj/proj/ANNSLib/experiments/glove_32t_ours.csv


# L_list=(100 100 125) #mainsearch

# L_str=$(IFS=,; echo "${L_list[*]}")
# taskset -c 0-17,36-53 ./build/tests/parallel_search /SSD/MainSearch/base.fbin \
# /SSD/MainSearch/query_test_unique.fbin \
# /SSD/models/nsg/mainsearch.L2000.R64.C2000.nsg \
# "$L_str" 100 "$thread_num" /SSD/MainSearch/gt.test_unique.bin "mainsearch" | tee -a /home/mqj/proj/ANNSLib/experiments/mainsearch_32t_ours.csv

