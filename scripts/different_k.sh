cd build && make -j && cd ..

thread_num=8

#100 150 200 250 300 350 400 450 

L_list=(10 20 30 40 50 60 70 80 90 100 150 200 250 300 350 400 450)

# L_list=(50 55 60 65 70 75 80 85 90 95 100 110 120 130 140 150 180 200 250 300 350 400 500 600 800 1000)

L_str=$(IFS=,; echo "${L_list[*]}")
taskset -c 0-$((thread_num - 1)) ./build/tests/parallel_search /SSD/MainSearch/base.fbin \
/SSD/MainSearch/query_test_unique.fbin \
/SSD/models/nsg/mainsearch.L2000.R64.C2000.nsg \
"$L_str" 500 "$thread_num" /SSD/MainSearch/gt.test_unique_top500.bin "mainsearch" | tee -a /home/mqj/proj/ANNSLib/experiments/mainsearch_8t_ours_k500.csv