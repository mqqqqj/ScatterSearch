cd build && make -j && cd ..

L_list=(100 150 200 250 300 350 400 450 500)

L_str=$(IFS=,; echo "${L_list[*]}")

# taskset -c 0-7 ./build/tests/pthread_search /SSD/MainSearch/base.fbin \
# /SSD/MainSearch/query_test_unique.fbin \
# /SSD/models/nsg/mainsearch.L2000.R64.C2000.nsg \
# "$L_str" 100 8 /SSD/MainSearch/gt.test_unique.bin

taskset -c 0-7 ./build/tests/pthread_search /SSD/LAION/LAION_base_imgemb_10M.fbin \
/SSD/LAION/LAION_test_query_textemb_50k.fbin \
/SSD/models/nsg/laion.L2000.R64.C2000.nsg \
"$L_str" 100 8 /SSD/LAION/gt.test50K.bin