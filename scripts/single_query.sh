cd build && make -j && cd ..

# taskset -c 0-7 ./build/tests/single_query /SSD/MainSearch/base_100k.fbin \
# /SSD/MainSearch/test_5k.fbin \
# /SSD/models/nsg/mainsearch100k.L2000.R64.C2000.nsg \
# 100 100 8 /SSD/MainSearch/gt.test5k_base100k.bin

# taskset -c 0-7 ./build/tests/single_query /SSD/MainSearch/base.fbin \
# /SSD/MainSearch/query_test_unique.fbin \
# /SSD/models/nsg/mainsearch.L2000.R64.C2000.nsg \
# 200 100 8 4442 /SSD/MainSearch/gt.test_unique.bin

taskset -c 0-7 ./build/tests/single_query /SSD/LAION/LAION_base_imgemb_10M.fbin \
/SSD/LAION/LAION_test_query_textemb_50k.fbin \
/SSD/models/nsg/laion.L2000.R64.C2000.nsg \
150 100 8 100 /SSD/LAION/gt.test50K.bin

# taskset -c 0-3 ./build/tests/single_query /SSD/LAION/LAION_base_imgemb_10K.fbin \
# /SSD/LAION/LAION_test_query_textemb_50k.fbin \
# /SSD/models/nsg/laion10K.L2000.R64.C2000.nsg \
# 20 20 4 111 /SSD/LAION/gt.test50K_base10K.bin