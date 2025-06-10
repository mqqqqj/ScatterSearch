cd build && make -j && cd ..

# taskset -c 0-7 ./build/tests/single_query /SSD/MainSearch/base_100k.fbin \
# /SSD/MainSearch/test_5k.fbin \
# /SSD/models/nsg/mainsearch100k.L2000.R64.C2000.nsg \
# 100 100 8 /SSD/MainSearch/gt.test5k_base100k.bin

taskset -c 0-7 ./build/tests/single_query /SSD/MainSearch/base.fbin \
/SSD/MainSearch/query_test_unique.fbin \
/SSD/models/nsg/mainsearch.L2000.R64.C2000.nsg \
200 100 8 4442 /SSD/MainSearch/gt.test_unique.bin