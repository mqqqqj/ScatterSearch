cd build && make -j && cd ..

# taskset -c 0-7 ./build/tests/parallel_search /SSD/MainSearch/base.fbin \
# /SSD/MainSearch/query_test_unique.fbin \
# /SSD/models/nsg/mainsearch.L2000.R64.C2000.nsg \
# 450 100 8 /SSD/MainSearch/gt.test_unique.bin

# taskset -c 0-7 ./build/tests/parallel_search /SSD/MainSearch/base_100k.fbin \
# /SSD/MainSearch/test_5k.fbin \
# /SSD/models/nsg/mainsearch100k.L2000.R64.C2000.nsg \
# 100 100 8 /SSD/MainSearch/gt.test5k_base100k.bin

# taskset -c 0-7 ./build/tests/parallel_search /SSD/Text-to-Image/base.10M.fbin \
# /SSD/Text-to-Image/query.10k.fbin \
# /SSD/models/nsg/t2i10m.L2000.R64.C2000.nsg \
# 300 100 8 /SSD/Text-to-Image/gt.10K_10M.bin

taskset -c 0-7 ./build/tests/parallel_search /SSD/LAION/LAION_base_imgemb_10M.fbin \
/SSD/LAION/LAION_test_query_textemb_50k.fbin \
/SSD/models/nsg/laion.L2000.R64.C2000.nsg \
350 100 8 /SSD/LAION/gt.test50K.bin