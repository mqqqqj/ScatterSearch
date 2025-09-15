cd build && make -j && cd ..

# taskset -c 0-7 ./build/tests/hop_find_first_knn /SSD/MainSearch/base.fbin \
# /SSD/MainSearch/query_test_unique.fbin \
# /SSD/models/nsg/mainsearch.L2000.R64.C2000.nsg \
# 10000 100 4 /SSD/MainSearch/gt.test_unique.bin

taskset -c 0-3 ./build/tests/hop_find_first_knn /SSD/LAION/LAION_base_imgemb_10M.fbin \
/SSD/LAION/LAION_test_query_textemb_50k.fbin \
/SSD/models/nsg/laion.L2000.R64.C2000.nsg \
10000 100 4 /SSD/LAION/gt.test50K.bin

# taskset -c 0-3 ./build/tests/hop_find_first_knn /SSD/Text-to-Image/base.10M.fbin \
# /SSD/Text-to-Image/query.10k.fbin \
# /SSD/models/nsg/t2i10m.L2000.R64.C2000.nsg \
# 10000 100 4 /SSD/Text-to-Image/gt.10K_10M.bin

# taskset -c 0-3 ./build/tests/hop_find_first_knn /SSD/DEEP10M/base.fbin \
# /SSD/DEEP10M/query.fbin \
# /SSD/models/nsg/deep10m.L2000.R64.C2000.nsg \
# 10000 100 4 /SSD/DEEP10M/gt.query.top100.bin

# taskset -c 0-7 ./build/tests/hop_find_first_knn /SSD/Glove200/base.1m.fbin \
# /SSD/Glove200/query.fbin \
# /SSD/models/nsg/glove200.L2000.R64.C200.nsg \
# 10000 100 4 /SSD/Glove200/gt.query.top100.bin

# taskset -c 0-7 ./build/tests/hop_find_first_knn /SSD/WebVid/webvid.base.2.5M.fbin \
# /SSD/WebVid/webvid.query.10k.fbin \
# /SSD/models/nsg/webvid.L2000.R64.C2000.nsg \
# 10000 100 4 /SSD/WebVid/gt.query.top100.bin