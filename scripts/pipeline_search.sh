cd build && make -j && cd ..

# taskset -c 0-7 ./build/tests/pipeline_search /SSD/Text-to-Image/base.10M.fbin \
# /SSD/Text-to-Image/query.10k.fbin \
# /SSD/models/nsg/t2i10m.L2000.R64.C2000.nsg \
# 400 100 8 /SSD/Text-to-Image/gt.10K_10M.bin

taskset -c 0-7 ./build/tests/pipeline_search /SSD/MainSearch/base.fbin \
/SSD/MainSearch/query_test_unique.fbin \
/SSD/models/nsg/mainsearch.L2000.R64.C2000.nsg \
350 100 8 /SSD/MainSearch/gt.test_unique.bin