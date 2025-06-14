cd build && make -j && cd ..

L_list=(300 400 500 500 700 800 900 1000 1100 1200 1400 1600 1800 2000) #  

# 将L_list转换为逗号分隔的字符串
L_str=$(IFS=,; echo "${L_list[*]}")

# taskset -c 0 ./build/tests/search /SSD/Text-to-Image/base.10M.fbin \
#     /SSD/Text-to-Image/query.10k.fbin \
#     /SSD/models/nsg/t2i10m.L2000.R64.C200.nsg \
#     "$L_str" 100 /SSD/Text-to-Image/gt.10K_10M.bin

taskset -c 0 ./build/tests/search /SSD/LAION/LAION_base_imgemb_10M.fbin \
/SSD/LAION/LAION_test_query_textemb_50k.fbin \
/SSD/models/nsg/laion.L2000.R64.C200.nsg \
"$L_str" 100 /SSD/LAION/gt.test50K.bin

# taskset -c 0 ./build/tests/search /SSD/MainSearch/base_100k.fbin \
# /SSD/MainSearch/test_5k.fbin \
# /SSD/models/nsg/mainsearch100k.L2000.R64.C2000.nsg \
# "$L_str" 100 /SSD/MainSearch/gt.test5k_base100k.bin

# taskset -c 0 ./build/tests/search /SSD/MainSearch/base.fbin \
# /SSD/MainSearch/query_test_unique.fbin \
# /SSD/models/nsg/mainsearch.L2000.R64.C2000.nsg \
# "$L_str" 100 /SSD/MainSearch/gt.test_unique.bin

# for vtune profiling
# /SSD/LAION/LAION_base_imgemb_10M.fbin /SSD/LAION/LAION_test_query_textemb_50k.fbin /SSD/models/nsg/laion.L2000.R64.C200.nsg 500 100 /SSD/LAION/gt.test50K.bin