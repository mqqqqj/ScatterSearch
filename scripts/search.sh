cd build && make -j && cd ..

L_list=(500) # 100 150 200 250 300 350 400 450 

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

# for vtune profiling
# /SSD/LAION/LAION_base_imgemb_10M.fbin /SSD/LAION/LAION_test_query_textemb_50k.fbin /SSD/models/nsg/laion.L2000.R64.C200.nsg 500 100 /SSD/LAION/gt.test50K.bin