cd build && make -j && cd ..

L_list=(100 150 200 250 300 350 400 450 500)
# L_list=(100 150 200 250 300 400 500)
# L_list=(200)

L_str=$(IFS=,; echo "${L_list[*]}")

# taskset -c 0-7 ./build/tests/parallel_search /SSD/MainSearch/base.fbin \
# /SSD/MainSearch/query_test_unique.fbin \
# /SSD/models/nsg/mainsearch.L2000.R64.C2000.nsg \
# "$L_str" 100 8 /SSD/MainSearch/gt.test_unique.bin "mainsearch_noet"

# taskset -c 0-7 ./build/tests/parallel_search /SSD/MainSearch/base_100k.fbin \
# /SSD/MainSearch/test_5k.fbin \
# /SSD/models/nsg/mainsearch100k.L2000.R64.C2000.nsg \
# "$L_str" 100 8 /SSD/MainSearch/gt.test5k_base100k.bin

# taskset -c 0-7 ./build/tests/parallel_search /SSD/Text-to-Image/base.10M.fbin \
# /SSD/Text-to-Image/query.10k.fbin \
# /SSD/models/nsg/t2i10m.L2000.R64.C2000.nsg \
# "$L_str" 100 8 /SSD/Text-to-Image/gt.10K_10M.bin "t2i_noet"

# taskset -c 0-7 ./build/tests/parallel_search /SSD/LAION/LAION_base_imgemb_10M.fbin \
# /SSD/LAION/LAION_test_query_textemb_50k.fbin \
# /SSD/models/nsg/laion.L2000.R64.C2000.nsg \
# "$L_str" 100 8 /SSD/LAION/gt.test50K.bin "tmp_laion"

# taskset -c 0-7 ./build/tests/parallel_search /SSD/DEEP10M/base.fbin \
# /SSD/DEEP10M/query.fbin \
# /SSD/models/nsg/deep10m.L2000.R64.C2000.nsg \
# "$L_str" 100 8 /SSD/DEEP10M/gt.query.top100.bin "deep"

# taskset -c 0-7 ./build/tests/parallel_search /SSD/WebVid/webvid.base.2.5M.fbin \
# /SSD/WebVid/webvid.query.10k.fbin \
# /SSD/models/nsg/webvid.L2000.R64.C2000.nsg \
# "$L_str" 100 8 /SSD/WebVid/gt.query.top100.bin "webvid"

taskset -c 0-7 ./build/tests/parallel_search /SSD/Glove200/base.1m.fbin \
/SSD/Glove200/query.fbin \
/SSD/models/nsg/glove200.L2000.R64.C200.nsg \
"$L_str" 100 8 /SSD/Glove200/gt.query.top100.bin "glove_noet"