cd build && make -j && cd ..

# L_list=(100 150 200 250 300 350 400 450 500)
# L_list=(100 150 200 250 300 350 400 450 500 550 600)
# L_list=(100 120 150 170 200 220 250 270 300)

# L_list=(400)

# for deltastepping and edge-wise
# L_list=(100 150 200 250 300 350 400 450 500 550 600 1000 1100 1700 1800 1900 2800 3000)
L_list=(1200 1300 1400 1500 1600 1800 1900 2100 2300 2500 2700 2900 3000)

thread_num=8

L_str=$(IFS=,; echo "${L_list[*]}")

# taskset -c 0-3 ./build/tests/parallel_search /SSD/MainSearch/base.fbin \
# /SSD/MainSearch/query_test_unique.fbin \
# /SSD/models/nsg/mainsearch.L2000.R64.C2000.nsg \
# "$L_str" 100 4 /SSD/MainSearch/gt.test_unique.bin "tmp_mainsearch"

# taskset -c 0-7 ./build/tests/parallel_search /SSD/MainSearch/base_100k.fbin \
# /SSD/MainSearch/test_5k.fbin \
# /SSD/models/nsg/mainsearch100k.L2000.R64.C2000.nsg \
# "$L_str" 100 8 /SSD/MainSearch/gt.test5k_base100k.bin

taskset -c 0-$((thread_num - 1)) ./build/tests/parallel_search /SSD/Text-to-Image/base.10M.fbin \
/SSD/Text-to-Image/query.10k.fbin \
/SSD/models/nsg/t2i10m.L2000.R64.C2000.nsg \
"$L_str" 100 "$thread_num" /SSD/Text-to-Image/gt.10K_10M.bin "t2i" | tee -a /home/mqj/proj/ANNSLib/plot/t2i_edgewise_8t.csv

# taskset -c 0-$((thread_num - 1)) ./build/tests/parallel_search /SSD/LAION/LAION_base_imgemb_10M.fbin \
# /SSD/LAION/LAION_test_query_textemb_50k.fbin \
# /SSD/models/nsg/laion.L2000.R64.C2000.nsg \
# "$L_str" 100 "$thread_num" /SSD/LAION/gt.test50K.bin "laion" | tee -a /home/mqj/proj/ANNSLib/plot/laion_8t_no_sync_rand_ep.csv

# taskset -c 0-$((thread_num - 1)) ./build/tests/parallel_search /SSD/DEEP10M/base.fbin \
# /SSD/DEEP10M/query.fbin \
# /SSD/models/nsg/deep10m.L2000.R64.C2000.nsg \
# "$L_str" 100 "$thread_num" /SSD/DEEP10M/gt.query.top100.bin "deep"

# taskset -c 0-7 ./build/tests/parallel_search /SSD/WebVid/webvid.base.2.5M.fbin \
# /SSD/WebVid/webvid.query.10k.fbin \
# /SSD/models/nsg/webvid.L2000.R64.C2000.nsg \
# "$L_str" 100 8 /SSD/WebVid/gt.query.top100.bin "webvid_noet" | tee -a /home/mqj/proj/ANNSLib/plot/webvid_8t_no_sync.csv

# taskset -c 0-$((thread_num - 1)) ./build/tests/parallel_search /SSD/Glove200/base.1m.fbin \
# /SSD/Glove200/query.fbin \
# /SSD/models/nsg/glove200.L2000.R64.C200.nsg \
# "$L_str" 100 "$thread_num" /SSD/Glove200/gt.query.top100.bin "glove_et"

# /SSD/LAION/LAION_base_imgemb_10M.fbin /SSD/LAION/LAION_test_query_textemb_50k.fbin /SSD/models/nsg/laion.L2000.R64.C2000.nsg "700" 100 8 /SSD/LAION/gt.test50K.bin "laion_et"