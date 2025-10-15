cd build && make -j && cd ..

thread_num=8

# L_list=(100 150 200 250 300 350 400 450 500 550 600 650 700 750 800 900 1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000 2200 2400 2600 2800 3000) #glove
# L_str=$(IFS=,; echo "${L_list[*]}")
# taskset -c 0-$((thread_num - 1)) ./build/tests/parallel_search /SSD/Glove200/base.1m.fbin \
# /SSD/Glove200/query.fbin \
# /SSD/models/nsg/glove200.L128.R16.C128.nsg \
# "$L_str" 100 "$thread_num" /SSD/Glove200/gt.query.top100.bin "glove" | tee -a /home/mqj/proj/ANNSLib/experiments/low_q_graph/glove_8t_ours_l128.csv


# L_list=(100 150 200 250 300 350 400 450 500 550 600 700 800 900 1000 1200 1400 1600)
# L_list=(1800 2000 2200 2400 2600 2800)
# L_str=$(IFS=,; echo "${L_list[*]}")
# taskset -c 0-$((thread_num - 1)) ./build/tests/parallel_search /SSD/LAION/LAION_base_imgemb_10M.fbin \
# /SSD/LAION/LAION_test_query_textemb_50k.fbin \
# /SSD/models/nsg/laion.L200.R32.C200.nsg \
# "$L_str" 100 "$thread_num" /SSD/LAION/gt.test50K.bin "laion" | tee -a /home/mqj/proj/ANNSLib/experiments/low_q_graph/laion_8t_ours_l200.csv

# L_list=(100 150 200 250 300 350 400 450 500 550 600 700 800 900 1000 1200 1400 1600 1800 2000 2200 2400 2600 2800) # WebVid
L_list=(3300 3400 3500 3600) # WebVid

L_str=$(IFS=,; echo "${L_list[*]}")
taskset -c 0-$((thread_num - 1)) ./build/tests/parallel_search /SSD/WebVid/webvid.base.2.5M.fbin \
/SSD/WebVid/webvid.query.10k.fbin \
/SSD/models/nsg/webvid.L200.R32.C200.nsg \
"$L_str" 100 "$thread_num" /SSD/WebVid/gt.query.top100.bin "webvid" | tee -a /home/mqj/proj/ANNSLib/experiments/low_q_graph/webvid_8t_ours_l200.csv
