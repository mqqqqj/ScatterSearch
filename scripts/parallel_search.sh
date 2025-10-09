cd build && make -j && cd ..
thread_num=8

# laion
# L_list=(100 150 200 250 300 350 400 450 500 550 600 700 800 900 1000 1200 1400 1600)
# L_list=(1800 2000 2200 2400 2600 2800)
# L_str=$(IFS=,; echo "${L_list[*]}")
# taskset -c 0-$((thread_num - 1)) ./build/tests/parallel_search /SSD/LAION/LAION_base_imgemb_10M.fbin \
# /SSD/LAION/LAION_test_query_textemb_50k.fbin \
# /SSD/models/nsg/laion.L2000.R64.C2000.nsg \
# "$L_str" 100 "$thread_num" /SSD/LAION/gt.test50K.bin "laion" | tee -a /home/mqj/proj/ANNSLib/experiments/laion_8t_ours.csv

#glove
# L_list=(100 150 200 250 300 350 375 400 425 450 475 500 525 550 575 600 650 700 750 800 900 1000 1100 1200)
# L_str=$(IFS=,; echo "${L_list[*]}")
# taskset -c 0-$((thread_num - 1)) ./build/tests/parallel_search /SSD/Glove200/base.1m.fbin \
# /SSD/Glove200/query.fbin \
# /SSD/models/nsg/glove200.L2000.R64.C200.nsg \
# "$L_str" 100 "$thread_num" /SSD/Glove200/gt.query.top100.bin "glove" | tee -a /home/mqj/proj/ANNSLib/experiments/glove_8t_ours.csv

#mainsearch
# L_list=(100 125 150 175 200 225 250 275 300 325 350 375 400 425 450 475 500 525 550 575 600)
L_list=(650 700 750 800 900 1000 1100 1200)
L_str=$(IFS=,; echo "${L_list[*]}")
taskset -c 0-$((thread_num - 1)) ./build/tests/parallel_search /SSD/MainSearch/base.fbin \
/SSD/MainSearch/query_test_unique.fbin \
/SSD/models/nsg/mainsearch.L2000.R64.C2000.nsg \
"$L_str" 100 "$thread_num" /SSD/MainSearch/gt.test_unique.bin "mainsearch" | tee -a /home/mqj/proj/ANNSLib/experiments/mainsearch_8t_ours.csv

#text-to-image
# L_list=(100 125 150 175 200 225 250 275 300 325 350 375 400 425 450 475 500 525 550 575 600)
# L_str=$(IFS=,; echo "${L_list[*]}")
# taskset -c 0-$((thread_num - 1)) ./build/tests/parallel_search /SSD/Text-to-Image/base.10M.fbin \
# /SSD/Text-to-Image/query.10k.fbin \
# /SSD/models/nsg/t2i10m.L2000.R64.C2000.nsg \
# "$L_str" 100 "$thread_num" /SSD/Text-to-Image/gt.10K_10M.bin "t2i" | tee -a /home/mqj/proj/ANNSLib/experiments/t2i_8t_ours.csv

# WebVid
# L_list=(100 150 200 250 300 350 400 450 500 550 600 700 800 900 1000 1200 1400 1600)
# L_list=(1800 2000 2200 2400 2600 2800)
# L_str=$(IFS=,; echo "${L_list[*]}")
# taskset -c 0-$((thread_num - 1)) ./build/tests/parallel_search /SSD/WebVid/webvid.base.2.5M.fbin \
# /SSD/WebVid/webvid.query.10k.fbin \
# /SSD/models/nsg/webvid.L2000.R64.C2000.nsg \
# "$L_str" 100 "$thread_num" /SSD/WebVid/gt.query.top100.bin "webvid" | tee -a /home/mqj/proj/ANNSLib/experiments/webvid_8t_ours.csv

#GIST
# L_list=(100 125 150 175 200 225 250 275 300 325 350 375 400 425 450 475 500 525 550 575 600)
# L_str=$(IFS=,; echo "${L_list[*]}")
# taskset -c 0-$((thread_num - 1)) ./build/tests/parallel_search /SSD/GIST1M/base.fbin \
# /SSD/GIST1M/query.fbin \
# /SSD/models/nsg/gist1m.L2000.R64.C200.nsg \
# "$L_str" 100 "$thread_num" /SSD/GIST1M/gt.query.top100.bin "gist" | tee -a /home/mqj/proj/ANNSLib/experiments/gist_8t_ours.csv

#openai
# L_list=(100 125 150 175 200 225 250 275 300 325 350 375 400 425 450 475 500 525 550 575 600)
# L_str=$(IFS=,; echo "${L_list[*]}")
# taskset -c 0-$((thread_num - 1)) ./build/tests/parallel_search /SSD/OpenAI-1M/base.fbin \
# /SSD/OpenAI-1M/query.fbin \
# /SSD/models/nsg/openai1m.L500.R64.C500.nsg \
# "$L_str" 100 "$thread_num" /SSD/OpenAI-1M/groundtruth.bin "gist" | tee -a /home/mqj/proj/ANNSLib/experiments/openai_8t_ours.csv

# taskset -c 0-$((thread_num - 1)) ./build/tests/parallel_search /SSD/Wiki10M/base.10M.fbin \
# /SSD/Wiki10M/queries.fbin \
# /SSD/models/nsg/wiki10m.L2000.R64.C2000.L2nsg \
# "$L_str" 100 "$thread_num" /SSD/Wiki10M/groundtruth.10M.neighbors.ibin "wiki"

# taskset -c 0-$((thread_num - 1)) ./build/tests/parallel_search /SSD/DEEP10M/base.fbin \
# /SSD/DEEP10M/query.fbin \
# /SSD/models/nsg/deep10m.L2000.R64.C2000.nsg \
# "$L_str" 100 "$thread_num" /SSD/DEEP10M/gt.query.top100.bin "deep"


# for deltastepping and edge-wise
# L_list=(100 150 200 250 300 350 400 450 500 550 600 1000 1100 1700 1800 1900 2800 3000)
# L_list=(1200 1300 1400 1500 1600 1800 1900 2100 2300 2500 2700 2900 3000)
