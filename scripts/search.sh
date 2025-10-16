cd build && make -j && cd ..

L_list=(100 200 300 400 500 700 900 1100 1400 1600 1800 2000 2200 2400 2600 2800 3000 3200 3400 3600 3800 4000 4200)

# 将L_list转换为逗号分隔的字符串
L_str=$(IFS=,; echo "${L_list[*]}")

# taskset -c 0 ./build/tests/search /SSD/Text-to-Image/base.10M.fbin \
# /SSD/Text-to-Image/query.10k.fbin \
# /SSD/models/nsg/t2i10m.L2000.R64.C20
# "$L_str" 100 /SSD/Text-to-Image/gt.10K_10M.bin t2i | tee -a /home/mqj/proj/ANNSLib/experiments/t2i_seq_nsg.csv

# taskset -c 0 ./build/tests/search /SSD/LAION/LAION_base_imgemb_10M.fbin \
# /SSD/LAION/LAION_test_query_textemb_50k.fbin \
# /SSD/models/nsg/laion.L2000.R64.C200.nsg \
# "$L_str" 100 /SSD/LAION/gt.test50K.bin laion | tee -a /home/mqj/proj/ANNSLib/experiments/laion_seq_nsg.csv

# taskset -c 0 ./build/tests/search /SSD/MainSearch/base.fbin \
# /SSD/MainSearch/query_test_unique.fbin \
# /SSD/models/nsg/mainsearch.L2000.R64.C2000.nsg \
# "$L_str" 100 /SSD/MainSearch/gt.test_unique.bin mainsearch | tee -a /home/mqj/proj/ANNSLib/experiments/mainsearch_seq_nsg.csv

# taskset -c 0 ./build/tests/search /SSD/WebVid/webvid.base.2.5M.fbin \
# /SSD/WebVid/webvid.query.10k.fbin \
# /SSD/models/nsg/webvid.L2000.R64.C2000.nsg \
# "$L_str" 100 /SSD/WebVid/gt.query.top100.bin webvid | tee -a /home/mqj/proj/ANNSLib/experiments/webvid_seq_nsg.csv

taskset -c 0-17,36-49 ./build/tests/search /SSD/Glove200/base.1m.fbin \
/SSD/Glove200/query.fbin \
/SSD/models/nsg/glove200.L2000.R64.C200.nsg \
"$L_str" 100 /SSD/Glove200/gt.query.top100.bin glove #| tee -a /home/mqj/proj/ANNSLib/experiments/glove_seq_nsg.csv

# taskset -c 0 ./build/tests/search /SSD/GIST1M/base.fbin \
# /SSD/GIST1M/query.fbin \
# /SSD/models/nsg/gist1m.L2000.R64.C200.nsg \
# "$L_str" 100 /SSD/GIST1M/gt.query.top100.bin gist | tee -a /home/mqj/proj/ANNSLib/experiments/gist_seq_nsg.csv