cd build && make -j && cd ..

intra_num=8

inter_num=4

L_list=(100 150 200 250 300 350 375 400 425 450 475 500 525 550 575 600 650 700 750 800 900 1000 1100 1200)

L_str=$(IFS=,; echo "${L_list[*]}")

taskset -c 0-17,36-49 ./build/tests/intraxinter \
/SSD/Glove200/base.1m.fbin \
/SSD/Glove200/query.fbin \
/SSD/models/nsg/glove200.L2000.R64.C200.nsg \
"$L_str" 100 $intra_num $inter_num /SSD/Glove200/gt.query.top100.bin "glove" | tee -a /home/mqj/proj/ANNSLib/experiments/intraxinter/glove_${intra_num}x${inter_num}_ours.csv

