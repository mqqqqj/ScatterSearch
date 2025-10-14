cd build && make -j && cd ..

intra_num=32

inter_num=1

export OMP_NUM_THREADS="$inter_num,$intra_num"

# L_list=(100 125 150 175 200 225 250 275 300 350 375 400 425 450 475 500 525 550 575 600 650 700 750 800 900 1000 1100 1200 )
# L_list=(1300 1400 1500 1600 1700 1800 1900 2000)
# L_list=(4000 4200 4400 4600 4800 5000)
L_list=(10 15 20 25 30 35 40 45 46 47 48 49 50 55 60 65 70 75 80 85 90 95)

L_str=$(IFS=,; echo "${L_list[*]}")

taskset -c 0-17,36-53 ./build/tests/intraxinter \
/SSD/Glove200/base.1m.fbin \
/SSD/Glove200/query.fbin \
/SSD/models/nsg/glove200.L2000.R64.C200.nsg \
"$L_str" 100 $intra_num $inter_num /SSD/Glove200/gt.query.top100.bin "glove" | tee -a /home/mqj/proj/ANNSLib/experiments/intraxinter/glove_${intra_num}x${inter_num}_ours.csv

