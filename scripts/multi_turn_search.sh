cd build && make -j && cd ..

L_list=(100 150 200 250 300 350 400 450 500 550 600)
# L_list=(100 120 150 170 200 220 250 270 300)

# L_list=(400)

thread_num=8

L_str=$(IFS=,; echo "${L_list[*]}")

taskset -c 0-$((thread_num - 1)) ./build/tests/multi_turn_search /SSD/LAION/LAION_base_imgemb_10M.fbin \
/SSD/LAION/LAION_test_query_textemb_50k.fbin \
/SSD/models/nsg/laion.L2000.R64.C2000.nsg \
"$L_str" 100 "$thread_num" /SSD/LAION/gt.test50K.bin "laion"