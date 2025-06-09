cd build && make -j && cd ..

taskset -c 0-7 ./build/tests/single_query /SSD/MainSearch/base_100k.fbin \
/SSD/MainSearch/test_5k.fbin \
/SSD/models/nsg/mainsearch100k.L2000.R64.C2000.nsg \
100 100 8 /SSD/MainSearch/gt.test5k_base100k.bin