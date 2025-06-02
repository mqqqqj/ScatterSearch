cd build && make -j && cd ..

taskset -c 0-7 ./build/tests/pipeline_search /SSD/Text-to-Image/base.10M.fbin \
/SSD/Text-to-Image/query.10k.fbin \
/SSD/models/nsg/t2i10m.L2000.R64.C2000.nsg \
200 100 4 /SSD/Text-to-Image/gt.10K_10M.bin | tee -a log.txt