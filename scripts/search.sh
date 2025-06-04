cd build && make -j && cd ..

L_list=(100 200 300)

for L in "${L_list[@]}"
do
    # taskset -c 0 ./build/tests/search /SSD/WebVid/webvid.base.2.5M.fbin \
    # /SSD/WebVid/webvid.query.train.10k.fbin \
    # /SSD/models/nsg/webvid.L2000.R64.C2000.nsg \
    # 600 100 /SSD/WebVid/gt.train10k.top100.bin

    taskset -c 0 ./build/tests/search /SSD/Text-to-Image/base.10M.fbin \
        /SSD/Text-to-Image/query.10k.fbin \
        /SSD/models/nsg/t2i10m.L2000.R64.C200.nsg \
        $L 100 /SSD/Text-to-Image/gt.10K_10M.bin | tee -a log.txt
done