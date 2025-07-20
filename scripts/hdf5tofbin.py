import h5py
import numpy as np
import struct

# base集合和query集合要遵守fbin文件格式，fbin文件格式为，num（4字节），dim（4字节），num*dim个float数据
# groundtruth集合的格式为：nq（4字节），k（4字节），nq*k个unsigned数据

# hdf5格式如下：
# train: numpy array of size (n_corpus, dim) containing the embeddings used to build the vector index
# test: numpy array of size (n_test, dim) containing the test query embeddings
# neighbors: numpy array of size (n_test, 100) containing the IDs of the true 100 k-nn of each test query

hdf5_path = "/SSD/landmark/landmark-nomic-768-normalized.hdf5"
base_fbin_path = "/SSD/landmark/base.fbin"
query_fbin_path = "/SSD/landmark/query.fbin"
gt_bin_path = "/SSD/landmark/groundtruth.bin"

with h5py.File(hdf5_path, "r") as f:
    # # 导出base.fbin
    # train = f["train"][:]
    # print(train.dtype)
    # num, dim = train.shape
    # print(f"train shape: {train.shape}")
    # with open(base_fbin_path, "wb") as fout:
    #     fout.write(struct.pack("i", num))
    #     fout.write(struct.pack("i", dim))
    #     if train.dtype == np.float32:
    #         for i in range(num):
    #             fout.write(struct.pack(f"{dim}f", *train[i]))
    #     elif train.dtype == np.float64:
    #         for i in range(num):
    #             fout.write(struct.pack(f"{dim}d", *train[i]))
    #     else:
    #         raise ValueError(f"Unsupported dtype for train: {train.dtype}")

    # # 导出query.fbin
    # test = f["test"][:]
    # print(test.dtype)
    # nq, dim2 = test.shape
    # print(f"test shape: {test.shape}")
    # assert dim2 == dim, "train和test的维度不一致"
    # with open(query_fbin_path, "wb") as fout:
    #     fout.write(struct.pack("i", nq))
    #     fout.write(struct.pack("i", dim2))
    #     if test.dtype == np.float32:
    #         for i in range(nq):
    #             fout.write(struct.pack(f"{dim2}f", *test[i]))
    #     elif test.dtype == np.float64:
    #         for i in range(nq):
    #             fout.write(struct.pack(f"{dim2}d", *test[i]))
    #     else:
    #         raise ValueError(f"Unsupported dtype for test: {test.dtype}")

    # 导出groundtruth.bin
    neighbors = f["neighbors"][:]
    print(neighbors.dtype)
    nq2, k = neighbors.shape
    print(f"neighbors shape: {neighbors.shape}")
    with open(gt_bin_path, "wb") as fout:
        fout.write(struct.pack("i", nq2))
        fout.write(struct.pack("i", k))
        # neighbors一般为int32或uint32，如果为int64则先转为int32
        if neighbors.dtype == np.int32 or neighbors.dtype == np.uint32:
            for i in range(nq2):
                fout.write(struct.pack(f"{k}I", *neighbors[i]))
        elif neighbors.dtype == np.int64:
            neighbors = neighbors.astype(np.int32)
            for i in range(nq2):
                fout.write(struct.pack(f"{k}I", *neighbors[i]))
        else:
            raise ValueError(f"Unsupported dtype for neighbors: {neighbors.dtype}")
