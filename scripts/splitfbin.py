import numpy as np
import struct
import os

def split_fbin_file(original_fbin_path, num_queries=1000):
    """
    将一个 .fbin 文件拆分为基础集 (base.fbin) 和查询集 (query.fbin)。

    .fbin 格式定义:
    - 第一个 4 字节整数 (int32): 向量数量 (n)
    - 第二个 4 字节整数 (int32): 向量维度 (d)
    - 之后是 n * d 个 4 字节浮点数 (float32)，代表所有向量的数据。

    参数:
    - original_fbin_path (str): 原始 .fbin 文件的路径。
    - num_queries (int): 要拆分出来作为查询集的向量数量（从文件末尾开始计算）。
    """
    try:
        # --- 1. 读取原始文件的头部信息 ---
        with open(original_fbin_path, 'rb') as f:
            # 读取向量总数 (n) 和维度 (d)
            n_total, d = struct.unpack('ii', f.read(8))
            print(f"原始文件 '{original_fbin_path}':")
            print(f"  - 向量总数 (n): {n_total}")
            print(f"  - 向量维度 (d): {d}")

            # 验证向量总数是否足够拆分
            if n_total < num_queries:
                print(f"❌ 错误：文件中的向量总数 ({n_total}) 少于要拆分的数量 ({num_queries})。")
                return

            # --- 2. 计算 base 和 query 的向量数量 ---
            n_base = n_total - num_queries
            n_query = num_queries
            
            print("\n计划拆分:")
            print(f"  - 'base.fbin' 将包含 {n_base} 个向量。")
            print(f"  - 'query.fbin' 将包含 {n_query} 个向量。")

            # --- 3. 读取所有向量数据 ---
            # 计算数据部分的总字节数
            data_bytes = n_total * d * 4  # 4 bytes per float32
            # 从文件当前位置（头部之后）读取所有数据
            all_vectors_data = f.read(data_bytes)
            
            # 将二进制数据转换为 numpy 数组以便于切片
            all_vectors = np.frombuffer(all_vectors_data, dtype=np.float32).reshape(n_total, d)

    except FileNotFoundError:
        print(f"❌ 错误: 文件 '{original_fbin_path}' 未找到。")
        return
    except Exception as e:
        print(f"❌ 读取文件时发生错误: {e}")
        return

    # --- 4. 拆分数据并写入新文件 ---

    # (a) 创建并写入 base.fbin
    base_vectors = all_vectors[:n_base]
    output_base_path = "base.fbin"
    try:
        with open(output_base_path, 'wb') as f_base:
            # 写入新的头部信息 (n_base, d)
            f_base.write(struct.pack('ii', n_base, d))
            # 写入 base 向量的数据
            f_base.write(base_vectors.astype(np.float32).tobytes())
        print(f"✅ 成功创建 '{output_base_path}' ({os.path.getsize(output_base_path) / (1024*1024):.2f} MB)")
    except Exception as e:
        print(f"❌ 写入 'base.fbin' 时发生错误: {e}")

    # (b) 创建并写入 query.fbin
    query_vectors = all_vectors[n_base:]
    output_query_path = "query.fbin"
    try:
        with open(output_query_path, 'wb') as f_query:
            # 写入新的头部信息 (n_query, d)
            f_query.write(struct.pack('ii', n_query, d))
            # 写入 query 向量的数据
            f_query.write(query_vectors.astype(np.float32).tobytes())
        print(f"✅ 成功创建 '{output_query_path}' ({os.path.getsize(output_query_path) / (1024*1024):.2f} MB)")
    except Exception as e:
        print(f"❌ 写入 'query.fbin' 时发生错误: {e}")


# === 使用示例 ===
if __name__ == "__main__":
    input_file = "/SSD/Crawl/crawl.fbin"  # <--- 修改这里

    # 调用拆分函数
    split_fbin_file(input_file, num_queries=1000)