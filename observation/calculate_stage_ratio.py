def calculate_average(file_path):
    """
    从文件中读取每行的整数，并计算它们的平均值。
    """
    numbers = []

    try:
        with open(file_path, 'r') as file:
            for i, line in enumerate(file, 1):
                try:
                    # 移除行首尾空白并转换为整数
                    number = int(line.strip())
                    numbers.append(number)
                except ValueError:
                    print(f"警告：第 {i} 行 '{line.strip()}' 不是有效的整数，已跳过。")
    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 未找到。请检查文件路径。")
        return None

    # 计算平均值
    if not numbers:
        print("文件中没有有效的整数数据，无法计算平均值。")
        return None
    
    average = sum(numbers) / len(numbers)
    print("---")
    print(f"总共有 {len(numbers)} 个整数。")
    print(f"所有整数的平均值为: {average:.2f}")
    
    return average

def calculate_average_ratio(file_path):
    """
    从指定文件中读取数据，计算每行两个数的比例，并返回所有比例的平均值。
    文件格式应为每行两个用逗号分隔的整数。
    """
    ratios = []
    avg_numerator = 0
    avg_denominator = 0
    try:
        with open(file_path, 'r') as file:
            for i, line in enumerate(file, 1):
                # 移除行首尾的空白字符并按逗号分割
                parts = line.strip().split(',')
                if len(parts) == 2:
                    try:
                        numerator = int(parts[0])
                        avg_numerator += numerator
                        denominator = int(parts[1])
                        avg_denominator += denominator
                        # 检查分母是否为零
                        if denominator != 0:
                            ratio = numerator / denominator
                            ratios.append(ratio)
                            print(f"第 {i} 行：{numerator}/{denominator} = {ratio:.5f}")
                        else:
                            print(f"警告：第 {i} 行的分母为零，跳过。")
                    except ValueError:
                        print(f"警告：第 {i} 行数据格式不正确，跳过。")
                else:
                    print(f"警告：第 {i} 行数据格式不正确，跳过。")
    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 未找到。请确保文件存在。")
        return None

    # 计算平均值
    if ratios:
        average_ratio = sum(ratios) / len(ratios)
        print("---")
        print(f"所有行的平均比例是：{average_ratio:.5f}")
        avg_numerator /= len(ratios)
        avg_denominator /= len(ratios)

        print(avg_numerator)
        print(avg_denominator)

        return average_ratio
    else:
        print("没有有效数据来计算平均值。")
        return None

# 调用函数并传入你的文件名
calculate_average_ratio('/home/mqj/proj/ANNSLib/observation/deep_16t_two_stage_ndc_my.csv')
calculate_average('/home/mqj/proj/ANNSLib/observation/deep_16t_first_stage_ndc.csv')