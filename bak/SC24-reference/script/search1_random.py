# 2024.03.15 Fri.
#  对整体搜索空间进行遍历 --- 不要一点减枝 如果剪枝 [(32768, 49152), (0, 16384)]
#  总体搜索空间 2^16 迭代次数取 2^16开方 --> 2^8 = 256
#   
#  在迭代为 128, 144, 160, 176, 192, 208, 224, 240, 256 记录下来当前迭代的最佳的性能
#  前128次的迭代过程也是 xgboost数据集的生成过程
# 
import os
import sys
import csv
import timeit
import random
def translate_value(value):
    binary_str = bin(value)[2:].zfill(16)  # 将整数值转换为16位二进制字符串
    result = []
    segment_count = 0
    segment_start = None
    prev_bit = None
    
    for i in range(1, 16):
        if i == 1:
            prev_bit = binary_str[i]
            segment_start = i
        elif binary_str[i] == prev_bit:
            continue
        else:
            if i - segment_start > 1:
                result.extend([segment_start, i - 1])
                segment_count += 1
            prev_bit = binary_str[i]
            segment_start = i

    if 16 - segment_start > 1:  # 处理最后一个 segment
        result.extend([segment_start, 15])
        segment_count += 1
        
    if segment_count == 1 and result[0] == 0 and result[-1] == 15:
        segment_count = 0
        result = []

    # return f"{binary_str} {binary_str[0]} {segment_count} {' '.join(map(str, result)) if segment_count > 0 else ''}"
    return f"{binary_str[0]} {segment_count} {' '.join(map(str, result)) if segment_count > 0 else ''}"

def unique_random_numbers(n, low, high):
    # 生成[low, high]的不重复随机数序列
    if high - low < n:
        raise ValueError("范围内无法生成足够的不重复随机数")
    return random.sample(range(low, high), n)

def main():
    batch_size       = 16
    layer_num        = 12
    seq_len          = int(sys.argv[1])
    head_num         = 12
    head_size        = 64
    mask_id          = int(sys.argv[2])
    if_fused         = 1
    iter_time        = 128
    round_time       = 5
    results = []
    min_times = {}
    global_best_config = 0
    global_min_time = min_times.get(1)
    

    t0_start = timeit.default_timer() 
    
    for ii in range(5):
        iteration_times = []
        min_times.clear()
        
        # train_data_filename = "./train_data/{}_seq{}_round{}_train_dataset.txt".format(mask_id, seq_len, ii + 1)
        unique_numbers = unique_random_numbers(iter_time, 0 ,65535) # 不对反码做剪枝
        for index, value in enumerate(unique_numbers, start=1):
            translation = translate_value(value)
            
            # 解析返回的字符串
            parts = translation.split()
            segment_num = int(parts[1])
            segments = []
            for i in range(2, len(parts), 2):
                start = int(parts[i]) - 1 
                end = int(parts[i+1]) - 1
                segments.extend([start, end])
        
        
            command = "python generate_genidx.py" + " {} {} {} {} {} {} {} {}".format(batch_size, layer_num, seq_len, head_num, head_size, mask_id, if_fused, value)
            output = os.popen(command).read()
            time_value = float(output.split("time costs:")[1].split("ms")[0].strip())
            
            #To print some INFO 
            if index % 16 == 1:
                print("round:",ii+1," iter:", index, " time_value:", time_value ,"\tvalue:", value, segment_num, segments)

            min_time = min_times.get(index)
            if min_time is None or time_value < min_time:
                min_times[index] = time_value
                min_time = min(min_times.values())
                if global_min_time is None or min_time < global_min_time:
                    global_min_time = min_time
                    global_best_config = value
            iteration_times.append(min_time)
            
            #在前128轮需要记录 训练的过程 作为xgboost的数据集
            # if index <= iter_time/2:
            #     binary_str = bin(value)[2:].zfill(16)
            #     output_str = binary_str+"   "+str(time_value)+"\n"
            #     with open(train_data_filename, "a") as train_file:
            #         train_file.write(output_str)
            
        results.append(iteration_times)
        
        

    #对搜索结果做记录
    csv_filename = f"./search_result/{mask_id}_rd_seq{seq_len}_Time.csv"
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["iteration"] + [f"round_{ii+1}" for ii in range(round_time)])  # 写入表头
        # for index in range(len(unique_numbers)):
        for index in range(iter_time):
            row = [index+1] + [results[ii][index] for ii in range(round_time)]
            writer.writerow(row)
            
            
    print(f"Result have been stored in {csv_filename} !")
    
    t0_end = timeit.default_timer()
    print("\nRandom search time costs:  \t{:.1f} s".format((t0_end - t0_start))) 
    print("Global Min Time:", global_min_time, "\tGlobal best Config:", global_best_config)
    
if __name__ == "__main__":
    main()
