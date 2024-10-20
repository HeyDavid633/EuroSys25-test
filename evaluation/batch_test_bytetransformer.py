import os
import csv

results = {}

for batch_size in [1, 8, 16]:
    batch_results = []
    for seqlen in [64, 128, 256, 384, 512, 768, 1024]:
        # 执行命令并捕获输出
        output = os.popen("python bert_transformer_test.py {} {} 12 64 --n_layers 1 --avg_seqlen 0 --dtype fp16 --iters 10".format(batch_size, seqlen)).read()

        # 提取执行时间
        try:
            time_value = float(output.split("time costs: ")[1].split(" ms")[0].strip())
            batch_results.append(time_value)
        except ValueError:
            print(f"Failed to extract time cost for batch_size={batch_size}, seqlen={seqlen}")
            
    results[batch_size] = batch_results

# 写入CSV文件
with open('bytetransformer_fp16.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['batch_size'] + [str(s) for s in [64, 128, 256, 384, 512, 768, 1024]])

    for batch_size, times in results.items():
        writer.writerow([str(batch_size)] + [str(t) for t in times])

csvfile.close()