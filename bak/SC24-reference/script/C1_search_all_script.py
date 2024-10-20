# 2024.03.25 Wed.
#  更换了我们的搜索方式 看看效果
# 
import os
# print("Start Search script ... ... ")

for mask in [2]:
# for mask in[0]:
    if(mask == 1): 
        mask_name = 'Strided_mask'
    else:
        mask_name = 'Fixed_mask'
    print("Stage{} {} train dataset generate start".format(mask, mask_name))
    
    # for seqlen in [64, 128, 256, 384, 512, 768, 1024]:
    for seqlen in [256, 384, 512, 768, 1024]:
        
        if seqlen == 256:
            for filename in ["search2_simulated_an", "search3_ours"]:
                print("{}   {}.py {} script start !".format(mask_name, filename, seqlen))
                os.system("python {}.py {} {}".format(filename, seqlen, mask))
                
        else:
            print("seqlen = ", seqlen, "-"*30 )
            for filename in ["search1_random", "search2_simulated_an", "search3_ours"]:
                print("{}   {}.py {} script start !".format(mask_name, filename, seqlen))
                os.system("python {}.py {} {}".format(filename, seqlen, mask))

print("Congratulations! Search finished!")
