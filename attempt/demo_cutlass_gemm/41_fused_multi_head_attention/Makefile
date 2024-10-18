# 定义编译器
NVCC := nvcc

# 定义编译和链接标志
CFLAGS := -std=c++11 -O3 -I/usr/local/cuda/include -I../include -I../ -gencode=arch=compute_80,code=sm_80
# LDFLAGS := -L/usr/local/cuda/lib64 -lcudart

# 定义源文件和目标文件
SRC := fused_multihead_attention_fixed_seqlen.cu
TARGET := 41_fused_multihead_attention_fixed_seqlen

# 默认目标
all: $(TARGET)

# 编译和链接生成可执行文件
$(TARGET): $(SRC)
	@echo "Current working directory: `pwd`"
	@rm -f $(TARGET)
# $(NVCC) $(CFLAGS) $(LDFLAGS) -o $@ $<
	$(NVCC) $(CFLAGS) -o $@ $<

# 清理生成的文件
clean:
	rm -f $(TARGET)
