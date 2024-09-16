# 9.15
# 对于torch.compile的初步使用
# 实际上没有关注到triton，直观对比了torch.compile和eager模式间的区别
import torch
import numpy as np
import torch._dynamo
from torchvision.models import densenet121

# @torch.compile
# def foo(x, y):
#     a = torch.sin(x)
#     b = torch.cos(y)
#     return a + b
# opt_foo1 = torch.compile(foo)
# print(opt_foo1(torch.randn(10, 10), torch.randn(10, 10)))

# Returns the result of running `fn()` and the time it took for `fn()` to run,
# in seconds. We use CUDA events and synchronization for the most accurate
# measurements.
def timed(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000

# Generates random input and targets data for the model, where `b` is
# batch size.
def generate_data(b):
    return (
        torch.randn(b, 3, 128, 128).to(torch.float32).cuda(),
        torch.randint(1000, (b,)).cuda(),
    )

N_ITERS = 10

def init_model():
    return densenet121().to(torch.float32).cuda()



model = init_model()
# Reset since we are using a different mode.
torch._dynamo.reset()

# 具体取决于所选的 mode 参数。“reduce-overhead” 模式使用 CUDA 图来进一步减少 Python 的开销
model_opt = torch.compile(model, mode="reduce-overhead")


# 直观演示一个 编译的时间差异
inp = generate_data(16)[0]
with torch.no_grad():
    print("eager:", timed(lambda: model(inp))[1])
    print("compile:", timed(lambda: model_opt(inp))[1])

    
    
# 推理，观察实际执行速度差异 --- 选取了中值
# eager_times = []
# for i in range(N_ITERS):
#     inp = generate_data(16)[0]
#     with torch.no_grad():
#         _, eager_time = timed(lambda: model(inp))
#     eager_times.append(eager_time)
#     print(f"eager eval time {i}: {eager_time}")
# print("-" * 30)

# compile_times = []
# for i in range(N_ITERS):
#     inp = generate_data(16)[0]
#     with torch.no_grad():
#         _, compile_time = timed(lambda: model_opt(inp))
#     compile_times.append(compile_time)
#     print(f"compile eval time {i}: {compile_time}")
# print("-" * 30)

# eager_med = np.median(eager_times)
# compile_med = np.median(compile_times)
# speedup = eager_med / compile_med
# assert(speedup > 1)
# print(f"(eval) eager median: {eager_med}, compile median: {compile_med}, speedup: {speedup}x")
# print("-" * 30)


model = init_model()
opt = torch.optim.Adam(model.parameters())

def train(mod, data):
    opt.zero_grad(True)
    pred = mod(data[0])
    loss = torch.nn.CrossEntropyLoss()(pred, data[1])
    loss.backward()
    opt.step()

eager_times = []
for i in range(N_ITERS):
    inp = generate_data(16)
    _, eager_time = timed(lambda: train(model, inp))
    eager_times.append(eager_time)
    print(f"eager train time {i}: {eager_time}")
print("~" * 10)

model = init_model()
opt = torch.optim.Adam(model.parameters())
train_opt = torch.compile(train, mode="reduce-overhead")

compile_times = []
for i in range(N_ITERS):
    inp = generate_data(16)
    _, compile_time = timed(lambda: train_opt(model, inp))
    compile_times.append(compile_time)
    print(f"compile train time {i}: {compile_time}")
print("~" * 10)
# 观察实际的运行 除了第一次编译很花时间，第二次也会比正常运行要长，但比第一次运行快得多。
# 因为 “reduce-overhead” 模式为 CUDA 图运行了一些预热迭代

eager_med = np.median(eager_times)
compile_med = np.median(compile_times)
speedup = eager_med / compile_med
assert(speedup > 1)
print(f"(train) eager median: {eager_med}, compile median: {compile_med}, speedup: {speedup}x")
print("~" * 10)
