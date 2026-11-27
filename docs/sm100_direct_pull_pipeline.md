# SM100 小 batch：撤回专职 CTA 投递，改用计算 CTA 内 K 流水线

这次保留的改动是非默认 `direct_pull=True` 路径的 K 分组预取。独立生产者 CTA 经 DSM 向计算 CTA 投递输入的实现已经撤回，没有作为开关留在算子中。默认 `direct_pull=False` 路径不变，仍是当前 EP4 测试中更快的选择。

## 测量条件与结果

GLM5.2 形状的随机输入/权重：H=6144、I=2048、topk=8、一个融合共享专家。路由专家权重 FP4/K32，共享专家权重 FP8/K128，输入 FP8/K32，输出 BF16。不是模型 checkpoint 或整模型推理测试。

EP4 使用全局 E=32，每卡八个路由专家，均衡时每专家 M=T/rank。它匹配目标 EP32/E=256 的每卡专家数及平均专家 token 数，不能复现 EP32 的通信拓扑、扇入或竞争。

最终同一进程内轮换测试各版本：CUDA Graph 每次 50 个调用，每版本预热 100 次 Graph，计时 11 次；每次取四个 rank 的最大延迟，再取中位数。包含算子的三个角色，排除权重转换、输入准备、JIT 和 capture。GPU 没有锁频，也没有独占整机保证。原始各次样本及源码 SHA256 见 [最终结果](benchmarks/gc_moe_ll_pipeline_final.json)。

| T/rank | 原 GC | 默认 LL | 旧直接拉取 | 新直接拉取 | 相对旧直接拉取延迟下降 |
|---:|---:|---:|---:|---:|---:|
| 4 | 48.35 µs | 41.46 µs | 57.97 µs | 53.82 µs | 7.17% |
| 8 | 47.86 µs | 41.82 µs | 62.30 µs | 57.86 µs | 7.13% |
| 16 | 51.69 µs | 42.91 µs | 70.43 µs | 63.19 µs | 10.28% |
| 32 | 61.21 µs | 55.78 µs | 85.72 µs | 75.45 µs | 11.99% |

新路径只是改善直接拉取的实验分支，尚未超过默认 LL 或原 GC。不能将以上改善百分比解释为相对默认 LL 的加速。

## 留下的具体改动

旧直接路径在某个专家输入块第一次进入 cluster 时，先把整块 X 和 SF 拉完，再通知 MMA；相邻两个 W1 N 输出任务复用同一个输入缓存。

新路径保持两个计算 CTA 和相同的任务复用方式：

1. 等待路由元数据就绪。换输入块前，等两 CTA 各自的最后一次 MMA 完成，保护旧缓存的读者。
2. 每个 lane 只在换块时解析自己负责的源 rank/token，缓存 X/SF 的源地址；K 循环不再反复读取元数据。
3. 一次清零缓存及 SF 填充区。跳过整组无效行，部分有效组用 cp.async 的零填充处理尾部。
4. 四个 K128 块组成一个 cp.async group。目标形状预发六组，即最多 24 个 K128 块在预取窗口内；小 H 自动收缩为 `min(6, H/512)` 组。
5. 只等待当前组完成，保留后续组在途；完成 warp 同步和 async-proxy fence 后，继续发更远的一组，再按原有 data_empty/data_full 协议逐 K128 通知 MMA。
6. 尾部排空剩余异步拷贝。整块缓存保留给连续 W1 N 任务，缓存命中时不再跨卡拉取。

X 仍直接落入最终 K128/SW128 布局，没有共享内存二次重排，也没有额外 DSM 复制或专职 CTA。

SF 有一个关键差别：旧 BM16 缓存允许相邻 K 的 128-word SF 读取窗口重叠，因为读取前整个缓存已准备好。流水线中，MMA 读取当前窗口时后续 SF 可能仍在写入，因此新布局使用独立的 128-word SF 槽，避免读取窗口与后续写入重叠。增加的共享内存由 host sizing 纳入计算，并相应调整 TMA stage 数。

实现位置：`gc_moe/common/direct_pull.py` 的 `prepare_peer_sources` / `prefetch_peer_k_group`；SM100 role2 激活搬运 warp 的缓存 miss 分支及 K 循环；`gc_moe/layout/direct_pull.py` 的 SF 槽布局。

## 为什么撤回独立生产者 CTA

普通三 CTA cluster 的启动及 DSM 读写小测试通过，但接入现有双 CTA MMA 的 kernel 后，Graph 实例化报 `CUDA_ERROR_INVALID_CLUSTER_SIZE`。实际实验改为四 CTA cluster：计算对占 CTA0/1，测试过一个生产者 CTA，也测试过让 CTA2/3 分别服务计算对的两半。

实现过的优化包括预申请多个目标 stage、源/目的缓冲分别确认可复用、scale 填充只清零一次、2 个生产者，以及合并 4/8 个 K128 块进行 DSM 投递。批量搬运后仍明显落后：八块一批、双生产者的同轮测试，T=4 为 96.83 µs（旧直接拉取 59.65），T=32 为 171.12 µs（旧路径 86.15）；单生产者为 100.61/173.79 µs。

资源布局也不利：旧路径约 75 个双 CTA 计算 cluster；四 CTA 配置只有约 37 个计算对。均衡小 batch 的 W1 任务需要更多轮调度，并额外付出 DSM 复制及同步成本。这里的归因来自实现资源配置与消融实验，并非 Nsight 吞吐计数器给出的精确分摊。

遵照“明显更慢就不保留”的要求，独立生产者代码、cluster 启动改动和相关开关均未留在项目中。只保留实验数据与结论，避免以后重复该方案。

## 预取深度选择

尝试过 2/4/6/8 组。六组对目标形状最好；八组没有进一步改善。探索轮 T=32 的新路径约为：两组 82.14 µs、四组 76.79 µs、六组 75.71 µs、八组 78.18 µs。探索轮之间的绝对值不应混作同轮公平对照；最终表来自上面的统一复测。

## 运行方式

正常调用仍是 `fp8_fp4_gc_moe_ll(..., direct_pull=True)`；H >= 1024 时启用新流水线，较小 H 使用原来的整块缓存实现。`GC_MOE_LL_DIRECT_PIPELINE=0` 可显式使用旧直接路径作对照。此开关和资源布局都进入编译/图缓存键。

```bash
unset PYTHONPATH
export PATH=/usr/local/cuda/bin:$PATH
export NCCL_DEBUG=WARN GC_MOE_NVLINK_BARRIER_TIMEOUT_SECS=20
../gc-ll-venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 \
  tests/bench_gc_moe_ll.py --tokens 4,8,16,32 \
  --variants gc,ll,ll_direct,ll_pipeline \
  --patterns balanced,random,skew,empty,mixed \
  --graph-calls 50 --warmup-graphs 100 --repeats 11 \
  --output results/pipeline-final.json
```

基准中的 `ll_direct` 显式关闭流水线，`ll_pipeline` 显式打开，`ll` 使用默认 staging 路径。

## 后续优化方向

更值得优先研究的是减少实际重复工作，而非继续增加生产者 CTA：例如降低默认 LL 的中间张量往返和最终 combine 延迟，或让多个输出任务共享输入而不减少可用计算对。W1/SwiGLU/W2 沿 I 分块融合需要同时处理 W2 的部分和归约成本，应独立做资源预算和端到端对照。

不能仅靠让同一层权重长期驻留 L2 来宣布模型加速：反复 replay 单个算子的缓存条件与完整模型逐层执行不同。本次保留的改动是调用内部的预取重叠与元数据复用。

## 正确性与异步访问验证

共 175 个非 sanitizer 形状/路由/版本组合通过独立 Torch 参考对照；输入、scale、路由及路由权重会更新，再 replay 已捕获的 Graph。参考包含中间 BF16 舍入、SwiGLU、路由加权、FP8 中间激活量化和 W2/合并。判据为 relative L2 < 0.025、cosine > 0.999 且结果有限；最终目标形状四版本的最大 relative L2 为 0.001575 以下。

- 目标 H6144：T=4/8/16/32，四版本 × 均衡、随机、热点、空路由、部分无效路由，共 80 项。
- 无共享专家：T=0/1/7/17/31，两直接路径、五种路由，共 50 项。
- H1536：T=1/17/32，三组预取与尾部排空，共 15 项。
- H1024：T=7/17，两组预取，共 10 项。
- H512、无共享专家：T=17，旧实现回退分支，两版本共 10 项。共享专家版本的合成 SF 只有一列，测试生成器产生的 stride 被原有 API 拒绝；没有为绕过验证修改接口。
- H1536、每卡两个专家、无共享专家：T=17/32，覆盖 BM32，共 10 项。

原始记录见 [边界 token](benchmarks/gc_moe_ll_pipeline_edges.json)、[H1536](benchmarks/gc_moe_ll_pipeline_h1536.json)、[H1024](benchmarks/gc_moe_ll_pipeline_h1024.json)、[旧路径回退](benchmarks/gc_moe_ll_pipeline_fallback.json)、[BM32](benchmarks/gc_moe_ll_pipeline_bm32.json)。

Compute Sanitizer 2026.1.1：

- memcheck：H6144，T=4/32，五种路由，eager 和 Graph replay，`ERROR SUMMARY: 0 errors`，见 [日志](benchmarks/gc_moe_ll_pipeline_memcheck.log)。
- racecheck：H1536，T=17，均衡/热点/部分无效路由，`0 hazards (0 errors, 0 warnings)`，见 [日志](benchmarks/gc_moe_ll_pipeline_racecheck.log)。

两者都使用 `--target-processes all --kernel-name kns=kernel_cutlass_kernel_ --report-api-errors no --error-exitcode 99`，只检查 GC CuTe kernels，关闭 CUDA Python 对不同驱动版本 API 探测的报告。它们是这些用例的设备检查结果，不是对任意输入或跨 rank 全局同步的形式化证明。Sanitizer 下输出的 BENCH 数值不用于性能结论。

[探索记录](benchmarks/gc_moe_ll_pipeline_experiments.json) 包含已撤回 cluster 方案和预取深度试验的原始结果；部分探索轮有明显异常延迟，最终性能结论仅使用统一预热、轮换计时的最终结果。

测试时保留了工作区原有的 `gc_moe/ptx/ld_st.py` 修改和未跟踪的 `gc_moe/layout/gc_moe_ll.py` 草稿；它们未纳入本次提交。结果文件的源码哈希反映实测工作区，不等同于把这些用户修改一并提交。
