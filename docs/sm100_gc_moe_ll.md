# SM100 小 batch MoE：实现与 EP4 实测

新增接口 `gc_moe.fp8_fp4_gc_moe_ll`，支持 T/rank ≤ 32、EP ≤ 32。

直接拉取路径现已去掉共享内存二次重排，最新实现、对照结果与限制见 [直接布局优化报告](sm100_direct_pull_optimization.md)。本页性能表保留初版 `878682f` 的记录。
这是对现有 SM100 MMA/量化/输出管线的低延迟特化：新写了路由登记、直接拉取缓存及其调度路径，复用已有 FP8×FP4 数学管线。不是将原接口改名，也不是完全重写所有 MMA/epilogue。

## 默认路径与实验路径

默认 `direct_pull=False`，因为在本次 GLM5.2 参数测试中更快：

1. 路由 CTA 按专家分工，最多八个 CTA；全局专家 `e` 的登记由 `e % route_grid` 的 CTA 独占。每个 CTA 扫描小路由表一次，过滤自己的专家，直接用共享内存计数的原子返回值分配 source 槽位。没有“先计数、再预留全局区间、再次扫描”的流程。多个 CTA 会重复读取这张小表，不能说整个 grid 只读了一遍。
2. CTA 间屏障保证旧代计数清理与 source 写入完成，再由各专家的唯一发布者发布 count。每个源 rank 对每个专家只有一份完成贡献，接收端等待 EP 份；零路由专家也参与。保留双代信号和系统级 release/acquire，避免 Graph 重放读到旧路由。
3. 并行搬运 CTA 将源 token/SF 拉到本卡激活环形区，跨卡搬运一次，供同一专家的多个 W1 输出块复用。
4. 计算使用 primary context 的并发 persistent kernel，无固定 Green Context 分区；计算 grid 给路由 CTA 留出驻留空间。BM 根据平均每专家 token 数选择 16/32。共享专家任务先发出，随后 W1/SwiGLU/FP8 量化及 W2 按 K128 就绪依赖流水执行。
5. W2 将贡献写到源 rank 的输出槽位，独立 finalize kernel 等待发布、累加、写回 BF16，并清理工作区。小 batch 的 finalize grid 按实际输出工作量缩小。

`direct_pull=True` 保留讨论中的直接拉取方案，供后续研究：

- 单个路由 CTA 建表，之后一条 lane 整理一条路由，输出 `(src_rank, src_token, topk_slot)` 和权重；不写 routed X/SF 全局暂存区。
- 计算 CTA 用 peer UVA + `cp.async` 将 X 直接写入 K128/SW128 缓存，将 SF 直接写入 MMA 消费的排列；MMA 描述符直接引用这些地址，不再将缓存搬进第二份 SMEM stage。权重 TMA 与 MMA 继续多级流水。
- 两个连续 W1 N tile 分配给同一 cluster，复用输入缓存。完整缓存需要共享内存，因此减少权重流水级数，并在编译前检查容量。
- cache 由 activation loader warp 填充、MMA 直接读取。切换到另一 pool block 前，各 CTA 等待紧邻上一任务的 MMA 完成，避免覆盖仍在使用的数据；复用同一 pool block 时保持缓存不变。
- 目前只接受 H 为 512 的倍数且 H ≤ 8192。实验路径未被默认自动选择。

第一版逐 K128 远端读取并等待的实现约 260–330 µs，等待链太长，已替换。整块缓存及 N tile 复用后明显改善，但最终仍慢于默认路径。增加 N 分组到 4 会减少并行度，实测也更慢。后续直接布局优化结果见本文开头链接。保留独立实验开关而非宣称“直接放共享内存一定快”。

没有实现完整 W1→SwiGLU→W2 的 I 分片融合或 split-K。以 M=32、H=6144、I=2048、I 分片=128 为例，16 份 FP32 W2 partial 在每专家会占约 12 MiB，8 专家约 96 MiB；现有 FP8 中间激活总共约 0.5 MiB。若把 partial 写到显存再归约，额外读写可能超过省下的中间激活流量。避免 partial 落地的融合方案仍需独立实现及验证。

## 测试口径

2026-11-26，单机四卡，CUDA capability 10.0，设备报告名 `NVIDIA L20A`（不能据此按 Ada 判断）。PyTorch 2.13.0+cu130，CuTe DSL 4.8.1，CUDA Python 13.2.0，系统 CUDA 13.2 ptxas。

使用 GLM5.2 **参数形状的合成权重/输入**，不是加载模型 checkpoint，也不是端到端模型测试：

- H=6144，I=2048，top-k=8；routed 权重 FP4 E2M1/K32 UE8M0，输入 FP8 E4M3/K32。
- 1 个共享专家，权重 FP8 E4M3/K128，按 128×128 block 量化。
- EP4 总 routed 专家数 **32**，每卡 8 专家；均衡路由使每专家 M=T/rank。
- EP32 对应总专家数 256，仍每卡 8 专家、M=T。EP4 对齐本卡计算/权重规模，但不复现 EP32 的 32 路 fan-in、跨机拓扑、远端比例及拥塞；不可把此处结果直接当 EP32 性能。
- 输入准备、权重重排和首次 JIT 不计时。一次外层 CUDA Graph 含 50 次调用，9 次重复，变换算子测量顺序。每次取四个 rank 的最大时间，再取中位数。
- 为排除修改公共代码对基线的影响，另外用 `git archive 7902831ec4a6691b6268b308bcff5ec0cb15ec86` 的纯净源码运行原 GC。下表基线采用这次独立控制测试，记录见 [pristine.json](benchmarks/gc_moe_ll_ep4_pristine.json)；LL 与实验列取自前述交替测量。两轮基线差异小于 1.6%。
- 交替测量原始数据及所有样本见 [gc_moe_ll_ep4.json](benchmarks/gc_moe_ll_ep4.json)。`ll_stage` 是默认算法，`ll_direct` 是实验直接拉取算法。

| T/rank | 原 GC，µs | 默认 LL，µs | 延迟下降 | 直接拉取实验，µs |
|---:|---:|---:|---:|---:|
| 4 | 48.112 | 42.097 | 12.50% | 66.243 |
| 8 | 48.790 | 41.698 | 14.54% | 69.900 |
| 16 | 50.167 | 44.558 | 11.18% | 83.903 |
| 32 | 58.887 | 54.867 | 6.83% | 97.009 |

三种算子均与独立 Torch 解量化参考比较，参考包含 W1 的 BF16 舍入、SwiGLU、路由权重乘法、K32 FP8 中间量化及 W2 BF16 舍入。以上 T 的均衡、随机、集中到少数专家、无有效路由场景及更换输入后的 Graph 重放均通过。另补测不带共享专家的 T=0/1/7/17/31，上述四种路由与 Graph 重放也全部通过。主性能测试记录最大相对 L2 误差约 0.001218；测试阈值为相对 L2 < 0.025、cosine > 0.999，且必须全部有限。

## 使用与复现

复用 `get_symm_buffer_for_gc_moe` 和 `transform_weights_for_gc_moe` 的权重/输入契约。**共享输入 scale 的排布必须查询 LL 的 BM，且传入与算子相同的 `direct_pull` 值**，不能直接使用原接口的 BM。目前 GLM5.2 均衡参数下，直接路径 T≤32 使用 BM16，默认路径 T≤16 使用 BM16、T=32 使用 BM32：

```python
direct_pull = False  # 实验直接路径改成 True；两处必须一致
bm = gc_moe.get_block_m_for_gc_moe_ll(
    ep, experts, capacity, tokens, topk, direct_pull=direct_pull)
# 按 BM 将 x_sf 放到 buffer.shared_l1_acts_sf，见测试 stage()。
gc_moe.fp8_fp4_gc_moe_ll(
    y, routed_w1, routed_w2, buffer,
    shared_w1, shared_w2, shared_recipe=(1, 128, 128), direct_pull=direct_pull,
)
```

共享 SF 行映射：令 `u=t % BM`，目的行为
`(t//BM)*align(BM,128) + (u//128)*128 + (u%32)*4 + (u%128)//32`。
buffer 内 `x`、`x_sf`、`topk_idx`、`topk_weights` 的准备方式与原接口相同。不要并发复用同一工作区；首次调用/JIT 在 Graph capture 外完成。

```bash
python -m torch.distributed.run --nproc_per_node=4 \
  tests/bench_gc_moe_ll.py --variants gc,ll,ll_direct \
  --graph-calls 50 --repeats 9 --output results/ll/reproduce.json
```

测试不依赖 DeepGEMM，自己生成量化张量和参考输出。默认正好是上述 EP4 形状。EP32 用 32 个 rank 和每卡 8 专家运行；跨机对称显存访问能力需要目标系统实际支持，本次没有验证多机。

实验开关必须在各 rank 一致设置：

- `GC_MOE_LL_L1_GROUP=1/2/4/8`：直接拉取的 W1 N 分组，默认 2。
- `GC_MOE_LL_USE_GREEN=1`：对照 Green 分区；默认 0。本机直接拉取实验下没有收益。
- `GC_MOE_LL_COMPUTE_SMS=<偶数>`：覆盖计算 grid 上限，默认使用扣除路由驻留预算后的 SM 数。缩小到 128 在本次直接拉取实验中没有收益。

主要代码：`apis/gc_moe_ll.py`（契约）、`impls/sm100_fp8_fp4_gc_moe.py`（特化角色）、`common/ll_routes.py`（直接拉取的元数据整理）、`common/direct_pull.py`（CTA 输入缓存）、`scheduler/gc_moe.py`（N 分组）、`runtime/gc_moe.py`（primary-context 调度）。

原有未提交的 `layout/gc_moe_ll.py` 草稿没有接入新接口，也没有删除。新路径暂时复用经过验证的 GC 工作区，特别是环形区代际与输出元数据；该草稿的容量/分片布局不能视为已验证实现。
