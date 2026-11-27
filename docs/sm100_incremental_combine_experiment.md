# SM100 incremental Combine experiment — 2026-11-27

Base: `ad8a4f2`. Branch: `exp/ll-incremental-combine`. Experimental, disabled by default; do not merge this implementation into the default LL path.

## Result

No end-to-end win. The optimized incremental implementation is slower than the committed default LL, and enabling overlap is also slower than launching the identical consumer after W2. This is a negative result for the implemented protocol, not a proof that all incremental Combine designs are inferior.

EP4, eight routed experts per rank, H=6144, I=2048, topk=8, one shared expert, balanced routing. Synthetic quantized weights with GLM geometry; not checkpoint validation. EP4 reproduces local expert/workload sizes, not EP32 communication fan-in. Three distinct-address shadow weight sets precede each timed target call. Final comparison: 21 windows x 24 target calls; each window takes the maximum across four ranks, then median across windows. Original GC and LL alternate inside each process. LL configurations run in separate processes; original GC serves as a drift control.

|T/rank|Original GC (us)|Default LL (us)|Incremental, early (us)|Same consumer, late (us)|Early vs default|
|---:|---:|---:|---:|---:|---:|
|4|53.913|47.813|65.417|62.167|+36.8%|
|8|54.029|48.615|68.564|65.283|+41.0%|
|16|55.629|49.811|76.201|72.516|+53.0%|
|32|65.003|57.309|86.587|84.325|+51.1%|

Producer-only CTA publication (normal compute grid and original Combine) measured T=4: 57.477 us, T=8: 58.448 us, T=16: 61.037 us, T=32: 67.621 us. This isolates the complete producer-side publication modification, not the latency of any single fence or store instruction. Those runs used 7 x 12 calls; raw samples and source hashes are included.

## Implementation

- One 128-thread consumer CTA owns four output chunks through its warps. FP32 partial sums stay in registers; final BF16 is written once. No atomic floating-point accumulation into the final output.
- W2 stores the original BF16 contributions in distinct source-rank slots, then publishes a per-token/per-TopK-slot/per-128-channel epoch.
- The optimized publication reuses the existing 256-thread epilogue CTA barrier. Selected threads issue system-release stores after that join. System-acquire flag reads and warp synchronization precede consumer data loads. No speculative reads of incomplete contributions.
- The original per-writer system fence plus half-warp publication was much more expensive. The tested CTA publication removes those redundant fences and extra warp joins.
- One lane checks each contribution in parallel; flag accesses remain 32-bit to match independent 32-bit publications (no mixed-size racing loads).
- Already-ready contributions are loaded into register staging before accumulation, increasing independent outstanding loads.
- Fixed-order mode preserves the shared-first / TopK order. Arrival mode consumes currently ready slots in scan order and may change floating-point rounding; it is tested against the reference, not promised bitwise-equivalent for arbitrary inputs.
- The early consumer is a separate graph node. Cleanup depends on Dispatch, Compute and all consumers. Epoch advancement and ring cleanup cannot overtake consumption.
- Reserve one compute CTA slot per consumer CTA (rounded configuration is even); tested 2/4/8/16 consumer CTAs. The final setting has 128 compute CTAs and 16 consumer CTAs on the 152-SM device.
- The consumer/footer shares a small binary and uses no shared-memory allocation. Removed the old footer register redistribution: reported consumer registers fell from 255 to 126, with little timing improvement.

## What was tried

Serial versus batched readiness polling; serial versus batched register loads; fixed versus arrival order; 256/512/1024 output elements per chunk; 2/4/8/16 consumer CTAs; per-writer fence versus reuse of the existing CTA join; early versus late graph placement; producer-only control; register-redistribution removal. See the JSON for every completed run, source SHA256, environment, accuracy and raw samples. Failed initial smoke/notification attempts are not performance evidence.

Final explicit configuration (environment is immutable for the lifetime of a benchmark process):

```sh
LL_INC=1 LL_INC_CTAS=16 LL_INC_CHUNK=512 LL_INC_ORDER=arrival \
LL_INC_POLL=batch LL_INC_LOAD=batch LL_INC_PUBLISH=cta LL_INC_OVERLAP=1 \
env -u PYTHONPATH PATH=/usr/local/cuda/bin:$PATH NCCL_DEBUG=WARN \
../gc-ll-venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 \
tests/bench_gc_moe_ll.py --tokens 4,8,16,32 --variants gc,ll_stage \
--patterns balanced --weight-rotations 3 --repeats 21 --graph-calls 24 \
--warmup-graphs 15 --output results/incremental/reproduce.json
```

Set `LL_INC=0` for the committed LL algorithm. Set `LL_INC_OVERLAP=0` for the identical consumer after producers complete. `LL_INC_PRODUCER_ONLY=1` keeps the original Combine and compute grid, but adds W2 publications. This experiment adds a ready region to the symmetric allocation even when disabled; old data offsets and the disabled execution path remain unchanged. Do not switch experimental environment settings inside a process: runtime contexts cache configuration.

## Correctness and evidence

- Arrival mode: T=0,1,4,8,16,31,32 x balanced/random/skew/empty/mixed, four ranks, including graph replay after route changes: passed Torch reference metrics. All 35 sampled cases also happened to match stock bitwise; this is not an arrival-order guarantee.
- Fixed mode: T=4,32 x all five patterns: passed reference and required bitwise stock equality.
- After the register-only cleanup, T=4,32 x all five patterns was rerun successfully.
- Compute Sanitizer memcheck is run with all child processes and a kernel-name filter for the three role kernels (including consumer/footer), API version-probe errors disabled. Sanitizer timings are excluded from performance tables. Final clean rerun completed with `ERROR SUMMARY: 0 errors` for T=4/32 and all five routing patterns. The interrupted `memcheck_final` attempt was invalidated when unrelated GPU processes appeared; it is not counted as a passing run.
- GPU application PIDs and their cwd are sampled every second. New runs wait while other GPU applications are present. A foreign process detected during a run aborts only this experiment and invalidates its result. No foreign application was observed in the recorded completed windows. This is sampled monitoring, not a system-wide reservation.

## Timeline interpretation

The traces contain three shadow calls and a final target call. In incremental traces each call has Dispatch, Compute, consumer, footer; the consumer and footer share a kernel name. Distinguish them by graph node and time. Early consumer duration includes polling and cannot be treated as pure reduction time. Profiler timestamps are diagnostic and are not the unprofiled benchmark.

At T=32, one final-source capture showed the consumer living for about 74 us, Compute for about 56 us, and a further 4 us footer. The consumer continued for about 16 us after Compute. The control also shows producer-side publications already cost roughly 10 us before any overlap consumer is added. These results explain why hiding a short original tail did not repay the new protocol and resource costs. No NCU counters were available in the earlier profiling session, so no hardware-bandwidth saturation claim is made.

Memory-order rationale: [NVIDIA PTX memory model, release/acquire and transitive causality](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#release-and-acquire-patterns). Data stores precede the existing CTA join, which precedes system-release publication; observing that publication with acquire precedes consumption. Sanitizer does not prove cross-GPU ordering.

## Scope

This is an opt-in experimental branch, not a production implementation. Only the single-node four-GPU SM100 setup was validated. No EP32, multi-node, long-duration epoch-wrap or concurrent use of the same buffer guarantee is added. The original working checkout and its other AI changes were not edited.
