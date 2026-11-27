"""Standalone Torch/CuTe correctness + EP4-as-EP32 workload benchmark.

No DeepGEMM dependency. Synthetic weights with GLM5.2 geometry, not model
checkpoint accuracy. Run with torchrun --nproc_per_node=4. Default E=8*world,
so each GPU keeps eight experts and balanced M/expert=T at EP4 and EP32.
"""
from pathlib import Path
import argparse
import json
import hashlib
import os
import statistics
import subprocess
import importlib.metadata
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
import gc_moe


def git_output(*args):
    result = subprocess.run(['git', *args], text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else None


def quantize(x, fp4=False, gran=32, shared_block=False):
    x = x.float()
    g = x.reshape(*x.shape[:-1], x.shape[-1] // gran, gran)
    amax = g.abs().amax(-1)
    if shared_block:
        amax = amax.reshape(-1, 128, amax.shape[-1]).amax(1, keepdim=True).expand(-1, 128, -1).reshape_as(amax)
    scale = torch.exp2(torch.ceil(torch.log2((amax / (6 if fp4 else 448)).clamp_min(1e-10))))
    normalized = g / scale[..., None]
    if fp4:
        bounds = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.], device=x.device)
        code = torch.bucketize(normalized.abs().contiguous(), bounds).to(torch.uint8)
        # round ties to even E2M1 mantissa
        for j in range(7):
            if (j + 1) % 2 == 0:
                code = torch.where(normalized.abs() == bounds[j], j + 1, code)
        code |= (normalized < 0).to(torch.uint8) * 8
        code = code.reshape_as(x)
        payload = (code[..., ::2] | (code[..., 1::2] << 4)).view(torch.int8)
    else:
        payload = normalized.to(torch.float8_e4m3fn).reshape_as(x)
    exponents = (torch.log2(scale).to(torch.int32) + 127).to(torch.uint8)
    exponents = torch.nn.functional.pad(exponents, (0, (-exponents.shape[-1]) % 4))
    sf = exponents.contiguous().view(torch.int32)
    # original SF is MN-major, preserving the transform's strides
    sf = sf.transpose(-1, -2).contiguous().transpose(-1, -2)
    return payload, sf


def dequant(pair, fp4=False, gran=32):
    q, packed = pair
    sf = torch.exp2(packed.contiguous().view(torch.uint8).float() - 127)
    if fp4:
        code = torch.stack((q & 15, (q >> 4) & 15), -1).flatten(-2)
        table = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], device=q.device)
        q = table[(code & 7).long()] * torch.where((code & 8) != 0, -1., 1.)
    return q.float() * sf.repeat_interleave(gran, -1)[..., :q.shape[-1]]


def linear(x, weights, weight=None):
    gate, up = (x @ weights[0].T).to(torch.bfloat16).float().chunk(2, -1)
    mid = gate * torch.sigmoid(gate) * up
    if weight is not None:
        mid *= weight[:, None]
    mid = dequant(quantize(mid))
    return (mid @ weights[1].T).to(torch.bfloat16).float()


def reference(x, idx, prob, raw_weights, shared, rank, world):
    gathered = []
    for a in (x, idx, prob):
        peers = [torch.empty_like(a) for _ in range(world)]
        dist.all_gather(peers, a)
        gathered.append(torch.cat(peers))
    ax, ai, ap = gathered
    result = torch.zeros_like(ax)
    for e in range(raw_weights[0][0].shape[0]):
        rows, slots = (ai == rank * raw_weights[0][0].shape[0] + e).nonzero(as_tuple=True)
        if rows.numel():
            ws = [dequant((q[e], sf[e]), True) for q, sf in raw_weights]
            result.index_add_(0, rows, linear(ax[rows], ws, ap[rows, slots]))
    dist.all_reduce(result)
    result = result[rank * x.shape[0]:(rank+1) * x.shape[0]]
    if shared:
        result += linear(x, [dequant(w, gran=128) for w in shared])
    return result.to(torch.bfloat16)


def stage(buffer, x, idx, prob, tokens, experts, world, topk, variant):
    buffer.x[:tokens].copy_(x[0])
    buffer.x_sf[:tokens].copy_(x[1])
    buffer.topk_idx[:tokens].copy_(idx)
    buffer.topk_weights[:tokens].copy_(prob)
    if buffer.num_shared_experts:
        query = gc_moe.get_block_m_for_gc_moe if variant == "gc" else gc_moe.get_block_m_for_gc_moe_ll
        query_options = {} if variant == "gc" else dict(direct_pull=variant in ("ll_direct", "ll_pipeline"))
        bm = query(world, experts, buffer.num_max_tokens_per_rank, tokens, topk, **query_options)
        rows = torch.arange(tokens, device='cuda')
        u = rows % bm
        sf_rows = rows // bm * ((bm + 127)//128*128) + (u//128)*128 + (u%32)*4 + (u%128)//32
        buffer.shared_l1_acts_sf.zero_()
        buffer.shared_l1_acts_sf[sf_rows] = x[1]


def metrics(actual, expected):
    a, b = actual.float().flatten(), expected.float().flatten()
    rel = (a - b).norm() / b.norm().clamp_min(1e-20)
    cos = (torch.nn.functional.cosine_similarity(a, b, dim=0)
           if b.norm().item() else torch.tensor(float(torch.equal(a,b)),device='cuda'))
    values = torch.tensor([rel, cos, ((a-b).abs().max() if a.numel() else 0.), float(torch.isfinite(a).all())], device='cuda')
    peers = [torch.empty_like(values) for _ in range(dist.get_world_size())]
    dist.all_gather(peers, values)
    vals = [v.tolist() for v in peers]
    assert all(v[0] < .025 and v[1] > .999 and v[3] for v in vals), vals
    return vals


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tokens', default='4,8,16,32')
    p.add_argument('--hidden', type=int, default=6144)
    p.add_argument('--intermediate', type=int, default=2048)
    p.add_argument('--local-experts', type=int, default=8)
    p.add_argument('--topk', type=int, default=8)
    p.add_argument('--shared', type=int, default=1, choices=(0,1))
    p.add_argument('--variants', default='gc,ll_stage,ll_direct')
    p.add_argument('--patterns', default='balanced,random,skew,empty,mixed')
    p.add_argument('--repeats', type=int, default=7)
    p.add_argument('--graph-calls', type=int, default=20)
    p.add_argument('--warmup-graphs', type=int, default=20)
    p.add_argument('--profile', action='store_true')
    p.add_argument('--output', default='results/gc_moe_ll.json')
    args = p.parse_args()
    rank = int(os.environ.get('RANK', '0')); world = int(os.environ.get('WORLD_SIZE', '1'))
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))
    dist.init_process_group('nccl', device_id=torch.device('cuda', int(os.environ.get('LOCAL_RANK', '0'))))
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device('cuda')
    h, i, le, k = args.hidden, args.intermediate, args.local_experts, args.topk
    e = le * world
    assert k <= e
    assert torch.cuda.get_device_capability()[0] == 10
    if rank == 0: print('CONFIG', vars(args), 'EP', world, flush=True)
    torch.manual_seed(17 + rank)
    raw = tuple(quantize(torch.randn(shape, device=device, dtype=torch.bfloat16)*.05, True)
                for shape in ((le, 2*i, h), (le, h, i)))
    weights = gc_moe.transform_weights_for_gc_moe(*raw)
    shared = None; sw = (None, None)
    if args.shared:
        torch.manual_seed(11)
        shared = tuple(quantize(torch.randn(shape, device=device, dtype=torch.bfloat16)*.05,
                                gran=128, shared_block=True)
                       for shape in ((2*i,h),(h,i)))
        sw = gc_moe.transform_weights_for_gc_moe(*shared)
    variants = args.variants.split(',')
    results = []
    for tokens in map(int, args.tokens.split(',')):
        buffers = {v: gc_moe.get_symm_buffer_for_gc_moe(dist.group.WORLD,e,max(tokens,1),k,h,i,args.shared)
                   for v in variants}
        ys = {v: torch.empty((tokens,h),device=device,dtype=torch.bfloat16) for v in variants}
        def call(v):
            fn = gc_moe.fp8_fp4_gc_moe if v == 'gc' else gc_moe.fp8_fp4_gc_moe_ll
            kwargs = {} if v in ('gc', 'll') else dict(direct_pull=v in ('ll_direct', 'll_pipeline'))
            previous = os.environ.get('GC_MOE_LL_DIRECT_PIPELINE')
            if v in ('ll_direct', 'll_pipeline'):
                os.environ['GC_MOE_LL_DIRECT_PIPELINE'] = '1' if v == 'll_pipeline' else '0'
            try:
                fn(ys[v], *weights, buffers[v], *sw, shared_recipe=(1,128,128), **kwargs)
            finally:
                if previous is None:
                    os.environ.pop('GC_MOE_LL_DIRECT_PIPELINE', None)
                else:
                    os.environ['GC_MOE_LL_DIRECT_PIPELINE'] = previous
        graphs = {}
        for pattern_index, pattern in enumerate(args.patterns.split(',')):
            torch.manual_seed(1000 + rank + tokens + pattern_index * 10000)
            x = quantize(torch.randn((tokens,h),device=device)*.5)
            if pattern == 'balanced':
                idx = ((torch.arange(tokens, device=device)[:,None]+rank*tokens)*k + torch.arange(k,device=device)[None,:]) % e
            elif pattern == 'empty':
                idx = torch.full((tokens,k),-1,device=device,dtype=torch.int64)
            elif pattern == 'skew':
                idx = torch.arange(k,device=device)[None,:].expand(tokens,-1).contiguous()
            else:
                idx = torch.rand((tokens,e),device=device).topk(k,dim=-1).indices
            if pattern == 'mixed':
                mask = (torch.arange(tokens*k,device=device).reshape(tokens,k)+rank) % 3 == 0
                idx = idx.masked_fill(mask,-1)
            prob = torch.randn((tokens,k),device=device).softmax(-1)
            expected = reference(dequant(x),idx,prob,raw,shared,rank,world)
            for v in variants:
                stage(buffers[v],x,idx,prob,tokens,e,world,k,v)
                dist.barrier(); torch.cuda.synchronize()
                if rank == 0: print('CHECK',tokens,pattern,v,flush=True)
                call(v); torch.cuda.synchronize()
                errs = metrics(ys[v],expected)
                if v not in graphs:
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        for _ in range(args.graph_calls): call(v)
                    graphs[v] = graph
                # Replay a graph captured with old routing after changing its inputs.
                for _ in range(max(3, args.warmup_graphs) if pattern == 'balanced' else 3): graphs[v].replay()
                torch.cuda.synchronize()
                errs = metrics(ys[v],expected)
                results.append(dict(tokens=tokens,pattern=pattern,variant=v,accuracy=errs))
            if pattern == 'balanced' and args.profile:
                for v in variants:
                    dist.barrier(); torch.cuda.synchronize()
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                           torch.profiler.ProfilerActivity.CUDA]) as prof:
                        graphs[v].replay()
                        torch.cuda.synchronize()
                    if rank == 0:
                        prof.export_chrome_trace(str(Path(args.output).with_suffix('')) + f'_{tokens}_{v}.trace.json')
            if pattern == 'balanced':
                samples = {v:[] for v in variants}
                for rep in range(args.repeats):
                    for v in variants[rep%len(variants):]+variants[:rep%len(variants)]:
                        dist.barrier(); torch.cuda.synchronize()
                        start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        start.record(); graphs[v].replay(); end.record(); end.synchronize()
                        us=torch.tensor(start.elapsed_time(end)*1000/args.graph_calls,device=device)
                        dist.all_reduce(us,op=dist.ReduceOp.MAX)
                        samples[v].append(us.item())
                for v in variants:
                    item=dict(tokens=tokens,pattern=pattern,variant=v,latency_us=statistics.median(samples[v]),samples_us=samples[v])
                    results.append(item)
                    if rank==0: print('BENCH',json.dumps(item),flush=True)
        dist.barrier()
        # Captured graphs retain their buffer/weight owners. Process lifetime
        # cleanup is intentional; do not destroy allocations still in graphs.
    if rank == 0:
        source_hashes = {str(p.relative_to(Path(__file__).resolve().parents[1])):
            hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((Path(__file__).resolve().parents[1] / 'gc_moe').rglob('*.py'))}
        out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(dict(args=vars(args),world=world,torch=torch.__version__,
            capability=torch.cuda.get_device_capability(),gpu=torch.cuda.get_device_name(),
            cuda=torch.version.cuda, cutlass=importlib.metadata.version('nvidia-cutlass-dsl'),
            source_sha256=source_hashes,
            git_commit=git_output('rev-parse','HEAD'),
            git_status=git_output('status','--porcelain'),
            ll_environment={k:v for k,v in os.environ.items() if k.startswith('GC_MOE_LL_')},
            results=results),indent=2))
    dist.destroy_process_group()


if __name__ == '__main__': main()
