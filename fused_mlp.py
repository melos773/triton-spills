"""Fused MLP block  y = gelu(x @ W1) @ W2  in one Triton kernel, plus a sweep that shows
how tile size, warps and maxnreg decide register spills, and what spills cost.

Run:  python fused_mlp.py            (correctness check + sweep table)
"""
import sys
import torch
import triton
import triton.language as tl


@triton.jit
def fused_mlp(x_ptr, w1_ptr, w2_ptr, y_ptr, M,
              D: tl.constexpr, H: tl.constexpr,
              BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    rmask = rows < M
    hcols = tl.arange(0, H)

    # h = x @ W1 : the whole [BLOCK_M, H] fp32 accumulator lives in registers
    acc = tl.zeros((BLOCK_M, H), dtype=tl.float32)
    for k in range(0, D, BLOCK_K):
        ks = k + tl.arange(0, BLOCK_K)
        xt = tl.load(x_ptr + rows[:, None] * D + ks[None, :], mask=rmask[:, None], other=0.0)
        wt = tl.load(w1_ptr + ks[:, None] * H + hcols[None, :])
        acc = tl.dot(xt, wt, acc)

    # tanh-GELU, then back to fp16 for the second matmul
    c = 0.7978845608028654
    inner = c * (acc + 0.044715 * acc * acc * acc)
    th = 2.0 / (1.0 + tl.exp(-2.0 * inner)) - 1.0
    h = (0.5 * acc * (1.0 + th)).to(tl.float16)

    # y = h @ W2, one BLOCK_N column strip at a time
    for n in range(0, D, BLOCK_N):
        ns = n + tl.arange(0, BLOCK_N)
        w2 = tl.load(w2_ptr + hcols[:, None] * D + ns[None, :])
        out = tl.dot(h, w2)
        tl.store(y_ptr + rows[:, None] * D + ns[None, :], out.to(tl.float16), mask=rmask[:, None])


def run(x, w1, w2, block_m, num_warps, maxnreg=None):
    M, D = x.shape
    H = w1.shape[1]
    y = torch.empty_like(x)
    opts = dict(BLOCK_M=block_m, BLOCK_K=32, BLOCK_N=64, num_warps=num_warps, num_stages=2)
    if maxnreg:
        opts["maxnreg"] = maxnreg
    k = fused_mlp[(triton.cdiv(M, block_m),)](x, w1, w2, y, M, D=D, H=H, **opts)
    return y, k


def reference(x, w1, w2):
    return torch.nn.functional.gelu(x @ w1, approximate="tanh") @ w2


def main():
    torch.manual_seed(0)
    M, D, H = 131072, 256, 256
    if len(sys.argv) > 1:
        H = int(sys.argv[1])
    x = torch.randn(M, D, device="cuda", dtype=torch.float16)
    w1 = torch.randn(D, H, device="cuda", dtype=torch.float16) / D ** 0.5
    w2 = torch.randn(H, D, device="cuda", dtype=torch.float16) / H ** 0.5
    ref = reference(x, w1, w2)

    t_torch = triton.testing.do_bench(lambda: reference(x, w1, w2))
    print(f"M={M} D={D} H={H}   torch (cuBLAS, 3 kernels): {t_torch * 1000:7.1f} us\n")
    head = f"{'BLOCK_M':>7} {'warps':>5} {'maxnreg':>7} {'regs':>5} {'spills':>6} {'time us':>8} {'vs torch':>8}  max err"

    def row(block_m, num_warps, maxnreg=None):
        try:
            y, k = run(x, w1, w2, block_m, num_warps, maxnreg)
        except triton.runtime.errors.OutOfResources as e:
            print(f"{block_m:>7} {num_warps:>5} {str(maxnreg or '-'):>7}   does not fit ({e.name})")
            return
        err = (y.float() - ref.float()).abs().max().item()
        assert err < 5e-2, f"wrong result: max err {err}"
        t = triton.testing.do_bench(lambda: run(x, w1, w2, block_m, num_warps, maxnreg))
        print(f"{block_m:>7} {num_warps:>5} {str(maxnreg or '-'):>7} {k.n_regs:>5} {k.n_spills:>6} "
              f"{t * 1000:>8.1f} {t_torch / t:>7.2f}x  {err:.4f}")

    print("1) tile size and warps (compiler picks the registers)\n" + head)
    for block_m in (16, 32, 64, 128):
        for num_warps in (4, 8):
            row(block_m, num_warps)

    print("\n2) register budget at BLOCK_M=64, 4 warps (maxnreg caps registers per thread)\n" + head)
    for maxnreg in (64, 96, 128, 160, 192, 224, 255):
        row(64, 4, maxnreg)


if __name__ == "__main__":
    main()
