# Fused MLP in Triton: tiles, registers and spills

I once spent a while convinced one of my GPU kernels was slow because of how it read memory,
and started planning a rewrite. The real cause was invisible in the code: the compiler had run
out of registers and was quietly spilling them to memory. One launch option fixed it. Since
then I check the compiler's numbers before touching the code, and this repo is a small,
self-contained demo of why.

One Triton kernel computes a transformer-style MLP block, `y = gelu(x @ W1) @ W2`, without
writing the hidden activations to memory. At the right tile size it is **1.2x faster than
PyTorch's cuBLAS path** (two matmuls plus a GELU kernel). The point of the repo is the
part around that number: how tile size and warp count decide the register budget, and how
much it costs when the budget is too small and the compiler spills registers to memory.

## Results

RTX 5080, Triton 3.7.1, PyTorch 2.11 (CUDA 12.8), fp16 in and out, fp32 accumulation.
Shape: `M = 131072` rows, `D = 256`, hidden `H = 256`. PyTorch reference: **565 us**.
Every row is checked against PyTorch (max abs error 0.002). Full output: [results_rtx5080.txt](results_rtx5080.txt).

**1. Tile size and warps** (the compiler picks the registers)

| BLOCK_M | warps | registers / thread | spills | time (us) | vs PyTorch |
|---:|---:|---:|---:|---:|---:|
| 16 | 4 | 92 | 0 | 684 | 0.83x |
| 32 | 4 | 128 | 0 | 604 | 0.94x |
| 64 | 4 | 255 | 0 | 477 | 1.18x |
| 64 | 8 | 128 | 0 | 469 | **1.20x** |
| 128 | any | does not fit in shared memory | | | |

Bigger tiles reuse each weight tile across more rows, so they win, until the tile no longer
fits. At `BLOCK_M = 64` with 4 warps the kernel needs every register a thread can have (255).
8 warps split the same tile over twice the threads: 128 registers each, same speed.

**2. Register budget** at `BLOCK_M = 64`, 4 warps, capped with Triton's `maxnreg`

| maxnreg | spills | time (us) | vs PyTorch |
|---:|---:|---:|---:|
| 64 | 278 | 1938 | 0.29x |
| 96 | 230 | 1511 | 0.37x |
| 128 | 152 | 1040 | 0.54x |
| 160 | 58 | 608 | 0.93x |
| 192 | 4 | 499 | 1.13x |
| 224 | 0 | 484 | 1.17x |
| 255 | 0 | 479 | 1.18x |

The same kernel with the same tile runs **4x slower** when it spills. The results are just as
correct, so nothing in the output warns you; only the compiler's numbers show it.

## What to take from it

- Before rewriting a slow Triton kernel, print `kernel.n_regs` and `kernel.n_spills` (the
  launch returns the compiled kernel). Spills are invisible in the output and expensive.
- Spills mean the tile is too big for the threads holding it: shrink the tile or add warps.
- The opposite also happens: the compiler sometimes allocates **fewer** registers than a
  kernel needs and spills anyway. Raising `maxnreg` (up to `65536 / (num_warps * 32)`, at
  most 255) fixes that without touching the code. In a larger fused kernel of mine this cut
  the time 3x (forward and backward).

## Run it

```
pip install torch triton      # on Windows: triton-windows
python fused_mlp.py           # correctness check + both sweeps
python fused_mlp.py 512       # a different hidden size
```

Needs an NVIDIA GPU. Numbers differ by card: shared memory and register file sizes decide
which tiles fit.

## Not covered

One shape, one GPU, a medium-sized hidden layer. A large hidden size (`H` of a few thousand,
as in real models) needs the hidden dimension tiled too, with partial sums for the second
matmul; this kernel keeps the whole hidden row in registers, so it stops at `H = 512` here.

## Thanks

To Shane ([@ShaneDasPro](https://github.com/ShaneDasPro)), who first got me into Triton.
