# Reference: the per-q-head dot kernel

`rdna35_decode_attn_dot.cu` is the kernel this directory's tools shipped
before OPTIMIZATIONS.md 009, kept byte for byte as of `2132b85be8`: one workgroup
per q head, VALU `fdot2` products with a score butterfly across lanes.
`golden_d512_dot.md` is its last D=512 golden.

It is not built by the backend. Its loader, knobs (`BFLY`, `GRIDT`, `DPL`,
`LDSPLIT`, `MSPLIT`, ...) and tools are the ones at `2132b85be8`:

    git show 2132b85be8:vllm/v1/attention/ops/rdna35_hip_decode.py

It stays as the baseline to beat where it still wins: D=256/512 at M=1 with
few kv heads, mostly at short context (see golden_d512_dot.md against
golden/ for the current kernel).
