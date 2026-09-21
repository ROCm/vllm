#!/usr/bin/env python
"""PyTorch reference for the f3a decode-attention correctness control.

Two modes:
  gen   S M seed dir       write q/k/v fp16 binaries + the fp32 reference
  check dir out.bin        compare the HIP output against the reference

Layout (dense, matching the HIP host code before it applies KV_PAD):
  q [M, HQ, D]   k,v [S, HKV, D]   out [M, HQ, D] fp32
Causal: query m sees keys 0 .. (S - M) + m.  GQA: head h -> kv head h//(HQ/HKV).
"""

import os
import sys

import numpy as np
import torch

HQ, HKV, D = 32, 16, 256


def reference(q, k, v, M, S):
    """fp32 reference; q/k/v are fp16 tensors, upcast for the math."""
    gqa = HQ // HKV
    ctx = S - M
    qf = q.float().permute(1, 0, 2)  # [HQ, M, D]
    kf = k.float().permute(1, 0, 2)  # [HKV, S, D]
    vf = v.float().permute(1, 0, 2)
    kf = kf.repeat_interleave(gqa, dim=0)
    vf = vf.repeat_interleave(gqa, dim=0)
    scores = torch.bmm(qf, kf.transpose(1, 2)) * (D ** -0.5)  # [HQ, M, S]
    pos = torch.arange(S).view(1, S)
    lim = (ctx + torch.arange(M)).view(M, 1)
    scores = scores.masked_fill((pos > lim).view(1, M, S), float("-inf"))
    p = torch.softmax(scores, dim=-1)
    out = torch.bmm(p, vf)  # [HQ, M, D]
    return out.permute(1, 0, 2).contiguous()


def gen(S, M, seed, d):
    os.makedirs(d, exist_ok=True)
    g = torch.Generator().manual_seed(seed)
    q = (torch.randn(M, HQ, D, generator=g) * 0.5).half()
    k = (torch.randn(S, HKV, D, generator=g) * 0.5).half()
    v = (torch.randn(S, HKV, D, generator=g) * 0.5).half()
    for name, t in (("q", q), ("k", k), ("v", v)):
        t.numpy().tofile(f"{d}/{name}.bin")
    reference(q, k, v, M, S).numpy().astype(np.float32).tofile(f"{d}/ref.bin")
    print(f"gen S={S} M={M} -> {d}")


def check(d, out_path, M):
    ref = np.fromfile(f"{d}/ref.bin", dtype=np.float32).reshape(M, HQ, D)
    got = np.fromfile(out_path, dtype=np.float32).reshape(M, HQ, D)
    if not np.isfinite(got).all():
        print(f"FAIL M={M}: {(~np.isfinite(got)).sum()} non-finite outputs")
        return 1
    aerr = np.abs(got - ref)
    rerr = aerr / np.maximum(np.abs(ref), 1e-3)
    i = int(np.argmax(aerr))
    ok = aerr.max() <= 2e-2
    print(f"{'PASS' if ok else 'FAIL'} M={M}  max_abs={aerr.max():.3e} "
          f"max_rel={rerr.max():.3e}  worst ref={ref.flat[i]:+.5f} "
          f"got={got.flat[i]:+.5f}  (tol 2e-2)")
    return 0 if ok else 1


if __name__ == "__main__":
    if sys.argv[1] == "gen":
        gen(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5])
    else:
        sys.exit(check(sys.argv[2], sys.argv[3], int(sys.argv[4])))
