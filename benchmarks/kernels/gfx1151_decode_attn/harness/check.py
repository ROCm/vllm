#!/usr/bin/env python
"""F3b validation: the brief's three criteria, not just max_abs.

  gen   S M seed dir     write q/k/v fp16 + fp32 reference
  check dir out.bin M    PASS iff max_rel <= 1e-3 and everything is finite
"""

import os
import sys

import numpy as np
import torch

HQ = int(os.environ.get("HQ", 32))
HKV = int(os.environ.get("HKV", 16))
D = int(os.environ.get("D", 256))
RTOL = 1e-3


def reference(q, k, v, M, S):
    gqa = HQ // HKV
    ctx = S - M
    qf = q.float().permute(1, 0, 2)
    kf = k.float().permute(1, 0, 2).repeat_interleave(gqa, dim=0)
    vf = v.float().permute(1, 0, 2).repeat_interleave(gqa, dim=0)
    scores = torch.bmm(qf, kf.transpose(1, 2)) * (D ** -0.5)
    pos = torch.arange(S).view(1, S)
    lim = (ctx + torch.arange(M)).view(M, 1)
    scores = scores.masked_fill((pos > lim).view(1, M, S), float("-inf"))
    out = torch.bmm(torch.softmax(scores, dim=-1), vf)
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


def check(d, out_path, M, tag=""):
    ref = np.fromfile(f"{d}/ref.bin", dtype=np.float32).reshape(M, HQ, D)
    got = np.fromfile(out_path, dtype=np.float32).reshape(M, HQ, D)
    if not np.isfinite(got).all():
        print(f"FAIL {tag} M={M}: {(~np.isfinite(got)).sum()} non-finite")
        return 1
    aerr = np.abs(got - ref)
    rerr = aerr / np.maximum(np.abs(ref), 1e-3)
    ok = rerr.max() <= RTOL
    print(f"{'PASS' if ok else 'FAIL'} {tag} M={M}  max_rel={rerr.max():.3e} "
          f"max_abs={aerr.max():.3e}  (crit max_rel<={RTOL:.0e})")
    return 0 if ok else 1


if __name__ == "__main__":
    if sys.argv[1] == "gen":
        gen(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5])
    else:
        sys.exit(check(sys.argv[2], sys.argv[3], int(sys.argv[4]),
                       sys.argv[4] if len(sys.argv) > 4 else ""))
