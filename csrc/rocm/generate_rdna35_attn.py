#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate the translation units of the RDNA3.5 decode-attention variants.

Reads vllm/v1/attention/ops/rdna35_variants.csv -- a header naming the
kernel's compile-time defines, one row per build -- and writes to OUT_DIR one
unit per (configuration, dtype), each building its rows in namespaces of their
own, and rdna35_variants.inc, the registry's table. Prints the units, one per line.

    generate_rdna35_attn.py CSV OUT_DIR
"""

import argparse
import csv
import sys
from pathlib import Path

PRELUDE = """\
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <type_traits>

#include "rdna35_decode_attn.h"
"""


def variant(cols: list[str], row: list[str]) -> str:
    """One build: the kernel in a namespace named after its row, under the
    row's defines, and the launch function the registry calls."""
    vid = "_".join(row)
    defs = "".join(f"#define {c} {v}\n" for c, v in zip(cols, row))
    undefs = "".join(f"#undef {c}\n" for c in cols)
    return (
        f"\n{defs}"
        f"namespace rdna35::v_{vid} {{\n"
        '#include "rdna35_decode_attn.cu"\n'
        f"}}  // namespace rdna35::v_{vid}\n"
        "namespace rdna35 {\n"
        f"void launch_{vid}(const LaunchArgs& a) {{ v_{vid}::launch(a); }}\n"
        "}  // namespace rdna35\n"
        f"{undefs}"
    )


def registry(cols: list[str], rows: list[list[str]]) -> str:
    """The variants' keys (rows, in column order), their launch functions, and
    the index of every column."""
    lines = [
        "// Generated from rdna35_variants.csv.  Do not edit.",
        "#pragma once",
        "",
        "namespace rdna35 {",
        "",
        f"inline constexpr int kNumFields = {len(cols)};",
        *(f"inline constexpr int kCol_{c} = {i};" for i, c in enumerate(cols)),
        "",
        *(f"void launch_{'_'.join(r)}(const LaunchArgs&);" for r in rows),
        "",
        "struct Variant {",
        "  int key[kNumFields];",
        "  LaunchFn fn;",
        "};",
        "",
        "inline constexpr Variant kVariants[] = {",
        *(f"    {{{{{','.join(r)}}}, &launch_{'_'.join(r)}}}," for r in rows),
        "};",
        "",
        "}  // namespace rdna35",
    ]
    return "\n".join(lines) + "\n"


def write_if_changed(path: Path, text: str) -> None:
    """Leave an unchanged file, and its timestamp, alone."""
    if not path.is_file() or path.read_text() != text:
        path.write_text(text)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("csv", type=Path)
    p.add_argument("out_dir", type=Path)
    args = p.parse_args()

    with args.csv.open(newline="") as fh:
        cols, *rows = csv.reader(fh)
    units: dict[str, list[str]] = {}
    for row in rows:
        if len(row) != len(cols) or not all(v.isdigit() for v in row):
            sys.exit(f"{args.csv}: bad row {','.join(row)}")
        f = dict(zip(cols, row))
        dtype = "bf16" if f["BF16"] != "0" else "fp16"
        unit = (
            f"q{f['NUM_Q_HEADS']}_kv{f['NUM_KV_HEADS']}_d{f['HEAD_DIM']}"
            f"_w{f['WINDOW']}_{dtype}"
        )
        units.setdefault(unit, []).append(variant(cols, row))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for unit, bodies in units.items():
        header = f"// Generated from rdna35_variants.csv, unit {unit}.  Do not edit.\n"
        write_if_changed(
            args.out_dir / f"{unit}.hip", header + PRELUDE + "".join(bodies)
        )
        print(args.out_dir / f"{unit}.hip")
    write_if_changed(args.out_dir / "rdna35_variants.inc", registry(cols, rows))
    print(
        f"RDNA35 decode attention: {len(rows)} variants in {len(units)} units",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
