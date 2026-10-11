#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate the translation units and registry table of the RDNA3.5
attention variants from a variant CSV, and print the units.

    generate_rdna_singleseq_attn.py CSV OUT_DIR [--kind decode|prefill]
"""

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Kind:
    """Names that differ between the decode and prefill code."""

    source: str  # kernel source
    prefix: str  # of generated identifiers and files
    args: str  # launch argument type
    fn: str  # launch function type
    table: str  # registry include
    names: str  # infix of the table identifiers


KINDS = {
    "decode": Kind(
        "rdna_singleseq_decode_attn.cu",
        "",
        "LaunchArgs",
        "LaunchFn",
        "rdna_singleseq_decode_variants.inc",
        "",
    ),
    "prefill": Kind(
        "rdna_singleseq_prefill_attn.cu",
        "prefill_",
        "PrefillArgs",
        "PrefillFn",
        "rdna_singleseq_prefill_variants.inc",
        "Prefill",
    ),
}

PRELUDE = """\
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <type_traits>

#include "rdna_singleseq_attn_launch.h"
"""


def variant(kind: Kind, cols: list[str], row: list[str]) -> str:
    """One build: the kernel under the row's defines, in its own namespace."""
    vid = kind.prefix + "_".join(row)
    defs = "".join(f"#define {c} {v}\n" for c, v in zip(cols, row))
    undefs = "".join(f"#undef {c}\n" for c in cols)
    return (
        f"\n{defs}"
        f"namespace rdna_singleseq::v_{vid} {{\n"
        f'#include "{kind.source}"\n'
        f"}}  // namespace rdna_singleseq::v_{vid}\n"
        "namespace rdna_singleseq {\n"
        f"void launch_{vid}(const {kind.args}& a) {{ v_{vid}::launch(a); }}\n"
        "}  // namespace rdna_singleseq\n"
        f"{undefs}"
    )


def registry(kind: Kind, src: str, cols: list[str], rows: list[list[str]]) -> str:
    """The registry table: columns, rows and launch functions."""
    n = kind.names
    fn = [f"launch_{kind.prefix}{'_'.join(r)}" for r in rows]
    lines = [
        f"// Generated from {src}.  Do not edit.",
        "#pragma once",
        "",
        "namespace rdna_singleseq {",
        "",
        f"inline constexpr int k{n}NumFields = {len(cols)};",
        *(f"inline constexpr int k{n}Col_{c} = {i};" for i, c in enumerate(cols)),
        f"inline constexpr const char* k{n}Columns[] = {{"
        + ", ".join(f'"{c}"' for c in cols)
        + "};",
        "",
        *(f"void {f}(const {kind.args}&);" for f in fn),
        "",
        f"struct {n}Variant {{",
        f"  int key[k{n}NumFields];",
        f"  {kind.fn} fn;",
        "};",
        "",
        f"inline constexpr {n}Variant k{n}Variants[] = {{",
        *(f"    {{{{{','.join(r)}}}, &{f}}}," for r, f in zip(rows, fn)),
        "};",
        "",
        "}  // namespace rdna_singleseq",
    ]
    return "\n".join(lines) + "\n"


def write_if_changed(path: Path, text: str) -> None:
    """Write only on change, keeping the timestamp otherwise."""
    if not path.is_file() or path.read_text() != text:
        path.write_text(text)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("csv", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument("--kind", choices=KINDS, default="decode")
    args = p.parse_args()
    kind = KINDS[args.kind]

    with args.csv.open(newline="") as fh:
        cols, *rows = csv.reader(fh)
    units: dict[str, list[str]] = {}
    for row in rows:
        if len(row) != len(cols) or not all(v.isdigit() for v in row):
            sys.exit(f"{args.csv}: bad row {','.join(row)}")
        f = dict(zip(cols, row))
        dtype = "bf16" if f["BF16"] != "0" else "fp16"
        unit = (
            f"{kind.prefix}q{f['NUM_Q_HEADS']}_kv{f['NUM_KV_HEADS']}"
            f"_d{f['HEAD_DIM']}_w{f['WINDOW']}_{dtype}"
        )
        units.setdefault(unit, []).append(variant(kind, cols, row))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for unit, bodies in units.items():
        header = f"// Generated from {args.csv.name}, unit {unit}.  Do not edit.\n"
        write_if_changed(
            args.out_dir / f"{unit}.hip", header + PRELUDE + "".join(bodies)
        )
        print(args.out_dir / f"{unit}.hip")
    write_if_changed(
        args.out_dir / kind.table, registry(kind, args.csv.name, cols, rows)
    )
    print(
        f"RDNA single-sequence {args.kind} attention: {len(rows)} variants in "
        f"{len(units)} units",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
