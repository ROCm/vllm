#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate the translation units of the RDNA3.5 attention variants.

Reads a variant list -- vllm/v1/attention/ops/rdna35_variants.csv (decode) or
rdna35_prefill_variants.csv (--kind prefill): a header naming the kernel's
compile-time defines, one row per build -- and writes to OUT_DIR one unit per
(configuration, dtype), each building its rows in namespaces of their own, and
the registry's table (rdna35_variants.inc, rdna35_prefill_variants.inc).
Prints the units, one per line.

    generate_rdna35_attn.py CSV OUT_DIR [--kind decode|prefill]
"""

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Kind:
    """What tells one variant list's generated code from the other's."""

    source: str  # the kernel source each variant includes
    prefix: str  # of namespaces, launch functions, units and table names
    args: str  # the launch argument type
    fn: str  # the launch function type
    table: str  # the registry's include
    names: str  # identifiers of the table: kNumFields, kCol_, Variant, kVariants


KINDS = {
    "decode": Kind(
        "rdna35_decode_attn.cu", "", "LaunchArgs", "LaunchFn", "rdna35_variants.inc", ""
    ),
    "prefill": Kind(
        "rdna35_prefill_attn.cu",
        "prefill_",
        "PrefillArgs",
        "PrefillFn",
        "rdna35_prefill_variants.inc",
        "Prefill",
    ),
}

PRELUDE = """\
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <type_traits>

#include "rdna35_decode_attn.h"
"""


def variant(kind: Kind, cols: list[str], row: list[str]) -> str:
    """One build: the kernel in a namespace named after its row, under the
    row's defines, and the launch function the registry calls."""
    vid = kind.prefix + "_".join(row)
    defs = "".join(f"#define {c} {v}\n" for c, v in zip(cols, row))
    undefs = "".join(f"#undef {c}\n" for c in cols)
    return (
        f"\n{defs}"
        f"namespace rdna35::v_{vid} {{\n"
        f'#include "{kind.source}"\n'
        f"}}  // namespace rdna35::v_{vid}\n"
        "namespace rdna35 {\n"
        f"void launch_{vid}(const {kind.args}& a) {{ v_{vid}::launch(a); }}\n"
        "}  // namespace rdna35\n"
        f"{undefs}"
    )


def registry(kind: Kind, src: str, cols: list[str], rows: list[list[str]]) -> str:
    """The variants' keys (rows, in column order), their launch functions, and
    the index of every column."""
    n = kind.names
    fn = [f"launch_{kind.prefix}{'_'.join(r)}" for r in rows]
    lines = [
        f"// Generated from {src}.  Do not edit.",
        "#pragma once",
        "",
        "namespace rdna35 {",
        "",
        f"inline constexpr int k{n}NumFields = {len(cols)};",
        *(f"inline constexpr int k{n}Col_{c} = {i};" for i, c in enumerate(cols)),
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
        f"RDNA35 {args.kind} attention: {len(rows)} variants in {len(units)} units",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
