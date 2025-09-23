#!/usr/bin/env python3

import os
import subprocess
import argparse

# Define extension transformation rules
EXTENSION_MAP = {
    ".cu": ".hip.cpp",
    ".cuh": ".hip.h",
    ".cpp": ".cpp",
    ".h": ".h"
}

def hipify_file(input_path, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cmd = ["hipify-perl", input_path]
    with open(output_path, "w") as out_file:
        subprocess.run(cmd, stdout=out_file)
    print(f"HIPified: {input_path} -> {output_path}")

def hipify_directory(src_dir, out_dir=None, in_place=False):
    for dirpath, _, filenames in os.walk(src_dir):
        for filename in filenames:
            base, ext = os.path.splitext(filename)
            if ext in EXTENSION_MAP:
                full_path = os.path.join(dirpath, filename)

                if in_place:
                    # In-place: use original path
                    output_path = full_path
                else:
                    # Mirrored structure, transformed filename if needed
                    relative_path = os.path.relpath(full_path, src_dir)
                    new_ext = EXTENSION_MAP[ext]
                    new_filename = base + new_ext if new_ext != ext else filename
                    output_path = os.path.join(out_dir, os.path.dirname(relative_path), new_filename)

                hipify_file(full_path, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HIPify CUDA/C++ files in a directory.")
    parser.add_argument("src_dir", help="Root directory of source files")
    parser.add_argument("--out-dir", help="Directory to store HIPified output")
    parser.add_argument("--in-place", action="store_true", help="Replace original files")
    args = parser.parse_args()

    if not args.in_place and not args.out_dir:
        parser.error("You must specify --out-dir if not using --in-place")

    hipify_directory(args.src_dir, args.out_dir, args.in_place)

