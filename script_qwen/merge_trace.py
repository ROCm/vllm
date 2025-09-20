import json
import glob
import os
import re
from tqdm import tqdm

def extract_events(file_path, rank):
    with open(file_path, 'rt') as f:
        trace = json.load(f)
    events = trace.get("traceEvents", trace.get("events", []))
    result = []
    for event in events:
        event["pid"] = rank
        result.append(event)
    result.append({
        "ph": "M",
        "pid": rank,
        "name": "process_name",
        "args": {
            "name": f"{os.path.basename(file_path)}"
        }
    })
    return result

def merge_traces(input_files):
    all_events = []
    print(f"�� 找到 {len(input_files)} 个 trace 文件，开始合并...")
    for f in tqdm(input_files, desc="合并进度"):
        match = re.search(r'wr(\d+)', f)
        rank = int(match.group(1))
        events = extract_events(f, rank)
        all_events.extend(events)
    return {
        "traceEvents": all_events,
        "displayTimeUnit": "ns"
    }

if __name__ == "__main__":
    input_files = sorted(glob.glob("*.json"))
    if not input_files:
        print("❌ 未找到任何 .pt.trace.json.gz 文件！")
        exit(1)

    merged = merge_traces(input_files)

    output_path = "merged_trace.json"
    print(f"💾 写入合并结果到 {output_path} ...")
    with open(output_path, "w") as out:
        json.dump(merged, out)

    print(f"✅ 合并完成，共处理 {len(input_files)} 个文件，输出文件大小约 {os.path.getsize(output_path) / 1024:.1f} KB")


