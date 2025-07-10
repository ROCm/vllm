export HIP_VISIBLE_DEVICES=6

export PYTHONPATH=/data/heyanguang/code/vllm_fa_batch_prefill/rocm_vllm:$PYTHONPATH
root_dir=/data/heyanguang/code/vllm_fa_batch_prefill/rocm_vllm

    # --base-url http://10.67.77.128:30000 \


python3 $root_dir/benchmarks/benchmark_serving.py \
    --backend openai \
    --base-url http://10.235.85.9:30000 \
    --model /data/heyanguang/code/models/Llama-3.1-8B-Instruct-FP8-KV \
    --dataset-name random \
    --seed 137 \
    --num-prompts 200
    # --num-prompts 100
    # --num-prompts 20
