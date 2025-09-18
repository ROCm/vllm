vllm serve /mnt/raid0/models/Qwen3-Coder-480B-A35B-Instruct-FP8-ptpc \
--trust-remote-code \
--disable-log-requests \
--max-model-len 32768 \
--tensor-parallel-size 8 \
--max_seq_len_to_capture 32768 \
--no-enable-prefix-caching \
--enable-expert-parallel \
--compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
--max_num_batched_tokens 32768
 
