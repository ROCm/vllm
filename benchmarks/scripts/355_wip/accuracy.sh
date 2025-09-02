lm_eval \
--model local-completions \
--tasks gsm8k \
--model_args model=Qwen/Qwen3-235B-A22B-FP8 \
--host localhost \
--port 8000 \
--batch_size 100
