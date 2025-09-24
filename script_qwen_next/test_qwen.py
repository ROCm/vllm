from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "/mnt/raid0/pretrained_model/Qwen/Qwen3-Next-80B-A3B-Instruct"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto",
)

# prepare the model input
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "user", "content": prompt},
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
import torch
with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                profile_memory=False,
                with_stack=True,
                with_flops=False,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    "./", use_gzip=True)) as p:
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=3,
    )
print(p.key_averages().table(
    sort_by="self_cuda_time_total", row_limit=-1))

output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
content = tokenizer.decode(output_ids, skip_special_tokens=True)
print("content:", content)
