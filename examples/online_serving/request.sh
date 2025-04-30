

for i in `seq 1 100`
do
	curl -X POST -s http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{"model": "/hf/Meta-Llama-3.1-8B-Instruct","prompt": "Write a summary of why scarcity and urgency are the strongest mental triggers and have been the driving force behind many of our best performing campaigns over the last 8 years.","max_tokens": 50,"temperature": 0, "top_k":1}' | awk -F'"' '{print $22}' &
	curl -X POST -s http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{"model": "/hf/Meta-Llama-3.1-8B-Instruct","prompt": "Write a summary of why scarcity and urgency are the strongest mental triggers and have been the driving force behind many of our best performing campaigns over the last 8 years.","max_tokens": 200,"temperature": 0, "top_k":1}' | awk -F'"' '{print $22}'

done
