curl http://127.0.0.1:9000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "列出三条显存优化技巧：",
    "max_tokens": 64,
    "temperature": 0.7
  }' 
 
