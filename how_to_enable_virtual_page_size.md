# HOW TO ENABLE VIRTUAL PAGE SIZE

```bash
# The default value of VLLM_VIRTUAL_PAGE_SZIE_FACTOR  is 1, means nothing to do
# if block size is 1 and VLLM_VIRTUAL_PAGE_SZIE_FACTOR is 16, allocate 16 blocks each time
# if block size is 2 and VLLM_VIRTUAL_PAGE_SZIE_FACTOR is 16, allocate 32 blocks each time
VLLM_VIRTUAL_PAGE_SZIE_FACTOR=16 vllm serve --block-size 1 ...
```