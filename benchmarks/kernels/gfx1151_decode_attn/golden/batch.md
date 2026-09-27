# Golden — batches: many decodes, and mixed

Best measured result for batches of sequences, every configuration of
`shapes.csv` (full attention and sliding window).  `matrix.py` measures one
sequence; this is what a server runs.  Replace only when a run beats this
one, and say in the commit message what changed to earn it.

| | |
| --- | --- |
| commit | `e2c7a309e1be2215d80e76d232cbe4a041006083` (OPTIMIZATIONS.md 027, 028, 033) |
| tool | `tools/batch.py` (`--windowed` for the windows) |
| decode batches | `NqMsS`: N sequences, M query tokens each, S keys; CUDA graphs, as vLLM replays them |
| mixed batches | decodes first, then prefills or extends (`qP` or `qPsS`); eager, as vLLM runs them |
| path | `kernel`: every call on the kernel; `split`: decodes on the kernel, the rest on Triton; `triton`: fallback |
| roofline | one dispatch + every decode sequence's Q, KV (its window), output and block table |
| host | Radeon 8060S (gfx1151), fp16, HND, block size 16, 10 layers rotated over >= 96 MiB |
| torch | 2.12.0+rocm10.1.0a20260803 |

## Full attention

Decode batches: 156 cells, vs Triton geomean 1.35x (min 0.98x, max 9.31x), median 92.7 % of roof.  Mixed: 104 cells, 44 split, 60 left whole on Triton; split cells vs Triton geomean 1.62x (min 1.01x); every cell >= 0.97x.

| models | Hq | Hkv | D | window | batch | graphs | path | roofline us | Triton us | ours us | vs | %roof |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemma-2b-it | 8 | 1 | 256 | - | 2q1s4k | yes | kernel | 35.5 | 50.2 | 40.3 | 1.25x | 88.2 % |
| gemma-2b-it | 8 | 1 | 256 | - | 8q1s4k | yes | kernel | 137.6 | 178.3 | 150.8 | 1.18x | 91.3 % |
| gemma-2b-it | 8 | 1 | 256 | - | 8q4s4k | yes | kernel | 138.4 | 205.6 | 159.7 | 1.29x | 86.7 % |
| gemma-2b-it | 8 | 1 | 256 | - | 32q1s1k | yes | kernel | 138.4 | 188.4 | 155.9 | 1.21x | 88.8 % |
| gemma-2b-it | 8 | 1 | 256 | - | 64q1s1k | yes | kernel | 275.4 | 473.7 | 297.2 | 1.59x | 92.7 % |
| gemma-2b-it | 8 | 1 | 256 | - | 32q4s1k | yes | kernel | 141.6 | 523.3 | 174.6 | 3.00x | 81.1 % |
| gemma-2b-it | 8 | 1 | 256 | - | 4q1s8k_q512 | no | split | - | 1380.9 | 301.7 | 4.58x | - |
| gemma-2b-it | 8 | 1 | 256 | - | 16q1s4k_2q1k | no | split | - | 1192.7 | 875.1 | 1.36x | - |
| gemma-2b-it | 8 | 1 | 256 | - | 16q1s2k_q64s2k | no | split | - | 405.2 | 378.3 | 1.07x | - |
| gemma-2b-it | 8 | 1 | 256 | - | 32q1s1k_q64s1k | no | split | - | 452.0 | 337.3 | 1.34x | - |
| gemma-4-E2B-it | 8 | 1 | 512 | - | 2q1s4k | yes | kernel | 69.6 | 213.3 | 81.9 | 2.60x | 84.9 % |
| gemma-4-E2B-it | 8 | 1 | 512 | - | 8q1s4k | yes | kernel | 273.8 | 719.1 | 324.5 | 2.22x | 84.4 % |
| gemma-4-E2B-it | 8 | 1 | 512 | - | 8q4s4k | yes | kernel | 275.4 | 1070.9 | 328.3 | 3.26x | 83.9 % |
| gemma-4-E2B-it | 8 | 1 | 512 | - | 32q1s1k | yes | kernel | 275.4 | 802.6 | 326.5 | 2.46x | 84.3 % |
| gemma-4-E2B-it | 8 | 1 | 512 | - | 64q1s1k | yes | kernel | 549.3 | 1536.0 | 633.6 | 2.42x | 86.7 % |
| gemma-4-E2B-it | 8 | 1 | 512 | - | 32q4s1k | yes | kernel | 281.7 | 1776.3 | 338.0 | 5.25x | 83.3 % |
| gemma-4-E2B-it | 8 | 1 | 512 | - | 4q1s8k_q512 | no | split | - | 3656.3 | 950.1 | 3.85x | - |
| gemma-4-E2B-it | 8 | 1 | 512 | - | 16q1s4k_2q1k | no | split | - | 4382.6 | 3539.9 | 1.24x | - |
| gemma-4-E2B-it | 8 | 1 | 512 | - | 16q1s2k_q64s2k | no | split | - | 1584.0 | 1320.3 | 1.20x | - |
| gemma-4-E2B-it | 8 | 1 | 512 | - | 32q1s1k_q64s1k | no | split | - | 1587.9 | 1092.1 | 1.45x | - |
| Qwen3.5-0.8B +1 | 8 | 2 | 256 | - | 2q1s4k | yes | kernel | 69.5 | 95.6 | 83.3 | 1.15x | 83.4 % |
| Qwen3.5-0.8B +1 | 8 | 2 | 256 | - | 8q1s4k | yes | kernel | 273.5 | 343.0 | 310.3 | 1.11x | 88.1 % |
| Qwen3.5-0.8B +1 | 8 | 2 | 256 | - | 8q4s4k | yes | kernel | 274.3 | 371.7 | 303.6 | 1.22x | 90.4 % |
| Qwen3.5-0.8B +1 | 8 | 2 | 256 | - | 32q1s1k | yes | kernel | 274.3 | 358.9 | 311.1 | 1.15x | 88.2 % |
| Qwen3.5-0.8B +1 | 8 | 2 | 256 | - | 64q1s1k | yes | kernel | 547.1 | 738.2 | 606.2 | 1.22x | 90.3 % |
| Qwen3.5-0.8B +1 | 8 | 2 | 256 | - | 32q4s1k | yes | kernel | 277.5 | 430.1 | 299.0 | 1.44x | 92.8 % |
| Qwen3.5-0.8B +1 | 8 | 2 | 256 | - | 4q1s8k_q512 | no | split | - | 1741.8 | 490.3 | 3.55x | - |
| Qwen3.5-0.8B +1 | 8 | 2 | 256 | - | 16q1s4k_2q1k | no | split | - | 2019.1 | 1298.2 | 1.56x | - |
| Qwen3.5-0.8B +1 | 8 | 2 | 256 | - | 16q1s2k_q64s2k | no | triton | - | 517.9 | 514.9 | 1.01x | - |
| Qwen3.5-0.8B +1 | 8 | 2 | 256 | - | 32q1s1k_q64s1k | no | triton | - | 442.5 | 444.4 | 1.00x | - |
| gemma-4-E4B-it | 8 | 2 | 512 | - | 2q1s4k | yes | kernel | 137.5 | 315.3 | 147.4 | 2.14x | 93.3 % |
| gemma-4-E4B-it | 8 | 2 | 512 | - | 8q1s4k | yes | kernel | 545.5 | 1111.3 | 587.2 | 1.89x | 92.9 % |
| gemma-4-E4B-it | 8 | 2 | 512 | - | 8q4s4k | yes | kernel | 547.1 | 1246.6 | 655.0 | 1.90x | 83.5 % |
| gemma-4-E4B-it | 8 | 2 | 512 | - | 32q1s1k | yes | kernel | 547.1 | 1163.3 | 587.0 | 1.98x | 93.2 % |
| gemma-4-E4B-it | 8 | 2 | 512 | - | 64q1s1k | yes | kernel | 1092.7 | 2232.7 | 1135.6 | 1.97x | 96.2 % |
| gemma-4-E4B-it | 8 | 2 | 512 | - | 32q4s1k | yes | kernel | 553.5 | 1128.7 | 648.6 | 1.74x | 85.3 % |
| gemma-4-E4B-it | 8 | 2 | 512 | - | 4q1s8k_q512 | no | split | - | 4737.2 | 1227.5 | 3.86x | - |
| gemma-4-E4B-it | 8 | 2 | 512 | - | 16q1s4k_2q1k | no | split | - | 6069.9 | 4097.5 | 1.48x | - |
| gemma-4-E4B-it | 8 | 2 | 512 | - | 16q1s2k_q64s2k | no | split | - | 1637.5 | 1358.3 | 1.21x | - |
| gemma-4-E4B-it | 8 | 2 | 512 | - | 32q1s1k_q64s1k | no | split | - | 1610.2 | 1112.8 | 1.45x | - |
| paligemma2-3b-mix-448-LLM +1 | 8 | 4 | 256 | - | 2q1s4k | yes | kernel | 137.4 | 182.9 | 147.4 | 1.24x | 93.2 % |
| paligemma2-3b-mix-448-LLM +1 | 8 | 4 | 256 | - | 8q1s4k | yes | kernel | 545.3 | 652.8 | 578.6 | 1.13x | 94.2 % |
| paligemma2-3b-mix-448-LLM +1 | 8 | 4 | 256 | - | 8q4s4k | yes | kernel | 546.1 | 677.2 | 607.3 | 1.12x | 89.9 % |
| paligemma2-3b-mix-448-LLM +1 | 8 | 4 | 256 | - | 32q1s1k | yes | kernel | 546.1 | 664.9 | 580.7 | 1.14x | 94.0 % |
| paligemma2-3b-mix-448-LLM +1 | 8 | 4 | 256 | - | 64q1s1k | yes | kernel | 1090.6 | 1521.1 | 1148.3 | 1.32x | 95.0 % |
| paligemma2-3b-mix-448-LLM +1 | 8 | 4 | 256 | - | 32q4s1k | yes | kernel | 549.2 | 821.8 | 605.7 | 1.36x | 90.7 % |
| paligemma2-3b-mix-448-LLM +1 | 8 | 4 | 256 | - | 4q1s8k_q512 | no | split | - | 1939.4 | 771.8 | 2.51x | - |
| paligemma2-3b-mix-448-LLM +1 | 8 | 4 | 256 | - | 16q1s4k_2q1k | no | split | - | 3239.6 | 1937.9 | 1.67x | - |
| paligemma2-3b-mix-448-LLM +1 | 8 | 4 | 256 | - | 16q1s2k_q64s2k | no | split | - | 881.5 | 854.0 | 1.03x | - |
| paligemma2-3b-mix-448-LLM +1 | 8 | 4 | 256 | - | 32q1s1k_q64s1k | no | split | - | 871.4 | 750.4 | 1.16x | - |
| deepseek-vl2-tiny | 10 | 10 | 128 | - | 2q1s4k | yes | kernel | 171.4 | 187.6 | 180.2 | 1.04x | 95.1 % |
| deepseek-vl2-tiny | 10 | 10 | 128 | - | 8q1s4k | yes | kernel | 681.0 | 729.2 | 705.5 | 1.03x | 96.5 % |
| deepseek-vl2-tiny | 10 | 10 | 128 | - | 8q4s4k | yes | kernel | 681.5 | 722.8 | 708.2 | 1.02x | 96.2 % |
| deepseek-vl2-tiny | 10 | 10 | 128 | - | 32q1s1k | yes | kernel | 681.5 | 748.5 | 705.0 | 1.06x | 96.7 % |
| deepseek-vl2-tiny | 10 | 10 | 128 | - | 64q1s1k | yes | kernel | 1361.6 | 1426.9 | 1398.6 | 1.02x | 97.4 % |
| deepseek-vl2-tiny | 10 | 10 | 128 | - | 32q4s1k | yes | kernel | 683.5 | 741.4 | 708.3 | 1.05x | 96.5 % |
| deepseek-vl2-tiny | 10 | 10 | 128 | - | 4q1s8k_q512 | no | triton | - | 1122.5 | 1124.5 | 1.00x | - |
| deepseek-vl2-tiny | 10 | 10 | 128 | - | 16q1s4k_2q1k | no | triton | - | 2150.0 | 2149.9 | 1.00x | - |
| deepseek-vl2-tiny | 10 | 10 | 128 | - | 16q1s2k_q64s2k | no | triton | - | 789.8 | 788.3 | 1.00x | - |
| deepseek-vl2-tiny | 10 | 10 | 128 | - | 32q1s1k_q64s1k | no | triton | - | 778.7 | 783.2 | 0.99x | - |
| Qwen2.5-0.5B-Instruct | 14 | 2 | 64 | - | 2q1s4k | yes | kernel | 18.5 | 24.7 | 21.7 | 1.14x | 85.2 % |
| Qwen2.5-0.5B-Instruct | 14 | 2 | 64 | - | 8q1s4k | yes | kernel | 69.6 | 84.2 | 80.8 | 1.04x | 86.1 % |
| Qwen2.5-0.5B-Instruct | 14 | 2 | 64 | - | 8q4s4k | yes | kernel | 69.9 | 90.9 | 83.9 | 1.08x | 83.3 % |
| Qwen2.5-0.5B-Instruct | 14 | 2 | 64 | - | 32q1s1k | yes | kernel | 69.9 | 89.0 | 82.5 | 1.08x | 84.7 % |
| Qwen2.5-0.5B-Instruct | 14 | 2 | 64 | - | 64q1s1k | yes | kernel | 138.3 | 171.2 | 155.0 | 1.10x | 89.2 % |
| Qwen2.5-0.5B-Instruct | 14 | 2 | 64 | - | 32q4s1k | yes | kernel | 71.3 | 123.5 | 86.2 | 1.43x | 82.8 % |
| Qwen2.5-0.5B-Instruct | 14 | 2 | 64 | - | 4q1s8k_q512 | no | triton | - | 540.3 | 540.6 | 1.00x | - |
| Qwen2.5-0.5B-Instruct | 14 | 2 | 64 | - | 16q1s4k_2q1k | no | triton | - | 478.1 | 477.1 | 1.00x | - |
| Qwen2.5-0.5B-Instruct | 14 | 2 | 64 | - | 16q1s2k_q64s2k | no | triton | - | 142.2 | 142.1 | 1.00x | - |
| Qwen2.5-0.5B-Instruct | 14 | 2 | 64 | - | 32q1s1k_q64s1k | no | triton | - | 142.0 | 142.2 | 1.00x | - |
| gemma-4-12b-it | 16 | 1 | 512 | - | 2q1s4k | yes | kernel | 69.7 | 263.5 | 83.7 | 3.15x | 83.2 % |
| gemma-4-12b-it | 16 | 1 | 512 | - | 8q1s4k | yes | kernel | 274.3 | 698.1 | 338.3 | 2.06x | 81.1 % |
| gemma-4-12b-it | 16 | 1 | 512 | - | 8q4s4k | yes | kernel | 277.5 | 1644.7 | 357.8 | 4.60x | 77.6 % |
| gemma-4-12b-it | 16 | 1 | 512 | - | 32q1s1k | yes | kernel | 277.5 | 1025.9 | 345.5 | 2.97x | 80.3 % |
| gemma-4-12b-it | 16 | 1 | 512 | - | 64q1s1k | yes | kernel | 553.5 | 2083.4 | 654.8 | 3.18x | 84.5 % |
| gemma-4-12b-it | 16 | 1 | 512 | - | 32q4s1k | yes | kernel | 290.2 | 3273.5 | 351.7 | 9.31x | 82.5 % |
| gemma-4-12b-it | 16 | 1 | 512 | - | 4q1s8k_q512 | no | split | - | 3897.3 | 1381.6 | 2.82x | - |
| gemma-4-12b-it | 16 | 1 | 512 | - | 16q1s4k_2q1k | no | split | - | 7000.0 | 6161.0 | 1.14x | - |
| gemma-4-12b-it | 16 | 1 | 512 | - | 16q1s2k_q64s2k | no | split | - | 2655.0 | 2285.9 | 1.16x | - |
| gemma-4-12b-it | 16 | 1 | 512 | - | 32q1s1k_q64s1k | no | split | - | 2558.7 | 1872.7 | 1.37x | - |
| MiniCPM-V-0.53B-bosch +1 | 16 | 2 | 64 | - | 2q1s4k | yes | kernel | 18.5 | 24.7 | 21.7 | 1.13x | 85.2 % |
| MiniCPM-V-0.53B-bosch +1 | 16 | 2 | 64 | - | 8q1s4k | yes | kernel | 69.6 | 84.2 | 80.5 | 1.05x | 86.4 % |
| MiniCPM-V-0.53B-bosch +1 | 16 | 2 | 64 | - | 8q4s4k | yes | kernel | 70.0 | 90.8 | 84.8 | 1.07x | 82.5 % |
| MiniCPM-V-0.53B-bosch +1 | 16 | 2 | 64 | - | 32q1s1k | yes | kernel | 70.0 | 91.4 | 82.7 | 1.11x | 84.6 % |
| MiniCPM-V-0.53B-bosch +1 | 16 | 2 | 64 | - | 64q1s1k | yes | kernel | 138.5 | 172.7 | 155.1 | 1.11x | 89.3 % |
| MiniCPM-V-0.53B-bosch +1 | 16 | 2 | 64 | - | 32q4s1k | yes | kernel | 71.6 | 122.7 | 88.4 | 1.39x | 80.9 % |
| MiniCPM-V-0.53B-bosch +1 | 16 | 2 | 64 | - | 4q1s8k_q512 | no | triton | - | 660.2 | 660.5 | 1.00x | - |
| MiniCPM-V-0.53B-bosch +1 | 16 | 2 | 64 | - | 16q1s4k_2q1k | no | triton | - | 507.4 | 505.8 | 1.00x | - |
| MiniCPM-V-0.53B-bosch +1 | 16 | 2 | 64 | - | 16q1s2k_q64s2k | no | triton | - | 142.4 | 142.3 | 1.00x | - |
| MiniCPM-V-0.53B-bosch +1 | 16 | 2 | 64 | - | 32q1s1k_q64s1k | no | triton | - | 141.8 | 141.3 | 1.00x | - |
| Qwen2.5-3B-Instruct +2 | 16 | 2 | 128 | - | 2q1s4k | yes | kernel | 35.5 | 45.4 | 38.6 | 1.18x | 92.1 % |
| Qwen2.5-3B-Instruct +2 | 16 | 2 | 128 | - | 8q1s4k | yes | kernel | 137.6 | 164.7 | 151.2 | 1.09x | 91.1 % |
| Qwen2.5-3B-Instruct +2 | 16 | 2 | 128 | - | 8q4s4k | yes | kernel | 138.4 | 179.6 | 154.5 | 1.16x | 89.6 % |
| Qwen2.5-3B-Instruct +2 | 16 | 2 | 128 | - | 32q1s1k | yes | kernel | 138.4 | 175.0 | 154.7 | 1.13x | 89.5 % |
| Qwen2.5-3B-Instruct +2 | 16 | 2 | 128 | - | 64q1s1k | yes | kernel | 275.4 | 401.1 | 294.7 | 1.36x | 93.4 % |
| Qwen2.5-3B-Instruct +2 | 16 | 2 | 128 | - | 32q4s1k | yes | kernel | 141.6 | 200.3 | 159.6 | 1.26x | 88.8 % |
| Qwen2.5-3B-Instruct +2 | 16 | 2 | 128 | - | 4q1s8k_q512 | no | triton | - | 756.4 | 756.2 | 1.00x | - |
| Qwen2.5-3B-Instruct +2 | 16 | 2 | 128 | - | 16q1s4k_2q1k | no | triton | - | 737.1 | 761.6 | 0.97x | - |
| Qwen2.5-3B-Instruct +2 | 16 | 2 | 128 | - | 16q1s2k_q64s2k | no | triton | - | 214.6 | 214.2 | 1.00x | - |
| Qwen2.5-3B-Instruct +2 | 16 | 2 | 128 | - | 32q1s1k_q64s1k | no | triton | - | 250.9 | 251.1 | 1.00x | - |
| Qwen3.5-35B-A3B +1 | 16 | 2 | 256 | - | 2q1s4k | yes | kernel | 69.6 | 97.3 | 79.7 | 1.22x | 87.3 % |
| Qwen3.5-35B-A3B +1 | 16 | 2 | 256 | - | 8q1s4k | yes | kernel | 273.8 | 349.8 | 298.0 | 1.17x | 91.9 % |
| Qwen3.5-35B-A3B +1 | 16 | 2 | 256 | - | 8q4s4k | yes | kernel | 275.4 | 422.9 | 317.6 | 1.33x | 86.7 % |
| Qwen3.5-35B-A3B +1 | 16 | 2 | 256 | - | 32q1s1k | yes | kernel | 275.4 | 403.3 | 296.8 | 1.36x | 92.8 % |
| Qwen3.5-35B-A3B +1 | 16 | 2 | 256 | - | 64q1s1k | yes | kernel | 549.3 | 829.7 | 577.4 | 1.44x | 95.1 % |
| Qwen3.5-35B-A3B +1 | 16 | 2 | 256 | - | 32q4s1k | yes | kernel | 281.7 | 527.7 | 316.8 | 1.67x | 88.9 % |
| Qwen3.5-35B-A3B +1 | 16 | 2 | 256 | - | 4q1s8k_q512 | no | split | - | 1801.7 | 553.5 | 3.26x | - |
| Qwen3.5-35B-A3B +1 | 16 | 2 | 256 | - | 16q1s4k_2q1k | no | split | - | 2510.8 | 1761.4 | 1.43x | - |
| Qwen3.5-35B-A3B +1 | 16 | 2 | 256 | - | 16q1s2k_q64s2k | no | triton | - | 526.1 | 527.8 | 1.00x | - |
| Qwen3.5-35B-A3B +1 | 16 | 2 | 256 | - | 32q1s1k_q64s1k | no | triton | - | 535.4 | 536.3 | 1.00x | - |
| gemma-4-26B-A4B-it | 16 | 2 | 512 | - | 2q1s4k | yes | kernel | 137.6 | 316.9 | 147.4 | 2.15x | 93.3 % |
| gemma-4-26B-A4B-it | 16 | 2 | 512 | - | 8q1s4k | yes | kernel | 546.1 | 1158.3 | 597.8 | 1.94x | 91.3 % |
| gemma-4-26B-A4B-it | 16 | 2 | 512 | - | 8q4s4k | yes | kernel | 549.2 | 1710.0 | 630.2 | 2.71x | 87.2 % |
| gemma-4-26B-A4B-it | 16 | 2 | 512 | - | 32q1s1k | yes | kernel | 549.2 | 1266.3 | 598.6 | 2.12x | 91.7 % |
| gemma-4-26B-A4B-it | 16 | 2 | 512 | - | 64q1s1k | yes | kernel | 1097.0 | 2480.0 | 1140.1 | 2.18x | 96.2 % |
| gemma-4-26B-A4B-it | 16 | 2 | 512 | - | 32q4s1k | yes | kernel | 562.0 | 1664.9 | 614.4 | 2.71x | 91.5 % |
| gemma-4-26B-A4B-it | 16 | 2 | 512 | - | 4q1s8k_q512 | no | split | - | 4928.1 | 1656.4 | 2.98x | - |
| gemma-4-26B-A4B-it | 16 | 2 | 512 | - | 16q1s4k_2q1k | no | split | - | 8832.5 | 6758.9 | 1.31x | - |
| gemma-4-26B-A4B-it | 16 | 2 | 512 | - | 16q1s2k_q64s2k | no | split | - | 2508.7 | 2078.8 | 1.21x | - |
| gemma-4-26B-A4B-it | 16 | 2 | 512 | - | 32q1s1k_q64s1k | no | split | - | 1652.1 | 1635.3 | 1.01x | - |
| Qwen3.5-9B | 16 | 4 | 256 | - | 2q1s4k | yes | kernel | 137.5 | 181.9 | 147.4 | 1.23x | 93.3 % |
| Qwen3.5-9B | 16 | 4 | 256 | - | 8q1s4k | yes | kernel | 545.5 | 663.0 | 579.4 | 1.14x | 94.2 % |
| Qwen3.5-9B | 16 | 4 | 256 | - | 8q4s4k | yes | kernel | 547.1 | 701.3 | 597.7 | 1.17x | 91.5 % |
| Qwen3.5-9B | 16 | 4 | 256 | - | 32q1s1k | yes | kernel | 547.1 | 718.0 | 583.4 | 1.23x | 93.8 % |
| Qwen3.5-9B | 16 | 4 | 256 | - | 64q1s1k | yes | kernel | 1092.7 | 1527.4 | 1151.3 | 1.33x | 94.9 % |
| Qwen3.5-9B | 16 | 4 | 256 | - | 32q4s1k | yes | kernel | 553.5 | 825.9 | 597.2 | 1.38x | 92.7 % |
| Qwen3.5-9B | 16 | 4 | 256 | - | 4q1s8k_q512 | no | split | - | 2002.4 | 847.5 | 2.36x | - |
| Qwen3.5-9B | 16 | 4 | 256 | - | 16q1s4k_2q1k | no | split | - | 3732.3 | 2440.4 | 1.53x | - |
| Qwen3.5-9B | 16 | 4 | 256 | - | 16q1s2k_q64s2k | no | split | - | 1126.6 | 893.6 | 1.26x | - |
| Qwen3.5-9B | 16 | 4 | 256 | - | 32q1s1k_q64s1k | no | split | - | 881.5 | 754.8 | 1.17x | - |
| Qwen3-1.7B +2 | 16 | 8 | 128 | - | 2q1s4k | yes | kernel | 137.4 | 150.8 | 147.3 | 1.02x | 93.3 % |
| Qwen3-1.7B +2 | 16 | 8 | 128 | - | 8q1s4k | yes | kernel | 545.3 | 589.6 | 584.8 | 1.01x | 93.2 % |
| Qwen3-1.7B +2 | 16 | 8 | 128 | - | 8q4s4k | yes | kernel | 546.1 | 604.8 | 583.8 | 1.04x | 93.5 % |
| Qwen3-1.7B +2 | 16 | 8 | 128 | - | 32q1s1k | yes | kernel | 546.1 | 604.2 | 576.8 | 1.05x | 94.7 % |
| Qwen3-1.7B +2 | 16 | 8 | 128 | - | 64q1s1k | yes | kernel | 1090.6 | 1151.3 | 1137.4 | 1.01x | 95.9 % |
| Qwen3-1.7B +2 | 16 | 8 | 128 | - | 32q4s1k | yes | kernel | 549.2 | 593.2 | 581.7 | 1.02x | 94.4 % |
| Qwen3-1.7B +2 | 16 | 8 | 128 | - | 4q1s8k_q512 | no | triton | - | 1093.1 | 1092.6 | 1.00x | - |
| Qwen3-1.7B +2 | 16 | 8 | 128 | - | 16q1s4k_2q1k | no | triton | - | 2032.3 | 2028.7 | 1.00x | - |
| Qwen3-1.7B +2 | 16 | 8 | 128 | - | 16q1s2k_q64s2k | no | triton | - | 667.4 | 666.0 | 1.00x | - |
| Qwen3-1.7B +2 | 16 | 8 | 128 | - | 32q1s1k_q64s1k | no | triton | - | 677.5 | 675.4 | 1.00x | - |
| gemma-3-12b-it | 16 | 8 | 256 | - | 2q1s4k | yes | kernel | 273.4 | 344.1 | 289.2 | 1.19x | 94.5 % |
| gemma-3-12b-it | 16 | 8 | 256 | - | 8q1s4k | yes | kernel | 1089.0 | 1271.9 | 1150.1 | 1.11x | 94.7 % |
| gemma-3-12b-it | 16 | 8 | 256 | - | 8q4s4k | yes | kernel | 1090.6 | 1682.1 | 1209.3 | 1.39x | 90.2 % |
| gemma-3-12b-it | 16 | 8 | 256 | - | 32q1s1k | yes | kernel | 1090.6 | 1514.1 | 1144.8 | 1.32x | 95.3 % |
| gemma-3-12b-it | 16 | 8 | 256 | - | 64q1s1k | yes | kernel | 2179.7 | 2970.1 | 2276.3 | 1.30x | 95.8 % |
| gemma-3-12b-it | 16 | 8 | 256 | - | 32q4s1k | yes | kernel | 1097.0 | 1531.9 | 1197.3 | 1.28x | 91.6 % |
| gemma-3-12b-it | 16 | 8 | 256 | - | 4q1s8k_q512 | no | split | - | 3364.4 | 1426.6 | 2.36x | - |
| gemma-3-12b-it | 16 | 8 | 256 | - | 16q1s4k_2q1k | no | split | - | 6492.5 | 3702.9 | 1.75x | - |
| gemma-3-12b-it | 16 | 8 | 256 | - | 16q1s2k_q64s2k | no | split | - | 1723.3 | 1473.1 | 1.17x | - |
| gemma-3-12b-it | 16 | 8 | 256 | - | 32q1s1k_q64s1k | no | split | - | 1710.9 | 1328.1 | 1.29x | - |
| Qwen3.6-27B | 24 | 4 | 256 | - | 2q1s4k | yes | kernel | 137.6 | 181.2 | 148.5 | 1.22x | 92.6 % |
| Qwen3.6-27B | 24 | 4 | 256 | - | 8q1s4k | yes | kernel | 545.8 | 667.2 | 579.6 | 1.15x | 94.2 % |
| Qwen3.6-27B | 24 | 4 | 256 | - | 8q4s4k | yes | kernel | 548.2 | 767.2 | 612.7 | 1.25x | 89.5 % |
| Qwen3.6-27B | 24 | 4 | 256 | - | 32q1s1k | yes | kernel | 548.2 | 756.2 | 575.4 | 1.31x | 95.3 % |
| Qwen3.6-27B | 24 | 4 | 256 | - | 64q1s1k | yes | kernel | 1094.9 | 1536.3 | 1134.6 | 1.35x | 96.5 % |
| Qwen3.6-27B | 24 | 4 | 256 | - | 32q4s1k | yes | kernel | 557.7 | 945.0 | 607.8 | 1.55x | 91.8 % |
| Qwen3.6-27B | 24 | 4 | 256 | - | 4q1s8k_q512 | no | split | - | 2057.4 | 954.1 | 2.16x | - |
| Qwen3.6-27B | 24 | 4 | 256 | - | 16q1s4k_2q1k | no | split | - | 4416.4 | 3109.2 | 1.42x | - |
| Qwen3.6-27B | 24 | 4 | 256 | - | 16q1s2k_q64s2k | no | split | - | 1185.3 | 1164.4 | 1.02x | - |
| Qwen3.6-27B | 24 | 4 | 256 | - | 32q1s1k_q64s1k | no | split | - | 1002.4 | 893.9 | 1.12x | - |
| Llama-3.2-3B-Instruct +1 | 24 | 8 | 128 | - | 2q1s4k | yes | kernel | 137.5 | 151.2 | 145.3 | 1.04x | 94.6 % |
| Llama-3.2-3B-Instruct +1 | 24 | 8 | 128 | - | 8q1s4k | yes | kernel | 545.4 | 591.9 | 573.3 | 1.03x | 95.1 % |
| Llama-3.2-3B-Instruct +1 | 24 | 8 | 128 | - | 8q4s4k | yes | kernel | 546.6 | 614.2 | 574.8 | 1.07x | 95.1 % |
| Llama-3.2-3B-Instruct +1 | 24 | 8 | 128 | - | 32q1s1k | yes | kernel | 546.6 | 600.2 | 574.0 | 1.05x | 95.2 % |
| Llama-3.2-3B-Instruct +1 | 24 | 8 | 128 | - | 64q1s1k | yes | kernel | 1091.7 | 1163.3 | 1129.7 | 1.03x | 96.6 % |
| Llama-3.2-3B-Instruct +1 | 24 | 8 | 128 | - | 32q4s1k | yes | kernel | 551.4 | 597.3 | 576.9 | 1.04x | 95.6 % |
| Llama-3.2-3B-Instruct +1 | 24 | 8 | 128 | - | 4q1s8k_q512 | no | triton | - | 1109.7 | 1109.5 | 1.00x | - |
| Llama-3.2-3B-Instruct +1 | 24 | 8 | 128 | - | 16q1s4k_2q1k | no | triton | - | 2089.1 | 2096.9 | 1.00x | - |
| Llama-3.2-3B-Instruct +1 | 24 | 8 | 128 | - | 16q1s2k_q64s2k | no | triton | - | 764.9 | 761.3 | 1.00x | - |
| Llama-3.2-3B-Instruct +1 | 24 | 8 | 128 | - | 32q1s1k_q64s1k | no | triton | - | 700.7 | 698.7 | 1.00x | - |
| Qwen2.5-7B-Instruct +4 | 28 | 4 | 128 | - | 2q1s4k | yes | kernel | 69.5 | 81.2 | 77.6 | 1.05x | 89.6 % |
| Qwen2.5-7B-Instruct +4 | 28 | 4 | 128 | - | 8q1s4k | yes | kernel | 273.7 | 305.1 | 301.9 | 1.01x | 90.7 % |
| Qwen2.5-7B-Instruct +4 | 28 | 4 | 128 | - | 8q4s4k | yes | kernel | 275.1 | 355.2 | 320.6 | 1.11x | 85.8 % |
| Qwen2.5-7B-Instruct +4 | 28 | 4 | 128 | - | 32q1s1k | yes | kernel | 275.1 | 343.3 | 302.4 | 1.14x | 91.0 % |
| Qwen2.5-7B-Instruct +4 | 28 | 4 | 128 | - | 64q1s1k | yes | kernel | 548.7 | 600.0 | 598.0 | 1.00x | 91.8 % |
| Qwen2.5-7B-Instruct +4 | 28 | 4 | 128 | - | 32q4s1k | yes | kernel | 280.7 | 340.0 | 316.4 | 1.07x | 88.7 % |
| Qwen2.5-7B-Instruct +4 | 28 | 4 | 128 | - | 4q1s8k_q512 | no | triton | - | 819.8 | 820.1 | 1.00x | - |
| Qwen2.5-7B-Instruct +4 | 28 | 4 | 128 | - | 16q1s4k_2q1k | no | triton | - | 1365.9 | 1363.7 | 1.00x | - |
| Qwen2.5-7B-Instruct +4 | 28 | 4 | 128 | - | 16q1s2k_q64s2k | no | triton | - | 514.5 | 515.8 | 1.00x | - |
| Qwen2.5-7B-Instruct +4 | 28 | 4 | 128 | - | 32q1s1k_q64s1k | no | triton | - | 424.1 | 423.7 | 1.00x | - |
| MiniCPM-V-custom-checkpoint-2400 | 32 | 2 | 128 | - | 2q1s4k | yes | kernel | 35.6 | 50.6 | 38.8 | 1.30x | 91.6 % |
| MiniCPM-V-custom-checkpoint-2400 | 32 | 2 | 128 | - | 8q1s4k | yes | kernel | 137.9 | 169.0 | 151.4 | 1.12x | 91.1 % |
| MiniCPM-V-custom-checkpoint-2400 | 32 | 2 | 128 | - | 8q4s4k | yes | kernel | 139.5 | 242.1 | 174.8 | 1.38x | 79.8 % |
| MiniCPM-V-custom-checkpoint-2400 | 32 | 2 | 128 | - | 32q1s1k | yes | kernel | 139.5 | 188.7 | 155.0 | 1.22x | 90.0 % |
| MiniCPM-V-custom-checkpoint-2400 | 32 | 2 | 128 | - | 64q1s1k | yes | kernel | 277.5 | 525.9 | 297.9 | 1.77x | 93.2 % |
| MiniCPM-V-custom-checkpoint-2400 | 32 | 2 | 128 | - | 32q4s1k | yes | kernel | 145.9 | 266.4 | 164.6 | 1.62x | 88.6 % |
| MiniCPM-V-custom-checkpoint-2400 | 32 | 2 | 128 | - | 4q1s8k_q512 | no | triton | - | 782.6 | 785.0 | 1.00x | - |
| MiniCPM-V-custom-checkpoint-2400 | 32 | 2 | 128 | - | 16q1s4k_2q1k | no | triton | - | 1035.5 | 1050.1 | 0.99x | - |
| MiniCPM-V-custom-checkpoint-2400 | 32 | 2 | 128 | - | 16q1s2k_q64s2k | no | triton | - | 360.9 | 360.9 | 1.00x | - |
| MiniCPM-V-custom-checkpoint-2400 | 32 | 2 | 128 | - | 32q1s1k_q64s1k | no | triton | - | 269.3 | 268.3 | 1.00x | - |
| Qwen3-30B-A3B-Instruct-2507 +1 | 32 | 4 | 128 | - | 2q1s4k | yes | kernel | 69.6 | 81.3 | 77.5 | 1.05x | 89.7 % |
| Qwen3-30B-A3B-Instruct-2507 +1 | 32 | 4 | 128 | - | 8q1s4k | yes | kernel | 273.8 | 306.9 | 302.4 | 1.01x | 90.5 % |
| Qwen3-30B-A3B-Instruct-2507 +1 | 32 | 4 | 128 | - | 8q4s4k | yes | kernel | 275.4 | 368.4 | 312.9 | 1.18x | 88.0 % |
| Qwen3-30B-A3B-Instruct-2507 +1 | 32 | 4 | 128 | - | 32q1s1k | yes | kernel | 275.4 | 355.3 | 292.0 | 1.22x | 94.3 % |
| Qwen3-30B-A3B-Instruct-2507 +1 | 32 | 4 | 128 | - | 64q1s1k | yes | kernel | 549.3 | 599.1 | 577.0 | 1.04x | 95.2 % |
| Qwen3-30B-A3B-Instruct-2507 +1 | 32 | 4 | 128 | - | 32q4s1k | yes | kernel | 281.7 | 340.5 | 303.1 | 1.12x | 93.0 % |
| Qwen3-30B-A3B-Instruct-2507 +1 | 32 | 4 | 128 | - | 4q1s8k_q512 | no | triton | - | 817.0 | 820.8 | 1.00x | - |
| Qwen3-30B-A3B-Instruct-2507 +1 | 32 | 4 | 128 | - | 16q1s4k_2q1k | no | triton | - | 1429.8 | 1427.5 | 1.00x | - |
| Qwen3-30B-A3B-Instruct-2507 +1 | 32 | 4 | 128 | - | 16q1s2k_q64s2k | no | triton | - | 515.7 | 517.0 | 1.00x | - |
| Qwen3-30B-A3B-Instruct-2507 +1 | 32 | 4 | 128 | - | 32q1s1k_q64s1k | no | triton | - | 424.6 | 423.9 | 1.00x | - |
| gemma-4-31B-it-AWQ +1 | 32 | 4 | 512 | - | 2q1s4k | yes | kernel | 273.8 | 734.4 | 288.2 | 2.55x | 95.0 % |
| gemma-4-31B-it-AWQ +1 | 32 | 4 | 512 | - | 8q1s4k | yes | kernel | 1090.6 | 2696.3 | 1154.6 | 2.34x | 94.5 % |
| gemma-4-31B-it-AWQ +1 | 32 | 4 | 512 | - | 8q4s4k | yes | kernel | 1097.0 | 3717.4 | 1239.5 | 3.00x | 88.5 % |
| gemma-4-31B-it-AWQ +1 | 32 | 4 | 512 | - | 32q1s1k | yes | kernel | 1097.0 | 2732.3 | 1141.3 | 2.39x | 96.1 % |
| gemma-4-31B-it-AWQ +1 | 32 | 4 | 512 | - | 64q1s1k | yes | kernel | 2192.4 | 4508.2 | 2251.2 | 2.00x | 97.4 % |
| gemma-4-31B-it-AWQ +1 | 32 | 4 | 512 | - | 32q4s1k | yes | kernel | 1122.4 | 3365.3 | 1186.9 | 2.84x | 94.6 % |
| gemma-4-31B-it-AWQ +1 | 32 | 4 | 512 | - | 4q1s8k_q512 | no | split | - | 6012.1 | 3044.5 | 1.97x | - |
| gemma-4-31B-it-AWQ +1 | 32 | 4 | 512 | - | 16q1s4k_2q1k | no | split | - | 17924.6 | 13254.6 | 1.35x | - |
| gemma-4-31B-it-AWQ +1 | 32 | 4 | 512 | - | 16q1s2k_q64s2k | no | split | - | 4245.2 | 3073.6 | 1.38x | - |
| gemma-4-31B-it-AWQ +1 | 32 | 4 | 512 | - | 32q1s1k_q64s1k | no | split | - | 3285.1 | 2139.2 | 1.54x | - |
| Llama-3.2-1B-Instruct | 32 | 8 | 64 | - | 2q1s4k | yes | kernel | 69.5 | 87.6 | 76.5 | 1.15x | 90.8 % |
| Llama-3.2-1B-Instruct | 32 | 8 | 64 | - | 8q1s4k | yes | kernel | 273.5 | 327.4 | 293.0 | 1.12x | 93.4 % |
| Llama-3.2-1B-Instruct | 32 | 8 | 64 | - | 8q4s4k | yes | kernel | 274.3 | 440.6 | 294.2 | 1.50x | 93.2 % |
| Llama-3.2-1B-Instruct | 32 | 8 | 64 | - | 32q1s1k | yes | kernel | 274.3 | 298.8 | 296.8 | 1.01x | 92.4 % |
| Llama-3.2-1B-Instruct | 32 | 8 | 64 | - | 64q1s1k | yes | kernel | 547.1 | 622.3 | 589.7 | 1.06x | 92.8 % |
| Llama-3.2-1B-Instruct | 32 | 8 | 64 | - | 32q4s1k | yes | kernel | 277.5 | 322.4 | 299.2 | 1.08x | 92.7 % |
| Llama-3.2-1B-Instruct | 32 | 8 | 64 | - | 4q1s8k_q512 | no | triton | - | 934.0 | 934.7 | 1.00x | - |
| Llama-3.2-1B-Instruct | 32 | 8 | 64 | - | 16q1s4k_2q1k | no | triton | - | 1367.5 | 1365.7 | 1.00x | - |
| Llama-3.2-1B-Instruct | 32 | 8 | 64 | - | 16q1s2k_q64s2k | no | triton | - | 412.6 | 414.0 | 1.00x | - |
| Llama-3.2-1B-Instruct | 32 | 8 | 64 | - | 32q1s1k_q64s1k | no | triton | - | 347.8 | 348.0 | 1.00x | - |
| Qwen3-4B +7 | 32 | 8 | 128 | - | 2q1s4k | yes | kernel | 137.5 | 150.7 | 146.8 | 1.03x | 93.6 % |
| Qwen3-4B +7 | 32 | 8 | 128 | - | 8q1s4k | yes | kernel | 545.5 | 596.9 | 583.2 | 1.02x | 93.5 % |
| Qwen3-4B +7 | 32 | 8 | 128 | - | 8q4s4k | yes | kernel | 547.1 | 614.4 | 583.5 | 1.05x | 93.8 % |
| Qwen3-4B +7 | 32 | 8 | 128 | - | 32q1s1k | yes | kernel | 547.1 | 590.0 | 578.6 | 1.02x | 94.6 % |
| Qwen3-4B +7 | 32 | 8 | 128 | - | 64q1s1k | yes | kernel | 1092.7 | 1158.6 | 1140.0 | 1.02x | 95.9 % |
| Qwen3-4B +7 | 32 | 8 | 128 | - | 32q4s1k | yes | kernel | 553.5 | 600.6 | 588.2 | 1.02x | 94.1 % |
| Qwen3-4B +7 | 32 | 8 | 128 | - | 4q1s8k_q512 | no | triton | - | 1105.4 | 1104.1 | 1.00x | - |
| Qwen3-4B +7 | 32 | 8 | 128 | - | 16q1s4k_2q1k | no | triton | - | 2322.6 | 2330.1 | 1.00x | - |
| Qwen3-4B +7 | 32 | 8 | 128 | - | 16q1s2k_q64s2k | no | triton | - | 815.0 | 814.8 | 1.00x | - |
| Qwen3-4B +7 | 32 | 8 | 128 | - | 32q1s1k_q64s1k | no | triton | - | 705.8 | 705.3 | 1.00x | - |
| SmolLM2-1.7B-Instruct | 32 | 32 | 64 | - | 2q1s4k | yes | kernel | 273.3 | 307.1 | 293.7 | 1.05x | 93.0 % |
| SmolLM2-1.7B-Instruct | 32 | 32 | 64 | - | 8q1s4k | yes | kernel | 1088.7 | 1155.9 | 1181.9 | 0.98x | 92.1 % |
| SmolLM2-1.7B-Instruct | 32 | 32 | 64 | - | 8q4s4k | yes | kernel | 1089.5 | 1182.9 | 1181.6 | 1.00x | 92.2 % |
| SmolLM2-1.7B-Instruct | 32 | 32 | 64 | - | 32q1s1k | yes | kernel | 1089.5 | 1190.4 | 1160.1 | 1.03x | 93.9 % |
| SmolLM2-1.7B-Instruct | 32 | 32 | 64 | - | 64q1s1k | yes | kernel | 2177.6 | 2321.7 | 2304.5 | 1.01x | 94.5 % |
| SmolLM2-1.7B-Instruct | 32 | 32 | 64 | - | 32q4s1k | yes | kernel | 1092.7 | 1212.1 | 1164.8 | 1.04x | 93.8 % |
| SmolLM2-1.7B-Instruct | 32 | 32 | 64 | - | 4q1s8k_q512 | no | triton | - | 1928.7 | 1937.5 | 1.00x | - |
| SmolLM2-1.7B-Instruct | 32 | 32 | 64 | - | 16q1s4k_2q1k | no | triton | - | 3916.1 | 3924.8 | 1.00x | - |
| SmolLM2-1.7B-Instruct | 32 | 32 | 64 | - | 16q1s2k_q64s2k | no | triton | - | 1298.2 | 1304.7 | 1.00x | - |
| SmolLM2-1.7B-Instruct | 32 | 32 | 64 | - | 32q1s1k_q64s1k | no | triton | - | 1239.0 | 1238.9 | 1.00x | - |
| Llama-2-7B +1 | 32 | 32 | 128 | - | 2q1s4k | yes | kernel | 545.1 | 609.3 | 567.9 | 1.07x | 96.0 % |
| Llama-2-7B +1 | 32 | 32 | 128 | - | 8q1s4k | yes | kernel | 2176.0 | 2290.2 | 2256.3 | 1.02x | 96.4 % |
| Llama-2-7B +1 | 32 | 32 | 128 | - | 8q4s4k | yes | kernel | 2177.5 | 2295.1 | 2258.6 | 1.02x | 96.4 % |
| Llama-2-7B +1 | 32 | 32 | 128 | - | 32q1s1k | yes | kernel | 2177.5 | 2255.8 | 2244.3 | 1.01x | 97.0 % |
| Llama-2-7B +1 | 32 | 32 | 128 | - | 64q1s1k | yes | kernel | 4353.6 | 4471.9 | 4486.0 | 1.00x | 97.0 % |
| Llama-2-7B +1 | 32 | 32 | 128 | - | 32q4s1k | yes | kernel | 2183.9 | 2287.4 | 2260.4 | 1.01x | 96.6 % |
| Llama-2-7B +1 | 32 | 32 | 128 | - | 4q1s8k_q512 | no | triton | - | 3662.8 | 3679.8 | 1.00x | - |
| Llama-2-7B +1 | 32 | 32 | 128 | - | 16q1s4k_2q1k | no | triton | - | 6747.9 | 6765.4 | 1.00x | - |
| Llama-2-7B +1 | 32 | 32 | 128 | - | 16q1s2k_q64s2k | no | triton | - | 2539.8 | 2544.4 | 1.00x | - |
| Llama-2-7B +1 | 32 | 32 | 128 | - | 32q1s1k_q64s1k | no | triton | - | 2421.7 | 2417.5 | 1.00x | - |
| NVIDIA-Nemotron-3-Nano-4B | 40 | 8 | 128 | - | 2q1s4k | yes | kernel | 137.5 | 150.8 | 145.3 | 1.04x | 94.6 % |
| NVIDIA-Nemotron-3-Nano-4B | 40 | 8 | 128 | - | 8q1s4k | yes | kernel | 545.7 | 603.8 | 572.9 | 1.05x | 95.2 % |
| NVIDIA-Nemotron-3-Nano-4B | 40 | 8 | 128 | - | 8q4s4k | yes | kernel | 547.6 | 773.5 | 634.7 | 1.22x | 86.3 % |
| NVIDIA-Nemotron-3-Nano-4B | 40 | 8 | 128 | - | 32q1s1k | yes | kernel | 547.6 | 592.6 | 574.3 | 1.03x | 95.4 % |
| NVIDIA-Nemotron-3-Nano-4B | 40 | 8 | 128 | - | 64q1s1k | yes | kernel | 1093.8 | 1167.9 | 1134.1 | 1.03x | 96.4 % |
| NVIDIA-Nemotron-3-Nano-4B | 40 | 8 | 128 | - | 32q4s1k | yes | kernel | 555.6 | 658.2 | 608.3 | 1.08x | 91.3 % |
| NVIDIA-Nemotron-3-Nano-4B | 40 | 8 | 128 | - | 4q1s8k_q512 | no | triton | - | 1140.7 | 1140.3 | 1.00x | - |
| NVIDIA-Nemotron-3-Nano-4B | 40 | 8 | 128 | - | 16q1s4k_2q1k | no | triton | - | 2530.6 | 2544.0 | 0.99x | - |
| NVIDIA-Nemotron-3-Nano-4B | 40 | 8 | 128 | - | 16q1s2k_q64s2k | no | triton | - | 869.4 | 868.5 | 1.00x | - |
| NVIDIA-Nemotron-3-Nano-4B | 40 | 8 | 128 | - | 32q1s1k_q64s1k | no | triton | - | 771.2 | 771.3 | 1.00x | - |

## Sliding window

Decode batches: 36 cells, vs Triton geomean 1.50x (min 1.10x, max 5.13x), median 93.1 % of roof.  Mixed: 24 cells, 20 split, 4 left whole on Triton; split cells vs Triton geomean 1.18x (min 0.99x); every cell >= 0.99x.

| models | Hq | Hkv | D | window | batch | graphs | path | roofline us | Triton us | ours us | vs | %roof |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemma-4-E2B-it | 8 | 1 | 256 | 512 | 2q1s4k | yes | kernel | 5.8 | 20.5 | 10.3 | 1.99x | 56.3 % |
| gemma-4-E2B-it | 8 | 1 | 256 | 512 | 8q1s4k | yes | kernel | 18.7 | 39.8 | 24.4 | 1.63x | 76.7 % |
| gemma-4-E2B-it | 8 | 1 | 256 | 512 | 8q4s4k | yes | kernel | 19.6 | 74.0 | 37.3 | 1.98x | 52.6 % |
| gemma-4-E2B-it | 8 | 1 | 256 | 512 | 32q1s1k | yes | kernel | 70.5 | 122.2 | 80.5 | 1.52x | 87.5 % |
| gemma-4-E2B-it | 8 | 1 | 256 | 512 | 64q1s1k | yes | kernel | 139.5 | 287.3 | 159.3 | 1.80x | 87.6 % |
| gemma-4-E2B-it | 8 | 1 | 256 | 512 | 32q4s1k | yes | kernel | 74.1 | 481.7 | 93.9 | 5.13x | 78.9 % |
| gemma-4-E2B-it | 8 | 1 | 256 | 512 | 4q1s8k_q512 | no | triton | - | 171.3 | 171.4 | 1.00x | - |
| gemma-4-E2B-it | 8 | 1 | 256 | 512 | 16q1s4k_2q1k | no | split | - | 554.7 | 552.1 | 1.00x | - |
| gemma-4-E2B-it | 8 | 1 | 256 | 512 | 16q1s2k_q64s2k | no | split | - | 179.7 | 179.7 | 1.00x | - |
| gemma-4-E2B-it | 8 | 1 | 256 | 512 | 32q1s1k_q64s1k | no | split | - | 279.5 | 225.7 | 1.24x | - |
| gemma-4-E4B-it | 8 | 2 | 256 | 512 | 2q1s4k | yes | kernel | 10.0 | 24.1 | 14.9 | 1.62x | 67.6 % |
| gemma-4-E4B-it | 8 | 2 | 256 | 512 | 8q1s4k | yes | kernel | 35.7 | 66.2 | 41.2 | 1.61x | 86.7 % |
| gemma-4-E4B-it | 8 | 2 | 256 | 512 | 8q4s4k | yes | kernel | 36.7 | 133.5 | 46.6 | 2.87x | 78.8 % |
| gemma-4-E4B-it | 8 | 2 | 256 | 512 | 32q1s1k | yes | kernel | 138.4 | 313.1 | 151.1 | 2.07x | 91.6 % |
| gemma-4-E4B-it | 8 | 2 | 256 | 512 | 64q1s1k | yes | kernel | 275.4 | 807.6 | 292.7 | 2.76x | 94.1 % |
| gemma-4-E4B-it | 8 | 2 | 256 | 512 | 32q4s1k | yes | kernel | 142.4 | 202.5 | 169.5 | 1.19x | 84.0 % |
| gemma-4-E4B-it | 8 | 2 | 256 | 512 | 4q1s8k_q512 | no | triton | - | 189.3 | 189.3 | 1.00x | - |
| gemma-4-E4B-it | 8 | 2 | 256 | 512 | 16q1s4k_2q1k | no | split | - | 679.0 | 634.4 | 1.07x | - |
| gemma-4-E4B-it | 8 | 2 | 256 | 512 | 16q1s2k_q64s2k | no | triton | - | 148.3 | 148.3 | 1.00x | - |
| gemma-4-E4B-it | 8 | 2 | 256 | 512 | 32q1s1k_q64s1k | no | triton | - | 227.8 | 227.6 | 1.00x | - |
| gemma-3-4b-it | 8 | 4 | 256 | 1024 | 2q1s4k | yes | kernel | 35.5 | 50.8 | 40.0 | 1.27x | 88.9 % |
| gemma-3-4b-it | 8 | 4 | 256 | 1024 | 8q1s4k | yes | kernel | 137.6 | 181.8 | 148.9 | 1.22x | 92.4 % |
| gemma-3-4b-it | 8 | 4 | 256 | 1024 | 8q4s4k | yes | kernel | 138.8 | 320.3 | 152.5 | 2.10x | 91.0 % |
| gemma-3-4b-it | 8 | 4 | 256 | 1024 | 32q1s1k | yes | kernel | 546.1 | 821.7 | 572.8 | 1.43x | 95.3 % |
| gemma-3-4b-it | 8 | 4 | 256 | 1024 | 64q1s1k | yes | kernel | 1090.6 | 1436.1 | 1130.6 | 1.27x | 96.5 % |
| gemma-3-4b-it | 8 | 4 | 256 | 1024 | 32q4s1k | yes | kernel | 549.2 | 760.6 | 576.9 | 1.32x | 95.2 % |
| gemma-3-4b-it | 8 | 4 | 256 | 1024 | 4q1s8k_q512 | no | split | - | 315.5 | 250.4 | 1.26x | - |
| gemma-3-4b-it | 8 | 4 | 256 | 1024 | 16q1s4k_2q1k | no | split | - | 1246.7 | 1065.1 | 1.17x | - |
| gemma-3-4b-it | 8 | 4 | 256 | 1024 | 16q1s2k_q64s2k | no | split | - | 438.6 | 443.5 | 0.99x | - |
| gemma-3-4b-it | 8 | 4 | 256 | 1024 | 32q1s1k_q64s1k | no | split | - | 828.1 | 711.3 | 1.16x | - |
| paligemma2-3b-mix-448-LLM | 8 | 4 | 256 | 4096 | 2q1s4k | yes | kernel | 137.4 | 168.1 | 147.9 | 1.14x | 92.9 % |
| paligemma2-3b-mix-448-LLM | 8 | 4 | 256 | 4096 | 8q1s4k | yes | kernel | 545.3 | 655.7 | 575.7 | 1.14x | 94.7 % |
| paligemma2-3b-mix-448-LLM | 8 | 4 | 256 | 4096 | 8q4s4k | yes | kernel | 546.1 | 809.2 | 582.3 | 1.39x | 93.8 % |
| paligemma2-3b-mix-448-LLM | 8 | 4 | 256 | 4096 | 32q1s1k | yes | kernel | 546.1 | 826.7 | 573.6 | 1.44x | 95.2 % |
| paligemma2-3b-mix-448-LLM | 8 | 4 | 256 | 4096 | 64q1s1k | yes | kernel | 1090.6 | 1451.6 | 1134.6 | 1.28x | 96.1 % |
| paligemma2-3b-mix-448-LLM | 8 | 4 | 256 | 4096 | 32q4s1k | yes | kernel | 549.2 | 769.3 | 575.6 | 1.34x | 95.4 % |
| paligemma2-3b-mix-448-LLM | 8 | 4 | 256 | 4096 | 4q1s8k_q512 | no | split | - | 947.3 | 486.3 | 1.95x | - |
| paligemma2-3b-mix-448-LLM | 8 | 4 | 256 | 4096 | 16q1s4k_2q1k | no | split | - | 2686.4 | 1909.1 | 1.41x | - |
| paligemma2-3b-mix-448-LLM | 8 | 4 | 256 | 4096 | 16q1s2k_q64s2k | no | split | - | 840.2 | 829.1 | 1.01x | - |
| paligemma2-3b-mix-448-LLM | 8 | 4 | 256 | 4096 | 32q1s1k_q64s1k | no | split | - | 827.7 | 711.5 | 1.16x | - |
| gemma-3-12b-it +2 | 16 | 8 | 256 | 1024 | 2q1s4k | yes | kernel | 69.5 | 90.6 | 79.2 | 1.14x | 87.8 % |
| gemma-3-12b-it +2 | 16 | 8 | 256 | 1024 | 8q1s4k | yes | kernel | 273.8 | 357.1 | 293.8 | 1.22x | 93.2 % |
| gemma-3-12b-it +2 | 16 | 8 | 256 | 1024 | 8q4s4k | yes | kernel | 276.1 | 396.3 | 301.0 | 1.32x | 91.7 % |
| gemma-3-12b-it +2 | 16 | 8 | 256 | 1024 | 32q1s1k | yes | kernel | 1090.6 | 1456.4 | 1128.6 | 1.29x | 96.6 % |
| gemma-3-12b-it +2 | 16 | 8 | 256 | 1024 | 64q1s1k | yes | kernel | 2179.7 | 2896.9 | 2246.7 | 1.29x | 97.0 % |
| gemma-3-12b-it +2 | 16 | 8 | 256 | 1024 | 32q4s1k | yes | kernel | 1097.0 | 1447.2 | 1152.9 | 1.26x | 95.1 % |
| gemma-3-12b-it +2 | 16 | 8 | 256 | 1024 | 4q1s8k_q512 | no | split | - | 492.0 | 438.5 | 1.12x | - |
| gemma-3-12b-it +2 | 16 | 8 | 256 | 1024 | 16q1s4k_2q1k | no | split | - | 2390.6 | 1991.9 | 1.20x | - |
| gemma-3-12b-it +2 | 16 | 8 | 256 | 1024 | 16q1s2k_q64s2k | no | split | - | 859.0 | 780.4 | 1.10x | - |
| gemma-3-12b-it +2 | 16 | 8 | 256 | 1024 | 32q1s1k_q64s1k | no | split | - | 1654.3 | 1321.9 | 1.25x | - |
| gemma-4-31B-it-AWQ +1 | 32 | 16 | 256 | 1024 | 2q1s4k | yes | kernel | 137.6 | 166.6 | 148.6 | 1.12x | 92.6 % |
| gemma-4-31B-it-AWQ +1 | 32 | 16 | 256 | 1024 | 8q1s4k | yes | kernel | 546.0 | 631.4 | 573.1 | 1.10x | 95.3 % |
| gemma-4-31B-it-AWQ +1 | 32 | 16 | 256 | 1024 | 8q4s4k | yes | kernel | 550.8 | 801.7 | 582.8 | 1.38x | 94.5 % |
| gemma-4-31B-it-AWQ +1 | 32 | 16 | 256 | 1024 | 32q1s1k | yes | kernel | 2179.7 | 2933.9 | 2247.2 | 1.31x | 97.0 % |
| gemma-4-31B-it-AWQ +1 | 32 | 16 | 256 | 1024 | 64q1s1k | yes | kernel | 4357.9 | 5693.6 | 4495.5 | 1.27x | 96.9 % |
| gemma-4-31B-it-AWQ +1 | 32 | 16 | 256 | 1024 | 32q4s1k | yes | kernel | 2192.4 | 2915.7 | 2298.8 | 1.27x | 95.4 % |
| gemma-4-31B-it-AWQ +1 | 32 | 16 | 256 | 1024 | 4q1s8k_q512 | no | split | - | 966.7 | 788.9 | 1.23x | - |
| gemma-4-31B-it-AWQ +1 | 32 | 16 | 256 | 1024 | 16q1s4k_2q1k | no | split | - | 4664.6 | 3855.8 | 1.21x | - |
| gemma-4-31B-it-AWQ +1 | 32 | 16 | 256 | 1024 | 16q1s2k_q64s2k | no | split | - | 1768.1 | 1534.0 | 1.15x | - |
| gemma-4-31B-it-AWQ +1 | 32 | 16 | 256 | 1024 | 32q1s1k_q64s1k | no | split | - | 3186.7 | 2621.7 | 1.22x | - |
