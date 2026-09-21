# F5-layout — ¿merece la pena cambiar el layout del KV cache?

## Recomendación

> ## SÍ, pero **no con padding**.
>
> **NO padear** (`KV_PAD` en `get_kv_cache_shape`): la ganancia de 1.07× medida
> en el harness denso **no se transfiere** al layout paginado, el padding mínimo
> defendible (`KV_PAD=8`) es **más lento** que no padear, y cuesta memoria.
>
> **SÍ cambiar el orden de dimensiones a HND**: `VLLM_KV_CACHE_LAYOUT=HND`.
> **1.09× en el kernel Triton de producción** (282 → 259 µs), **coste de memoria
> exactamente cero** (misma `page_size_bytes`, 256 KiB), **cero bloques
> perdidos**, y **cero líneas de kernel** — `get_kv_cache_stride_order` ya lo
> soporta (`vllm/v1/attention/backends/triton_attn.py:388`).
>
> El cambio a integrar es **una línea** en
> `vllm/distributed/kv_transfer/kv_connector/utils.py:51` (el default `"NHD"`),
> o el default de `VLLM_KV_CACHE_LAYOUT` en gfx1151.

El trade-off que planteaba el brief (latencia vs bloques perdidos) **no llega a
existir**: la vía gratuita gana más que la vía cara.

---

## 1. ¿Se reproduce la ganancia de KV_PAD en el layout paginado real? → **NO**

Construí `fleet/hip/decode_attn_paged.hip`: el kernel v5 re-alojado sobre el
layout real de vLLM — tensor único paginado, K y V empaquetados en la dim de
contenido, block table indirecto. Matemática, máscara y epílogo byte-idénticos
a v5; sólo cambia el direccionamiento.

Baseline paginado NHD `KV_PAD=0` = **161.8 µs** (el denso de f3b daba 166.96 µs;
el layout real es ya algo mejor porque K y V del mismo token son contiguos).

Barrido de `KV_PAD` (S=2048, M=4, 200 it):

| KV_PAD | stride_head | página | µs | vs pad 0 |
|---|---|---|---|---|
| 0 | 1024 B | 256 KiB | **161.8** | 1.00× |
| 8 | 1040 B | 260 KiB | 165.6 | **0.98× (peor)** |
| 16 | 1056 B | 264 KiB | 163.4 | 0.99× (peor) |
| 32 | 1088 B | 272 KiB | 166.8 | 0.97× (peor) |
| 64 | 1152 B | 288 KiB | 155.4 | 1.04× |
| 128 | 1280 B | 320 KiB | 175.5 | 0.92× (peor) |

Dos conclusiones:

1. **La ganancia se degrada de 1.07× a 1.04×** al pasar al layout real. El brief
   tenía razón en no asumirla transferible.
2. **No es monótona.** Sólo `KV_PAD=64` ayuda; 8, 16, 32 y 128 son *más lentos*
   que 0. No es una tendencia que se pueda truncar barato: es una resonancia
   puntual del stride.

### Punto 2 del brief — "¿cuál es el padding mínimo que captura la ganancia?"

**No existe.** `KV_PAD=8` (+1.6 % de memoria) da **−2.3 % de rendimiento**.
El único padding que gana es 64, que es justamente el que el brief ya
identificaba como indefendible. La hipótesis "el coste es lineal, luego habrá
un punto intermedio barato" es **falsa** aquí.

---

## 2. La alternativa sin coste de memoria → **HND. Esto es el hallazgo.**

| config | stride_head | stride_tok | página | µs | vs NHD |
|---|---|---|---|---|---|
| NHD BS=16 KV_PAD=0 (hoy) | 1024 B | 16 KiB | 256 KiB | 161.7 | 1.00× |
| NHD BS=16 KV_PAD=64 | 1152 B | 18 KiB | 288 KiB **(+12.5 %)** | 155.4 | 1.04× |
| **HND BS=16 KV_PAD=0** | 16 KiB | **1024 B** | **256 KiB (+0 %)** | **150.2** | **1.077×** |
| HND BS=32 | 32 KiB | 1024 B | 512 KiB | 150.1 | 1.077× |
| HND BS=64 | 64 KiB | 1024 B | 1 MiB | 149.9 | 1.079× |
| HND BS=128 | 128 KiB | 1024 B | 2 MiB | 155.8 | 1.038× |

**HND gana más que `KV_PAD=64` y no cuesta un byte.** Verificado en código:

```
NHD: logical=(1024,16,16,512) stride_order=(0,2,1,3) page_size_bytes=262144
HND: logical=(1024,16,16,512) stride_order=(0,1,2,3) page_size_bytes=262144
```

`get_kv_cache_shape` devuelve la **misma forma lógica**; HND es sólo una
permutación. `page_size_bytes` es idéntico ⇒ **mismo número de bloques, misma
concurrencia, mismo KV-cache útil.**

**HND y KV_PAD no suman** — HND+`KV_PAD=64` (159.2 µs) es *peor* que HND solo
(151.8 µs). El padding sólo compensaba, a medias, la patología que HND elimina
de raíz. Una vez en HND, padear es puro coste.

### Mecanismo

Encaja exactamente con el diagnóstico de conflicto de sets que ya tenía la
flota. En NHD las 16 kv-heads de un token están entrelazadas cada 1 KiB, así que
las 16 heads vivas compiten por los mismos sets de L1/GL1. En HND los tokens
consecutivos de una misma kv-head son contiguos (`stride_tok = 1024 B`) y las
heads quedan separadas por 16 KiB: cada head barre su propia región secuencial.

**Predicción falsable, y se cumple**: si el mecanismo es competencia entre
kv-heads, la ganancia debe escalar con `num_kv_heads` y desaparecer con HKV=1.

| HQ / HKV / D | NHD (q1s4k) | HND | ganancia |
|---|---|---|---|
| 32 / 16 / 256 | 367 µs | 345 µs | 1.06× |
| 32 / 8 / 128 | 99 µs | 88 µs | 1.13× |
| 64 / 8 / 128 | 100 µs | 89 µs | 1.12× |
| 32 / 4 / 128 | 44 µs | 44 µs | 1.00× |
| 32 / 2 / 128 | 30 µs | 30 µs | 1.00× |
| **8 / 1 / 64** | 10 µs | 10 µs | **1.00× (desaparece)** |

Con una sola kv-head el efecto se anula, como debe ser. El mecanismo está
confirmado de forma independiente.

---

## 3. Confirmación en el kernel Triton de PRODUCCIÓN (no sólo mi HIP)

El benchmark oficial `benchmarks/attention_benchmarks/benchmark.py` honra
`get_kv_cache_stride_order`, así que basta la variable de entorno. **Cero
cambios de código.**

```
q4s2k, HQ=32 HKV=16 D=256, TRITON_ATTN, cuda graphs
       rep1     rep2     rep3
NHD    282 µs   282 µs   282 µs
HND    258 µs   259 µs   259 µs        -> 1.090×, varianza ~1 µs
```

El mismo kernel Triton que la flota midió en 198.87 µs mejora un **8.3 %** sólo
cambiando el `stride_order`.

### No regresa en ningún régimen

| batch spec | tipo | NHD | HND | ganancia |
|---|---|---|---|---|
| q1s1k | decode | 104 µs | 99 µs | 1.05× |
| q1s4k | decode | 367 µs | 345 µs | 1.06× |
| q1s16k | decode | 1432 µs | 1333 µs | 1.07× |
| q4s2k | spec-decode | 284 µs | 260 µs | 1.09× |
| q8s2k | spec-decode | 283 µs | 259 µs | 1.09× |
| **q2k** | **prefill** | 5578 µs | 5118 µs | **1.09×** |
| **16q1s2k** | **decode bs=16** | 3393 µs | 2936 µs | **1.16×** |
| **64q1s2k** | **decode bs=64** | 12317 µs | 11027 µs | **1.12×** |

**Gana en los 8 regímenes, y gana MÁS en batch alto** (1.12–1.16× a bs=16/64)
que en batch 1. Justo el régimen que le importa a un servidor: más secuencias
vivas ⇒ más presión de caché ⇒ más conflicto que eliminar. Prefill también mejora.

### Camino de escritura sin penalización

HND hace el `reshape_and_cache` strided, así que había que comprobarlo:

```
NHD: reshape_and_cache 256 tokens = 19.34 us   kv.stride=(131072, 512, 8192, 1)
HND: reshape_and_cache 256 tokens = 19.35 us   kv.stride=(131072, 8192, 512, 1)
```

**Indistinguible (0.05 %).** La ganancia de lectura no se paga en la escritura.

### `block_size` (kernel Triton real)

| | q1s4k | q4s2k | 16q1s2k |
|---|---|---|---|
| NHD bs=16 | 367 | 281 | 3395 |
| NHD bs=32 | 516 | 368 | 3597 |
| NHD bs=64 | 852 | 271 | 3392 |
| **HND bs=16** | **345** | **259** | 2949 |
| HND bs=32 | 468 | 325 | 3012 |
| HND bs=64 | 909 | 240 | 2845 |

`block_size` **no es una palanca fiable**: no hay un valor que gane en las tres
columnas, y su efecto es errático (bs=32 es peor que 16 y que 64 en q1s4k). HND
con el `block_size=16` actual es la elección robusta. **No tocar `block_size`.**

---

## 4. Trade-off end-to-end

El brief pedía la curva "a partir de qué contexto/concurrencia deja de
compensar". **Para HND la pregunta no aplica**: el coste es cero en las dos
dimensiones que importan.

| | KV_PAD=64 | **HND** |
|---|---|---|
| latencia decode | 1.04× | **1.09×** |
| `page_size_bytes` | 288 KiB (+12.5 %) | **256 KiB (+0 %)** |
| bloques con la misma VRAM | **−11 %** | **0 %** |
| tokens en caché | −11 % | 0 % |
| coste de escritura | — | **0 %** |
| líneas de kernel a tocar | reindexado + ABI | **0** |
| ganancia a bs=64 | no medida | **1.12×** |

Para `KV_PAD=64` sí se puede cerrar el argumento: **nunca compensa.** Cuesta
un 11 % de bloques para ganar un 4 % de latencia, cuando existe una opción que
gana un 9 % gratis. Y como HND+pad es *peor* que HND solo, tampoco hay una
versión combinada que rescate el padding.

El único eje donde HND no gana es `num_kv_heads ≤ 4`, donde simplemente empata
(1.00×). **No hay ningún punto medido donde HND pierda.**

---

## 5. Correctitud

Protocolo del brief: `max_rel ≤ 1e-3` + caso S≈48 + control negativo.

**16/16 PASS** — M ∈ {1,2,4,5} × S ∈ {2048, 48} × layout ∈ {NHD, HND}.
`max_rel` entre 1.65e-05 y 3.95e-05, ~30× por debajo del criterio.

**Controles negativos** (`fleet/hip/decode_attn_paged_mutants.hip`), a S=48,
donde el error no se diluye:

| mutación | max_rel | detectado |
|---|---|---|
| sin mutar (NHD) | 3.43e-05 | PASS ✓ |
| `MUTATE=1` off-by-one en máscara causal | 3.73e+01 | **FAIL ✓** |
| `MUTATE=2` índice de layout intercambiado (slot↔head) | 2.07e+02 | **FAIL ✓** |
| `MUTATE=3` olvidar el offset del empaquetado K/V | 2.24e+02 | **FAIL ✓** |
| `MUTATE=1` bajo HND | 3.73e+01 | **FAIL ✓** |

Añadí dos mutantes *específicos del layout* (`MUTATE=2,3`) porque el riesgo real
de este trabajo no es la máscara sino transcribir mal el direccionamiento. El
test los pilla por 5 órdenes de magnitud.

**Suite de vLLM**: `tests/v1/attention/test_attention_backends.py` da
**97 failed / 18 passed / 9 skipped bajo NHD y exactamente lo mismo bajo HND** —
cero regresiones. Los fallos son preexistentes y ambientales: HTTP 401
`GatedRepoError` de `meta-llama/Meta-Llama-3-8B` (sin `HF_TOKEN`), no numéricos.

---

## 6. Qué tocar en vLLM

**No hay que tocar `get_kv_cache_shape` ni `get_kv_cache_stride_order` ni
`block_size`.** Todo está ya implementado y probado. Sólo cambia el *default*.

Opciones, de menos a más invasiva:

1. **Cero código — operacional.** Arrancar con `VLLM_KV_CACHE_LAYOUT=HND`.
   Sirve para validar en producción antes de tocar nada.

2. **Default por plataforma (recomendado).** El default vive en
   `vllm/distributed/kv_transfer/kv_connector/utils.py:51` (`return "NHD"`),
   vía `get_kv_cache_layout()` en
   `vllm/v1/attention/backends/utils.py:163`. Devolver `"HND"` cuando
   `current_platform.is_rocm()` y `on_gfx1151()` — el mismo predicado que ya usa
   `triton_attn.py:53` (`_ON_GFX1151`). El override de usuario y el de conector
   ya tienen prioridad sobre el default, así que no rompe nada.

Consideraciones de integración, ya verificadas:

- `get_kv_cache_stride_order` soporta HND desde hace tiempo (`triton_attn.py:401-406`),
  incluido el caso `include_num_layers_dimension`.
- `page_size_bytes` no cambia ⇒ el KV cache manager, el perfilado de memoria y
  el número de bloques no se ven afectados. **Ningún cambio en `kv_cache_utils.py`.**
- NIXL ya **exige** HND (`kv_connector/v1/nixl/connector.py:107`) y FlashInfer
  tiene rutas con `assert get_kv_cache_layout() == "HND"`. HND no es un camino
  exótico; es el que ya usa el PD-disagregado.
- `reshape_and_cache` no se degrada (medido).
- El caché de `get_kv_cache_layout` es `lru_cache`; si se cambia el default hay
  que respetar `set_kv_cache_layout` + `cache_clear` en tests, como ya hace
  `runner.py:506-509`.

**Antes de abrir PR** (requisito de `AGENTS.md`): comprobar duplicados con
`gh pr list --repo vllm-project/vllm --search "kv cache layout HND rocm"`, y
aportar evals end-to-end con `vllm bench` — mis medidas son de kernel y de
benchmark de atención, no de servidor completo.

---

## 7. Limitaciones honestas

1. **No medí un servidor end-to-end.** La atención es una fracción del paso de
   decode; un 9 % de kernel no es un 9 % de tokens/s. Lo que sí está cerrado es
   que **no hay coste que compensar**, así que la decisión no depende de esa
   cifra.
2. **fp16/auto**. No probé `kv_cache_dtype=fp8` ni los modos
   `per_token_head_scales`, que cambian la forma de la fila (`padded_hs`) y por
   tanto los strides. Merecen una comprobación aparte antes de fijar el default.
3. **Una GPU** (gfx1151, 20 CU, L2 2 MiB). El mecanismo es de jerarquía de caché;
   otras arquitecturas tendrán otra geometría de sets. El gating por `on_gfx1151()`
   es deliberadamente conservador.
4. **HKV ≥ 8** es donde está la ganancia. Con HKV ≤ 4 empata. Un default global
   para todo ROCm necesitaría medir más formas; el gating propuesto no lo asume.

---

## Artefactos

| ruta | qué es |
|---|---|
| `/scratch/rogarcia/vllm/fleet/hip/decode_attn_paged.hip` | v5 sobre el layout paginado real; knobs `LAYOUT`/`BS`/`KV_PAD`/`PAGE_PAD`/shuf/poolx |
| `/scratch/rogarcia/vllm/fleet/hip/decode_attn_paged_mutants.hip` | idem + `MUTATE=2,3` (controles negativos de layout) |
| `/scratch/rogarcia/vllm/fleet/hip/psweep.sh` | barrido paginado, un lock para N builds y N medidas |

### Reproducir el resultado principal (30 s, sin compilar nada)

```bash
cd /scratch/rogarcia/f3-hip/benchmarks/attention_benchmarks
AMDSMI=/scratch/rogarcia/vllm-build/.venv/lib/python3.12/site-packages/_rocm_sdk_core/share/amd_smi
for L in NHD HND; do
  PATH="/scratch/rogarcia/vllm-build/.venv/bin:$PATH" \
  PYTHONPATH="/scratch/rogarcia/f3-hip:$AMDSMI" \
  VLLM_KV_CACHE_LAYOUT=$L quiet-lock measure amd-gpu-lock \
    /scratch/rogarcia/vllm-build/.venv/bin/python benchmark.py \
    --backends TRITON_ATTN --batch-specs q4s2k 16q1s2k \
    --num-q-heads 32 --num-kv-heads 16 --head-dim 256 --inter-batch-cooldown 0
done
```

### Reproducir el barrido de kernel

```bash
cd /scratch/rogarcia/f3-hip/fleet/hip
bash psweep.sh "-DNSEG=1 -DKPW=4 -DLAYOUT=0 -DKV_PAD=0" \
               "-DNSEG=1 -DKPW=4 -DLAYOUT=0 -DKV_PAD=64" \
               "-DNSEG=1 -DKPW=4 -DLAYOUT=1 -DKV_PAD=0"
```

No se tocó nada fuera de `fleet/`, no se commiteó nada, no se usó `git clean`.
