# f6-combined — ¿se acumulan HND (layout) y el kernel HIP v5?

## Respuesta en una línea

**SÍ se acumulan, y de forma limpiamente multiplicativa.** La ganancia del
kernel HIP es **idéntica en los dos layouts** (1.234× en NHD, 1.236× en HND) y
la ganancia de HND es **idéntica en los dos kernels** (1.071× en Triton, 1.072×
en HIP). Combinado: **198.7 → 150.2 µs = 1.323×**, **90.5 % del roofline**.

La hipótesis del brief — "ambas atacan el mismo mecanismo, el kernel HIP ya
evitaría el conflicto y HND no añadiría nada" — queda **refutada por la medida**.
Atacan dos capas distintas de la misma jerarquía (§ Mecanismo).

**Corrección al brief:** el 1.09× de HND no se sostiene en el régimen del
baseline de 198.87 µs; el número correcto es **1.07×** (ver § Confound).

---

## La matriz 2×2

Caso `Hq=32, Hkv=16, M=4, D=256, S=2048, BS=16`. Roofline **135.87 µs**
(32.0 MiB de KV / 230 GiB/s). Baseline **198.87 µs** (f0).

Cada celda: **µs · % del roofline · × vs baseline 198.87**

| | **NHD** (stride_head 1 KiB) | **HND** (stride_head 16 KiB) | ganancia HND |
|---|---|---|---|
| **Triton** | **198.65 µs**<br>68.4 % roof<br>1.001× | **185.73 µs**<br>73.2 % roof<br>**1.070×** | **1.070×** |
| **HIP paged v5** | **161.4 µs**<br>84.2 % roof<br>**1.232×** | **150.2 µs**<br>**90.5 % roof**<br>**1.324×** | **1.075×** |
| **ganancia kernel** | **1.231×** | **1.236×** | |

Producto de las dos ganancias aisladas: 1.070 × 1.232 = **1.318×**.
Combinación medida: **1.324×**. **Coinciden dentro del ruido (0.5 %).**

Detalle de las réplicas:

| celda | réplicas (µs) | mediana |
|---|---|---|
| Triton NHD | 198.46 / 198.73 / 198.76 | **198.73** |
| Triton HND | 186.51 / 185.42 / 185.27 | **185.42** |
| HIP NHD | 161.42 / 160.73 / 161.80 | **161.42** |
| HIP HND | 150.34 / 150.13 / 150.80 | **150.34** |

Spread por celda < 0.7 %. La separación NHD↔HND (7 %) y Triton↔HIP (23 %)
están ambas un orden de magnitud por encima del ruido.

### Robustez: asignador realista

Con block table permutada aleatoriamente y pool 4× (`shuf=1 poolx=4`), que es
lo que produce el asignador de vLLM tras fragmentarse:

| | NHD | HND | ganancia |
|---|---|---|---|
| HIP paged, shuf=1 poolx=4 | 161.73 µs | 151.22 µs | **1.069×** |

La ganancia sobrevive intacta. No es un artefacto de tener las páginas en orden.

---

## Confound corregido: el 1.09× era 1.07×

El brief hereda "HND = 1.09× sobre Triton" de f5. Ese 1.09× se midió con el
benchmark **sin** `--min-working-set-mb 96` (282 → 259 µs). El baseline de
198.87 µs de f0 **sí** lleva ese flag. Son dos regímenes de working-set
distintos y no se pueden mezclar en la misma matriz.

Medido en el régimen del baseline (3 A/B alternados):

```
Triton q4s2k, --min-working-set-mb 96, cuda graphs
  NHD  198.46 / 198.73 / 198.76 µs
  HND  186.51 / 185.42 / 185.27 µs      -> 1.070×
```

Y medido sin el flag, reproduzco el 1.09× de f5. Ambos números son correctos;
son configuraciones distintas. **El que va al PR junto al 198.87 µs es 1.07×.**

Esto rebaja ligeramente la proyección: no 1.09 × 1.19 = 1.30×, sino
**1.07 × 1.23 = 1.32×** — porque la ganancia del kernel también sube al pasar
del denso (166.96 µs de f3b) al paginado (161.4 µs).

---

## Mecanismo: por qué sí se acumulan

Las dos palancas **no** atacan el mismo nivel, aunque ambas toquen "el patrón de
acceso":

**HND arregla la dispersión entre kv-heads.** En NHD las 16 kv-heads de un token
están entrelazadas cada 1 KiB; las 16 heads vivas simultáneamente caen en los
mismos sets de L1/GL1. En HND cada head recibe un bloque contiguo de 16 KiB por
página y barre su propia región secuencial. Esto es una propiedad **del layout
en memoria**, no del kernel: ningún kernel que lea las 16 heads puede evitarla,
porque el conflicto lo causan *bloques distintos* leyendo *heads distintas* del
*mismo* tensor.

**El kernel HIP arregla el interior del bucle.** `NSEG=1`, `KPW=4`, `b128`,
0 barriers, 0 LDS, epílogo wave→global. Esto ataca ocupación, profundidad de
pipeline e instrucciones por byte. Es ortogonal al layout.

Un kernel no puede "evitar por su patrón de acceso" un conflicto que nace de que
16 workgroups distintos leen 16 regiones separadas por 1 KiB. Por eso el
1.07× de HND reaparece intacto encima del kernel HIP.

### Predicción falsable, verificada en el kernel HIP

Si el mecanismo de HND es competencia entre kv-heads, su ganancia debe escalar
con `H_kv` y desaparecer cuando hay pocas heads vivas. f5 lo verificó en Triton;
lo repito **dentro del kernel HIP**, que es donde el brief dudaba:

| H_kv | HIP NHD | HIP HND | ganancia |
|---|---|---|---|
| 4 | 84.57 µs | 85.47 µs | **0.99× (desaparece / leve regresión)** |
| 8 | 100.71 µs | 98.54 µs | 1.022× |
| 16 | 161.80 µs | 150.80 µs | **1.073×** |

Monótona en `H_kv` y anulada en `H_kv=4`, exactamente como en Triton. **El mismo
mecanismo opera en los dos kernels** — que es precisamente la razón por la que
se acumulan en vez de solaparse: el kernel HIP nunca lo tocó.

Corolario práctico: **HND sólo paga para `H_kv` alto.** Para Llama-style
(`H_kv=8`, D=128) la ganancia es marginal; para el caso de estudio
(`H_kv=16`, D=256) es el 7 %.

---

## Correctitud

Criterio del brief (no el max_abs defectuoso): `max_rel ≤ 1e-3`, un caso S≈48,
y controles negativos.

| config | S | max_rel | max_abs | veredicto |
|---|---|---|---|---|
| paged LAYOUT=0 (NHD) | 2048 | 1.895e-05 | 2.42e-08 | **PASS** |
| paged LAYOUT=1 (HND) | 2048 | 1.895e-05 | 2.42e-08 | **PASS** |
| paged LAYOUT=0 (NHD) | 48 | 3.431e-05 | 8.94e-08 | **PASS** |
| paged LAYOUT=1 (HND) | 48 | 3.431e-05 | 8.94e-08 | **PASS** |
| H_kv=4 / 8 NHD y HND | 2048 | 1.63e-05 / 1.68e-05 | 2.42e-08 | **PASS** |
| shuf=1 poolx=4, ambos layouts | 2048 | 1.895e-05 | 2.42e-08 | **PASS** |

**Controles negativos** (`decode_attn_paged_mutants.hip`, S=48, LAYOUT=0):

| mutante | max_rel | max_abs | ¿lo caza `max_rel≤1e-3`? | ¿lo cazaría `max_abs≤2e-2`? |
|---|---|---|---|---|
| MUTATE=1 máscara causal off-by-one | 3.732e+01 | 5.44e-02 | **sí** | sí (por poco) |
| MUTATE=2 slot↔head intercambiado | 2.070e+02 | 4.13e-01 | **sí** | sí |
| MUTATE=3 offset de V olvidado | 2.243e+02 | 4.13e-01 | **sí** | sí |

Los tres fallan; los correctos pasan con 4 órdenes de margen. El criterio
discrimina. (Nota: `MUTATE=2` bajo `LAYOUT=1` es degenerado — la mutación *es*
la fórmula HND — así que hay que correrlo bajo `LAYOUT=0`; corrido ahí, falla.)

**NHD y HND dan salida idéntica bit a bit** (mismo `max_rel` a 4 dígitos en
todas las filas): el layout no cambia la aritmética, sólo el direccionamiento.

---

## Lo que va al PR

- **HND solo**, cero código, sólo `VLLM_KV_CACHE_LAYOUT=HND`:
  **198.7 → 185.7 µs, 1.070×**, 73.2 % del roofline, sin coste de memoria
  (`page_size_bytes` idéntico).
- **Kernel HIP solo**, sobre el layout actual NHD:
  **198.7 → 161.4 µs, 1.232×**, 84.2 % del roofline.
- **Los dos juntos: 198.7 → 150.2 µs, 1.324×, 90.5 % del roofline.**
  Es un número que **sí se sostiene** — está medido directamente, no derivado
  de un producto.

Aviso al redactar: **no escribir 1.09× para HND** si el baseline citado es
198.87 µs. El 1.09× pertenece al régimen sin `--min-working-set-mb`.

Caveat de alcance: todas las cifras son `H_kv=16, D=256, M=4, S=2048, bs=1`.
La parte HND se anula a `H_kv≤4` (medido). La parte del kernel HIP no está
integrada en vLLM — es un kernel standalone con el direccionamiento paginado
real, no un backend.

---

## Reproducir

```bash
# Fila Triton (30 s, sin compilar)
cd /scratch/rogarcia/f0-baseline/benchmarks/attention_benchmarks
AMDSMI=/scratch/rogarcia/vllm-build/.venv/lib/python3.12/site-packages/_rocm_sdk_core/share/amd_smi
for L in NHD HND; do
  PATH="/scratch/rogarcia/vllm-build/.venv/bin:$PATH" \
  PYTHONPATH="/scratch/rogarcia/f0-baseline:$AMDSMI" \
  VLLM_KV_CACHE_LAYOUT=$L quiet-lock measure amd-gpu-lock \
    /scratch/rogarcia/vllm-build/.venv/bin/python benchmark.py \
    --backends TRITON_ATTN --batch-specs q4s2k \
    --num-q-heads 32 --num-kv-heads 16 --head-dim 256 \
    --min-working-set-mb 96 --inter-batch-cooldown 0
done

# Fila HIP
cd /scratch/rogarcia/f3-hip/fleet/hip
TAG=f6 bash psweep.sh "-DNSEG=1 -DKPW=4 -DLAYOUT=0" "-DNSEG=1 -DKPW=4 -DLAYOUT=1"

# Correctitud + controles negativos
S=48 ITERS=0 TAG=f6nc SRC=$PWD/decode_attn_paged_mutants.hip bash psweep.sh \
  "-DNSEG=1 -DKPW=4 -DLAYOUT=0" "-DNSEG=1 -DKPW=4 -DLAYOUT=1" \
  "-DNSEG=1 -DKPW=4 -DLAYOUT=0 -DMUTATE=1" \
  "-DNSEG=1 -DKPW=4 -DLAYOUT=0 -DMUTATE=2" \
  "-DNSEG=1 -DKPW=4 -DLAYOUT=0 -DMUTATE=3"

# Barrido H_kv
for HK in 4 8 16; do
  TAG=f6h$HK HQ=32 HKV=$HK D=256 bash psweep.sh \
    "-DNSEG=1 -DKPW=4 -DLAYOUT=0 -DNUM_KV_HEADS=$HK" \
    "-DNSEG=1 -DKPW=4 -DLAYOUT=1 -DNUM_KV_HEADS=$HK"
done
```

### Cambios hechos en `fleet/` (nada fuera)

- `fleet/hip/check.py`: `HQ`/`HKV`/`D` ahora se leen de entorno (antes fijos a
  32/16/256), para poder validar el barrido de `H_kv`. Defaults sin cambio.
- `fleet/hip/psweep.sh`: el directorio de datos incluye la forma, para que el
  barrido de `H_kv` no reutilice la referencia de otra forma.
- Copiado `decode_attn_paged_mutants.hip` al worktree `f3-hip` (sólo estaba en
  el canónico).
