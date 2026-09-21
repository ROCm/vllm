# F9 — barrido de contexto S ∈ {128, 1k, 2k, 4k, 8k, 16k, 32k}

`Hq=32, Hkv=16, D=256, M=4`, fp16/bf16, block_size=16.
Roofline del protocolo: 230 GiB/s → `t_roof = 2·S·Hkv·D·2 / 230GiB/s`.

**Estado: COMPLETO.** 28/28 celdas + barrido NSEG en ambos extremos + controles.

## Resumen en cinco líneas

1. El % del roofline **no satura en 72 %**: Triton NHD sube monótono 35.7 % (128) → 76.6 % (32k).
2. **HND no se diluye en DRAM**: ganancia plana 1.065-1.073× en Triton de 1k a 32k.
3. **S=128 es el único cruce**: HIP+HND (14.44 µs) pierde contra HIP+NHD (14.10 µs), 0.976×.
4. **El kernel HIP nunca pierde** contra Triton; su ventaja es máxima a S=128 (1.688×), mínima a 32k (1.172×).
5. **NSEG=1 gana en todo el rango**, y el castigo por subirlo crece con S (a 32k, NSEG=16 cuesta 1.50×).

## Rooflines de referencia

| S | KV/capa | t_roof (µs) | régimen |
|---|---|---|---|
| 128 | 2.0 MiB | 8.49 | cabe en L2 |
| 1024 | 16.0 MiB | 67.93 | cabe en MALL |
| 2048 | 32.0 MiB | 135.87 | límite MALL |
| 4096 | 64.0 MiB | 271.74 | DRAM |
| 8192 | 128 MiB | 543.48 | DRAM |
| 16384 | 256 MiB | 1086.96 | DRAM |
| 32768 | 512 MiB | 2173.91 | DRAM |

## Tabla principal

### Fila 1 — Triton NHD (COMPLETA)

`benchmark.py --backends TRITON_ATTN --min-working-set-mb 96`, fp16, mediana.

| S | capas | µs | % roofline |
|---|---|---|---|
| 128 | 48 | 23.80 | 35.7 % |
| 1024 | 10 | 111.02 | 61.2 % |
| 2048 | 10 | 199.15 | 68.2 % |
| 4096 | 10 | 376.52 | 72.2 % |
| 8192 | 10 | 737.07 | 73.7 % |
| 16384 | 10 | 1441.03 | 75.4 % |
| 32768 | 10 | 2838.81 | 76.6 % |

El % de roofline **no satura en 72 %**: sigue subiendo lentamente hasta 76.6 %
a 32k. Y en S=128 se desploma a 35.7 % (mucho peor que el 55.3 % de S=512): el
overhead fijo del kernel domina por completo.

### Fila 2 — Triton HND (COMPLETA)

Mismo comando con `VLLM_KV_CACHE_LAYOUT=HND`.

| S | capas | µs | % roofline | ganancia vs NHD |
|---|---|---|---|---|
| 128 | 48 | 22.98 | 37.0 % | 1.036× |
| 1024 | 10 | 103.89 | 65.4 % | 1.069× |
| 2048 | 10 | 185.71 | 73.2 % | 1.072× |
| 4096 | 10 | 350.97 | 77.4 % | 1.073× |
| 8192 | 10 | 689.69 | 78.8 % | 1.069× |
| 16384 | 10 | 1353.00 | 80.3 % | 1.065× |
| 32768 | 10 | 2644.74 | 82.2 % | 1.073× |

**HND no se diluye en DRAM.** La ganancia es planísima, 1.065–1.073×, en todo
el rango de 1k a 32k, incluso cuando el KV es 16× el MALL. Solo se cae en
S=128 (1.036×), donde el overhead fijo, no el ancho de banda, es el cuello.

### TABLA COMPLETA 7 × 4 (28/28 celdas)

`µs (% del roofline)`. Triton: `benchmark.py`, fp16, `--min-working-set-mb 96`.
HIP: `decode_attn_paged_f9.hip -DNSEG=1 -DKPW=4`, ncopy ≥ 10 (48 a S=128).

| S | Triton NHD | Triton HND | HIP NHD | HIP HND |
|---|---|---|---|---|
| 128 | 23.80 (35.7 %) | 22.98 (36.9 %) | **14.10 (60.2 %)** | 14.44 (58.8 %) |
| 1024 | 111.02 (61.2 %) | 103.89 (65.4 %) | 85.21 (79.7 %) | **80.26 (84.6 %)** |
| 2048 | 199.15 (68.2 %) | 185.71 (73.2 %) | 162.61 (83.6 %) | **153.16 (88.7 %)** |
| 4096 | 376.52 (72.2 %) | 350.97 (77.4 %) | 316.60 (85.8 %) | **295.75 (91.9 %)** |
| 8192 | 737.07 (73.7 %) | 689.69 (78.8 %) | 618.70 (87.8 %) | **585.46 (92.8 %)** |
| 16384 | 1441.03 (75.4 %) | 1353.00 (80.3 %) | 1220.56 (89.1 %) | **1157.49 (93.9 %)** |
| 32768 | 2838.81 (76.6 %) | 2644.74 (82.2 %) | 2421.45 (89.8 %) | **2308.08 (94.2 %)** |

### Ganancias

| S | HND en Triton | HND en HIP | HIP vs Triton (ambos NHD) | combo (HIP+HND vs Triton NHD) |
|---|---|---|---|---|
| 128 | 1.036× | **0.976×** | **1.688×** | 1.648× |
| 1024 | 1.069× | 1.062× | 1.303× | 1.383× |
| 2048 | 1.072× | 1.062× | 1.225× | 1.300× |
| 4096 | 1.073× | 1.070× | 1.189× | 1.273× |
| 8192 | 1.069× | 1.057× | 1.191× | 1.259× |
| 16384 | 1.065× | 1.054× | 1.181× | 1.245× |
| 32768 | 1.073× | 1.049× | 1.172× | 1.230× |

### Detalle HIP (µs, ncopy)

`decode_attn_paged_f9.hip` (= `decode_attn_paged.hip` + `dkv[64]` para poder
rotar más copias), `-DNSEG=1 -DKPW=4`, M=4, BS=16, shuf=0, poolx=1.
`ncopy` = número de copias del KV rotadas = equivalente a las "capas" de Triton.

| S | ncopy | HIP NHD µs | % roof | HIP HND µs | % roof | HND/NHD |
|---|---|---|---|---|---|---|
| 128 | 48 | 14.10 | 60.2 % | 14.44 | 58.8 % | **0.976×** |
| 1024 | 10 | 85.21 | 79.7 % | 80.26 | 84.6 % | 1.062× |
| 2048 | 10 | 162.61 | 83.6 % | 153.16 | 88.7 % | 1.062× |
| 4096 | 10 | 316.60 | 85.8 % | 295.75 | 91.9 % | 1.070× |
| 8192 | 10 | 618.70 | 87.8 % | 585.46 | 92.8 % | 1.057× |
| 16384 | 10 | 1220.56 | 89.1 % | 1157.49 | 93.9 % | 1.054× |
| 32768 | 10 | 2421.45 | 89.8 % | 2308.08 | 94.2 % | 1.049× |

Memoria: a S=32768, `ncopy=10` son 10 × 512 MiB = **5.0 GiB** de pool; encaja
sin problema (el sistema tiene 123 GiB y la GPU usa memoria unificada).
Validación: todas las celdas `PASS max_rel` entre 1.48e-05 y 2.76e-05
(criterio 1e-3).

### Barrido NSEG en los extremos

**S=32768, HND, ncopy=10** (misma tanda, 60 iters):

| NSEG | µs | % roof |
|---|---|---|
| **1** | **2341.92** | **92.8 %** |
| 2 | 2860.85 | 76.0 % |
| 4 | 3067.83 | 70.9 % |
| 8 | 3153.93 | 68.9 % |
| 16 | 3521.96 | 61.7 % |

**S=128, ncopy=48** (misma tanda, 500 iters):

| variante | µs | % roof |
|---|---|---|
| NSEG=1 KPW=4 NHD | 13.94 | 60.9 % |
| NSEG=1 KPW=2 HND | **13.85** | **61.3 %** |
| NSEG=1 KPW=4 HND | 14.21 | 59.7 % |
| NSEG=1 KPW=8 HND | 18.03 | 47.1 % |
| NSEG=2 KPW=4 NHD | 14.72 | 57.7 % |
| NSEG=4 KPW=4 HND | 18.20 | 46.6 % |
| NSEG=8 KPW=4 HND | 20.92 | 40.6 % |
| NSEG=16 KPW=4 HND | 28.78 | 29.5 % |

**NSEG=1 gana en los dos extremos, y por más margen que a S=2048.** La
hipótesis de "S grande → más paralelismo → NSEG mayor" queda **refutada**:
a S=32768 NSEG=2 ya cuesta 1.22× y NSEG=16 cuesta 1.50×. La dispersión
entre tandas es ~1.5 % (14.10 vs 13.94; 14.44 vs 14.21).

### S=128 es el caso peligroso — cuantificado

**Triton**, S=128, con y sin el flag:

| | capas | working set | µs | % roof |
|---|---|---|---|---|
| sin `--min-working-set-mb` | 10 | 20 MiB | 17.79 | 47.7 % (falso) |
| con `--min-working-set-mb 96` | 48 | 96 MiB | 23.80 | 35.7 % |

**sesgo optimista = +33.8 %**, del mismo orden que el +42.5 % que midió F0 en
`q1s512` 32/8/128. A S≥1024 el flag es un no-op (110.98 vs 111.02 a 1k;
198.86 vs 199.15 a 2k): ahí 10 capas ya superan los 96 MiB.

**HIP**, S=128, barrido de `ncopy` (LAYOUT=0, NSEG=1):

| ncopy | working set | µs | GiB/s |
|---|---|---|---|
| 1 | 2 MiB | 8.76 | 223.0 |
| 4 | 8 MiB | 8.79 | 222.1 |
| 12 | 24 MiB | 8.69 | 224.9 |
| **24** | **48 MiB** | **13.91** | **140.4** |
| 48 | 96 MiB | 14.10 | 138.6 |

**La rodilla está exactamente en el MALL de 32 MiB** (entre 24 y 48 MiB de
working set), y cuesta un factor 1.60×. Con ncopy≤12 el kernel "alcanza"
223 GiB/s = 103 % del roofline de 230 GiB/s, que es la señal delatora de una
medida caliente. Confirma que el protocolo del flag es correcto y necesario.

### Controles negativos (S=48, criterio `max_rel ≤ 1e-3`)

| mutante | max_rel | veredicto |
|---|---|---|
| limpio NHD / HND | 3.43e-05 | PASS |
| MUTATE=1 (off-by-one causal) | 3.73e+01 | FAIL ✓ |
| MUTATE=2 (layout cruzado) | 2.07e+02 | FAIL ✓ |
| MUTATE=3 (V leído del offset de K) | 2.24e+02 | FAIL ✓ |

Nota: `MUTATE=2` solo parchea la rama `LAYOUT==0` del `kv_off`, así que bajo
`-DLAYOUT=1` es un no-op y pasa. Hay que ejecutarlo con `-DLAYOUT=0`.

---

## Conclusiones

**1. El % del roofline no satura en 72 % — sigue subiendo.**
Triton NHD va de 35.7 % (S=128) a 76.6 % (S=32k), monótono. La curva no se
estanca; simplemente amortiza overhead fijo cada vez mejor. Extrapolando, el
techo asintótico de Triton está en ~78-80 %.

**2. HND aguanta en DRAM: no se diluye.**
En Triton la ganancia de HND es **planísima, 1.065-1.073×** de 1k a 32k, aunque
a 32k el KV (512 MiB) sea 16× el MALL. La hipótesis de que HND solo ataca
conflictos de caché **queda refutada**: lo que arregla es el patrón de acceso
del propio kernel (los 16 heads de un token quedan a 16 KiB de distancia en NHD
frente a 1 KiB en HND), y eso importa igual venga de MALL o de DRAM.

En el kernel HIP la ganancia sí **decae suavemente** con S: 1.070× a 4k →
1.049× a 32k. El HIP ya satura tanto el bus (94.2 % del roofline a 32k) que
queda menos margen que recuperar.

**3. Sí hay un S donde HIP+HND pierde contra HIP+NHD: S=128.**
Único cruce de la tabla: HND 14.44 µs vs NHD 14.10 µs = **0.976×**, reproducido
en dos tandas independientes (14.21 vs 13.94). Con solo 8 páginas de 16 tokens,
HND reparte 16 heads × 16 tokens en un stride de 16 KiB dentro de una página
diminuta; el stride grande `stride_head=16384` no se amortiza. Triton no cruza
(1.036× a S=128) porque su overhead fijo tapa la diferencia.

**Ningún S en el que el kernel HIP pierda contra Triton.** La ventaja
HIP/Triton (mismo layout) es mínima a 32k (1.172×) y máxima a 128 (**1.688×**),
justo lo contrario de lo esperado: la ventaja del HIP viene sobre todo de su
**overhead fijo mucho menor**, no de un bucle interno mejor. A S=128 el HIP
saca 60.2 % del roofline donde Triton saca 35.7 %.

**4. NSEG óptimo = 1 en todo el rango. La hipótesis "S grande → NSEG mayor" es
falsa.** A S=32768 (HND): NSEG=1 → 2341.92 µs, NSEG=2 → 2860.85 (1.22×),
NSEG=16 → 3521.96 (1.50×). El castigo por subir NSEG **crece** con S, no
decrece, porque `reduce_segments` lee `NSEG·NWAVE` parciales por (m, h) y el
tráfico extra escala con NSEG mientras el paralelismo ya está saturado por los
32 heads × 8 waves. A S=128 igual: NSEG=1 → 13.85-14.21, NSEG=16 → 28.78
(2.08×). **La recomendación NSEG=1 de F6 se generaliza a todo el rango.**

**5. KPW sigue siendo 4 salvo en S=128, donde KPW=2 empata.**
A S=128 con HND: KPW=2 → 13.85, KPW=4 → 14.21, KPW=8 → 18.03 µs. La diferencia
KPW 2 vs 4 (2.5 %) está al borde de la dispersión entre tandas (1.5 %), así que
no es una recomendación fuerte; lo claro es que **KPW=8 es malo en todas partes**.

### Dónde deja de compensar cada optimización

| optimización | compensa en | deja de compensar |
|---|---|---|
| HND (Triton) | **todo el rango**, 1.065-1.073× | nunca (baja a 1.036× en S=128 pero sigue ganando) |
| HND (HIP) | S ≥ 1024, 1.05-1.07× | **S=128: pierde (0.976×)** |
| kernel HIP vs Triton | **todo el rango** | nunca; mejor cuanto más corto el contexto |
| NSEG=1 | **todo el rango** | nunca |
| KPW=4 | S ≥ 1024 | S=128: KPW=2 empata o mejora ligeramente |

---

## Reproducir

```bash
source /scratch/rogarcia/vllm/fleet/fleetenv.sh /scratch/rogarcia/f3-hip

# Filas Triton (NHD y HND)
bash /scratch/rogarcia/vllm/fleet/f9/tri.sh NHD out_nhd q4s128 q4s1k q4s2k q4s4k q4s8k q4s16k q4s32k
bash /scratch/rogarcia/vllm/fleet/f9/tri.sh HND out_hnd q4s128 q4s1k q4s2k q4s4k q4s8k q4s16k q4s32k
/scratch/rogarcia/vllm-build/.venv/bin/python /scratch/rogarcia/vllm/fleet/f9/sum.py /scratch/rogarcia/vllm/fleet/f9/out_*.json

# Filas HIP (ajusta NCOPY para que ncopy*KV_por_capa >= 96 MiB)
cd /scratch/rogarcia/f3-hip/fleet/hip
S=128   NCOPY=48 ITERS=500 TAG=x bash /scratch/rogarcia/vllm/fleet/f9/hsweep.sh \
   "-DNSEG=1 -DKPW=4 -DLAYOUT=0" "-DNSEG=1 -DKPW=4 -DLAYOUT=1"
S=32768 NCOPY=10 ITERS=60  TAG=x bash /scratch/rogarcia/vllm/fleet/f9/hsweep.sh \
   "-DNSEG=1 -DKPW=4 -DLAYOUT=0" "-DNSEG=1 -DKPW=4 -DLAYOUT=1"

# Controles negativos (MUTATE=2 requiere LAYOUT=0)
S=48 ITERS=0 NCOPY=4 TAG=nc SRC=$PWD/decode_attn_paged_mutants.hip \
  bash /scratch/rogarcia/vllm/fleet/f9/hsweep.sh \
   "-DNSEG=1 -DKPW=4 -DLAYOUT=0" "-DNSEG=1 -DKPW=4 -DLAYOUT=0 -DMUTATE=1" \
   "-DNSEG=1 -DKPW=4 -DLAYOUT=0 -DMUTATE=2" "-DNSEG=1 -DKPW=4 -DLAYOUT=0 -DMUTATE=3"
```

## Artefactos

| ruta | qué es |
|---|---|
| `fleet/f9/tri.sh` | fila Triton, un layout, N specs, salida JSON |
| `fleet/f9/hsweep.sh` | fila HIP con `NCOPY` controlable (`psweep.sh` lo fijaba en 4) |
| `fleet/f9/sum.py` | resume los JSON a µs + % del roofline + capas |
| `fleet/f9/tri_{nhd,hnd}_{a,b}.json` | datos crudos Triton |
| `fleet/f9/tri_nhd_nowsflag.json` | control sin `--min-working-set-mb` |
| `f3-hip/fleet/hip/decode_attn_paged_f9.hip` | `decode_attn_paged.hip` + `dkv[64]` + `ncopy`/`NSEG`/`S` en la línea TIME |

`decode_attn_paged_f9.hip` difiere del original solo en el host: array de copias
de 16 → 64 y más campos en el printf. El kernel es byte-idéntico.

## Limitaciones

- Una sola secuencia (batch=1), `shuf=0`, `poolx=1`. Con un allocator realista
  (`shuf=1`) las cifras HIP a S grande podrían empeorar; F5 lo midió a S=2048.
- Triton fp16 vía `benchmark.py` (dtype "auto" del modelo), HIP fp16 dense→paged
  reempaquetado en host. Los dos usan `block_size=16`.
- El % del roofline usa 230 GiB/s del protocolo. El HIP a 32k mide 216.6 GiB/s
  reales, así que el "94.2 %" es contra un techo nominal, no contra un STREAM.

## Log

- entorno verificado (`fleetcheck` ok), GPU Radeon 8060S, 123 GiB de RAM.
- S=2048 Triton NHD reproducido a 199.15 µs (baseline conocido: 198.79 µs, 0.2 %).
- Triton NHD y HND completos. 48 capas a S=128, 10 en el resto.
- HIP NHD y HND completos, 7 valores de S. ncopy=48 a S=128, 10 en el resto.
  A S=32768 el pool son 10 × 512 MiB = 5.0 GiB, sin problema de memoria.
- Barrido NSEG a S=128 y S=32768, más KPW a S=128.
- Rodilla del MALL localizada barriendo ncopy a S=128 (entre 24 y 48 MiB).
- Controles negativos: 3/3 mutantes detectados por `max_rel`.
- Nada fuera de `fleet/` tocado; sin commits, sin `git clean`.
