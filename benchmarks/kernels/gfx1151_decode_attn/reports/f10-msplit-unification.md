# F10 — un solo kernel: MSPLIT, y el barrido contra Triton en cuatro formas

`D=256, M=4`, fp16, HND, `block_size=16`, batch = 1 secuencia.
Roofline del protocolo: 230 GiB/s → `t_roof = 2·S·Hkv·D·2 / 230GiB/s`.

**Estado: COMPLETO.** 28/28 celdas contra Triton en cuatro formas, barrido
`MSPLIT × Hkv × NSEG` (48 celdas), fork eliminado, heurística conectada.

## Resumen en seis líneas

1. **Producción y el fork eran un kernel con un parámetro.** `MSPLIT` = cuántas
   waves se reparten la dimensión de tokens; 1 es producción, `MAXM` el fork.
2. **Ganamos en las 28 celdas** contra `TRITON_ATTN`, de 1.06× a 1.73×.
3. **A contexto largo llegamos al 92.9-96.5 %** del roofline; Triton se queda
   en 81.8-88.7 % y no recupera esos 8-14 puntos ni con S infinito.
4. **`_segments_for` ya era correcto**: apunta a 32 workgroups y acierta en las
   cuatro formas. El hueco que se le atribuyó era de `MSPLIT`.
5. **El predictor por bytes/token está refutado**: la ganancia de `MSPLIT`
   no es monótona en `Hkv` (+3.8 / +7.5 / +11.1 / +3.6 % a S=128).
6. **La ocupación no limita**, ahora también por el eje de LDS: 33 KiB y la
   mitad de waves residentes es más rápido que 8 KiB y el doble.

## Tabla principal — nuestro kernel contra el que vLLM usa hoy

Defaults de ambos backends, 7 repeticiones, mediana de medianas.

### `Hq=4 Hkv=2` — roofline 1.06 µs a S=128

| S | Triton | RDNA35_HIP | speedup |
| --- | --- | --- | --- |
| 128 | 9.22 (11.5 %) | 6.92 (15.3 %) | 1.33× |
| 512 | 14.06 (30.2 %) | 11.28 (37.7 %) | 1.25× |
| 1024 | 20.34 (41.7 %) | 16.10 (52.7 %) | 1.26× |
| 4096 | 50.48 (67.3 %) | 42.39 (80.1 %) | 1.19× |
| 8192 | 94.69 (71.7 %) | 81.59 (83.3 %) | 1.16× |
| 16384 | 173.32 (78.4 %) | 152.20 (89.3 %) | 1.14× |
| 32768 | 332.06 (81.8 %) | 292.63 (92.9 %) | 1.13× |

### `Hq=8 Hkv=4` — gemma-3-4b

| S | Triton | RDNA35_HIP | speedup |
| --- | --- | --- | --- |
| 128 | 11.52 (18.4 %) | 7.62 (27.8 %) | 1.51× |
| 512 | 22.03 (38.5 %) | 15.29 (55.5 %) | 1.44× |
| 1024 | 32.17 (52.8 %) | 24.54 (69.2 %) | 1.31× |
| 4096 | 92.56 (73.4 %) | 82.50 (82.3 %) | 1.12× |
| 8192 | 167.13 (81.3 %) | 154.29 (88.1 %) | 1.08× |
| 16384 | 318.49 (85.3 %) | 297.12 (91.5 %) | 1.07× |
| 32768 | 621.96 (87.4 %) | 584.57 (93.0 %) | 1.06× |

### `Hq=16 Hkv=8`

| S | Triton | RDNA35_HIP | speedup |
| --- | --- | --- | --- |
| 128 | 16.81 (25.3 %) | 10.08 (42.1 %) | 1.67× |
| 512 | 31.63 (53.7 %) | 24.44 (69.5 %) | 1.29× |
| 1024 | 50.67 (67.0 %) | 42.41 (80.1 %) | 1.19× |
| 4096 | 167.79 (81.0 %) | 153.51 (88.5 %) | 1.09× |
| 8192 | 319.76 (85.0 %) | 295.67 (91.9 %) | 1.08× |
| 16384 | 625.29 (86.9 %) | 582.80 (93.3 %) | 1.07× |
| 32768 | 1225.28 (88.7 %) | 1145.45 (94.9 %) | 1.07× |

### `Hq=32 Hkv=16` — forma de referencia

| S | Triton | RDNA35_HIP | speedup |
| --- | --- | --- | --- |
| 128 | 23.36 (36.4 %) | 13.54 (62.7 %) | 1.73× |
| 512 | 56.84 (59.8 %) | 39.93 (85.1 %) | 1.42× |
| 1024 | 101.53 (66.9 %) | 79.40 (85.6 %) | 1.28× |
| 4096 | 346.73 (78.4 %) | 288.98 (94.0 %) | 1.20× |
| 8192 | 683.92 (79.5 %) | 572.48 (94.9 %) | 1.19× |
| 16384 | 1339.70 (81.1 %) | 1130.93 (96.1 %) | 1.18× |
| 32768 | 2642.79 (82.3 %) | 2253.20 (96.5 %) | 1.17× |

La ventaja es máxima donde manda el coste fijo y mínima donde manda el bus.
Pero el suelo de Triton a contexto largo (81.8-88.7 %) no es un límite del
hardware: nosotros llegamos al 92.9-96.5 % con los mismos bytes.

## MSPLIT: la parametrización que fusiona los dos kernels

```text
MPW    = MAXM / MSPLIT      tokens que lleva cada wave
NSLICE = NWAVE / MSPLIT     rodajas de KV por workgroup
wave w -> tokens {w % MSPLIT, +MSPLIT, ...},  rodaja w / MSPLIT
```

La fila de LDS para `(token m, rodaja s)` es
`(m/MSPLIT)*NWAVE + m%MSPLIT + s*MSPLIT`, que degenera en `m*NWAVE + s` a
`MSPLIT=1` y en `m + s*MAXM` a `MSPLIT=MAXM`. Ningún extremo necesita caso
especial.

Aceptación, con el protocolo de `58769e8912` (compilar cada extremo y
diferenciar el ISA contra el kernel que dice sustituir):

| comparación | instrucciones | diferencia |
| --- | --- | --- |
| `MSPLIT=1` vs producción | 1274 vs 1278 | una `s_barrier` muerta y su `buffer_gl0_inv`, más 4 de direcciones |
| `MSPLIT=4` vs el fork | 592 vs 593 | un `s_delay_alu` y un `add` intercambiado |

Cargas, `v_dot2acc`, `ds_bpermute`, LDS y VGPR idénticos en ambos casos.

`MSTAGE` desapareció al fusionar: era `MAXM/MSPLIT` con otro nombre, que es por
qué sus 32 KiB y los 8 KiB del split siempre fueron la misma fórmula.

### Coste de LDS

`(MAXM/MSPLIT) · NWAVE · (HEAD_DIM·4 + 8)` — predice 33024 y 66048 exactamente.
Los dos arrays escalares `lds_m`/`lds_l` son los 512 B que hicieron fallar
`MAXM=8` a `MSPLIT=1` contra el techo de 64 KiB. `MSPLIT` sube solo cuando no
cabe, y el loader lo resuelve en `__post_init__` para que el nombre del símbolo
refleje lo que se compila — HIP resuelve funciones de dispositivo por nombre a
nivel de proceso.

## Barrido MSPLIT × Hkv, con GQA=2 constante

`NSEG` es el que `_segments_for` elige (32 workgroups en las cuatro formas).
Cambio respecto a `MSPLIT=1`:

| Hkv | NSEG | S=128 | S=32768 | ¿gana en ambos? |
| --- | --- | --- | --- | --- |
| 2 | 8 | +3.8 % | −4.3 % | no |
| 4 | 4 | +7.5 % | +4.5 % | sí |
| 8 | 2 | +11.1 % | +8.1 % | sí |
| 16 | 1 | +3.6 % | −6.4 % | no |

Absolutos a `MSPLIT ∈ {1,2,4}` con ese NSEG. A S=128:

| Hkv | MSPLIT=1 | MSPLIT=2 | MSPLIT=4 |
| --- | --- | --- | --- |
| 2 | 6.91 | 6.65 | 6.19 |
| 4 | 8.25 | 7.63 | 7.22 |
| 8 | 11.31 | 10.05 | 9.43 |
| 16 | 13.53 | 13.04 | 15.35 |

A S=32768:

| Hkv | MSPLIT=1 | MSPLIT=2 | MSPLIT=4 |
| --- | --- | --- | --- |
| 2 | 292.72 | 305.38 | 317.30 |
| 4 | 611.86 | 584.62 | 633.90 |
| 8 | 1246.11 | 1145.40 | 1266.89 |
| 16 | 2253.99 | 2397.96 | 3390.71 |

`MSPLIT > 1` mejora el contexto corto en las cuatro formas. El contexto largo
solo en dos. La heurística embarcada es esa tabla, declarada como tabla.

## Refutado en este bloque

| hipótesis | veredicto |
| --- | --- |
| La ganancia de `MSPLIT` escala con los bytes de KV por token | **refutada**: +3.8 / +7.5 / +11.1 / +3.6 %, pico en medio |
| La ganancia sigue al número de streams únicos de KV | **refutada**: son 64 en las cuatro formas a `MSPLIT=2`, y 128 a `MSPLIT=1` |
| `_segments_for` deja 14-17 % sin reclamar | **falso**: a `MSPLIT=1` su NSEG es el mejor de los cuatro probados |
| Gastar LDS cuesta rendimiento vía ocupación | **refutada**: 33 KiB y 24 waves/WGP baten a 16 KiB y 48 |
| El alineamiento con los 20 WGPs es criterio | **refutada por tercera vez**: NSEG=5 son 40 WGs (2.0/WGP) y pierde contra 6 y 7 |
| DPP en la mariposa de reducción | **refutada de nuevo**: +1.5 % a S=128, **−34 %** a 32k; el ISA prohíbe emparejar DPP en VOPD |
| El `s_load` de la block table es un cuello serie | **refutada**: prefetcharlo da +2.2 % a 1024 y −1.9 % a 32k, neto nulo |

## Coste fijo: lo que sigue sin ceder

Ajuste lineal sobre `Hq=8/Hkv=4`: **~5.2 µs fijos**, con el marginal ya al 90 %
del pico. A S=128 ese fijo es 2.5× el presupuesto entero del roofline.

Atacado desde cuatro ángulos independientes, todos fallidos:

| ataque | resultado |
| --- | --- |
| −13 instrucciones de preámbulo (25 %) | nada |
| −28 instrucciones y −10 máscaras de `exec` del epílogo | nada |
| quitar el bloqueo escalar de la block table | < 1 % |
| quitar 5 niveles de latencia LDS por tile | +1.5 % corto, −34 % largo |

La ablación del segundo kernel aísla **2.2 µs planos** (≈1.4 de lanzamiento,
≈0.7 de trabajo). El resto es el lanzamiento del primer kernel más la latencia
inevitable del primer toque a DRAM. **No queda nada a nivel de instrucción.**

## Abierto

- **Reducción fusionada**: ~1.4 µs recuperables. Rinde más donde peor estamos
  (`Hkv=2` a S=128, 15.3 % del roofline). Riesgo: spin acotado y contador
  determinista bajo captura de CUDA graph.
- **Eje B**: el kernel sirve una secuencia y rechaza batch > 1 hacia Triton.
  Todo este informe es batch = 1.
- **`_MSPLIT_KV_HEADS`** es una tabla de cuatro puntos. Ensancharla exige medir.
- **Desplome a 128 workgroups**: medido idéntico en dos formas
  (`Hq=8`/NSEG=16 y `Hq=32`/NSEG=4, ambos 128 WGs, 1024 waves). Sin diagnosticar.
- **Inestabilidad del arnés**: Triton a S=512 falla o da outliers de forma
  esporádica. Tratar como ruido todo lo que esté por debajo del ~2 %.

## Reproducir

```bash
cd <worktree>
export PATH=<venv>/bin:$PATH PYTHONPATH=$PWD VLLM_KV_CACHE_LAYOUT=HND
C="128 512 1024 4096 8192 16384 32768"

# tabla principal
amd-gpu-lock python benchmarks/kernels/gfx1151_decode_attn/tools/sweep.py \
    --hq 8 --hkv 4 --triton --reps 7 --contexts $C

# barrido MSPLIT
amd-gpu-lock python benchmarks/kernels/gfx1151_decode_attn/tools/sweep.py \
    --hq 8 --hkv 4 --msplit 2 --nseg 4 --contexts $C

# correctness, incluida la cola parcial S=50 y el control negativo
amd-gpu-lock python benchmarks/kernels/gfx1151_decode_attn/tools/check.py \
    --hq 8 --hkv 4 --msplit 1 2 4 --nseg 1 4 --contexts 48 50 1020 1024
amd-gpu-lock python benchmarks/kernels/gfx1151_decode_attn/tools/check.py \
    --hq 8 --hkv 4 --mutate 1 --contexts 48 --layouts 1 --nseg 1
```
