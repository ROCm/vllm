# f2-micro — microbenchmarks HIP que deciden el diseño del kernel de atención

Protocolo: `quiet-lock measure amd-gpu-lock`, reloj realtime (nunca
SHADER_CYCLES), warm-up de 3 s dentro de la ventana de telemetría, 2 ventanas
descartadas + 4–7 útiles, mediana y rango, sclk reportado en cada fila, y
**fila de control repetida al final de cada barrido** para descartar deriva.
Todo medido a **2890–2900 MHz (boost)**, stdev ≤ 5 MHz salvo en la primera
fila de cada barrido.

## Resultado en una línea

En el layout que vLLM **realmente** usa (NHD paginado, `stride_head = 1 KiB`),
**padear no sirve: el valor sin padear ya es óptimo (1.00×)**. Lo que sí existe
es un **acantilado de 1.90×–3.68× cuando `stride_head % 2048 == 0`**, y he
demostrado por dos vías independientes que su mecanismo es **conflicto de sets
en GL1, no interleave de canal DDR**.

## Cifras

Caso de estudio `Hkv=16, D=256, S=2048, fp16`, split-KV NSEG=5 → grid 80 WG ×
1 wave, `global_load_b128` non-temporal, 31.2 MiB/llamada, working set
432 MiB. Roofline de la board 232 GiB/s ⇒ **141 µs** es el suelo teórico.

| configuración de `stride_head` | µs | GiB/s | % roofline | vs. real |
|---|---:|---:|---:|---:|
| **1024 B (lo que vLLM asigna hoy)** | **147.6** | **206.8** | **89 %** | 1.000× |
| 1152 B (`%256 == 128`, el óptimo que predecía F0) | 150.6 | 202.7 | 87 % | 0.980× |
| 1280 B (mejor del barrido) | 145.8 | 209.3 | 90 % | 1.012× |
| 1088 B (`%256 == 64`) | 160.0 | 190.8 | 82 % | 0.923× |
| **2048 B (acantilado)** | **283.3** | **107.7** | 46 % | **0.521×** |
| **4096 B (acantilado)** | **551.2** | **55.4** | 24 % | **0.268×** |

## Hipótesis probadas

| # | hipótesis (§) | veredicto | evidencia |
|---|---|---|---|
| 1 | Padear el stride de KV da **2.55×** (§7.1) | **REFUTADA** | en el layout real, el óptimo es **1.012×** y está dentro del ruido; el valor actual ya es bueno |
| 1a | El mecanismo es **2.39× más bytes de DRAM** (§7.1) | **REFUTADA** | `FETCH_SIZE` = 31.25 MiB (×1.000) en **todos** los strides, incluidos los acantilados |
| 1b | El predictor es `stride_head % 256`, óptimo en 128 (F0) | **REFUTADA** | `%256 == 128` sale **peor** que el actual (202.7 vs 206.8). El predictor real es `% 2048` |
| 1c | El mecanismo es **interleave de canal LPDDR5X** | **REFUTADA** | el efecto desaparece al bajar de 16 a 1 head concurrente; un efecto de canal no dependería de eso |
| 1d | El mecanismo es **conflicto de sets de caché** | **CONFIRMADA** | `GL1C_STALL_GL2_GL1` ×7.3 en el acantilado con `FETCH_SIZE` y `GL2C hit` intactos; el efecto escala con heads concurrentes |
| 1e | El stride de **página** con paged KV importa | **REFUTADA** | irrelevante entre 256 KiB y 1 MiB (206.5–207.1 GiB/s, rango 0.3 %) |
| 4 | Profundidad óptima **4–6 loads/SIMD**, tope ~10 (§7.2) | **PARCIAL** | satura en **4**; plano de 4 a 16; **a 20 cae un 15 %** |
| 4a | El default del compilador da **0.7 %** del pico (§7.2) | **REFUTADA** | el peor caso construible (2 en vuelo) da **72 %** |
| 2 | dot2/VOPD vs WMMA en `P@V` (§5.3) | parcial | VGPR medidos; throughput bloqueado (ver abajo) |
| 3, 5 | Partir `D_v`; bancos de VGPR en WMMA | no iniciados | sin tiempo de GPU |

---

## Experimento 1 — el stride del KV (§7.1)

### El layout del documento de diseño no existe en este stack

`TritonAttentionBackend.get_kv_cache_shape` con el layout por defecto
(`get_kv_connector_cache_layout()` → **NHD**, `kv_connector/utils.py:51`)
asigna `(num_blocks, block_size, num_kv_heads, 2*head_size)` con **K y V
empaquetados en la dimensión de contenido**. Con `Hkv=16, block_size=16,
D=256, fp16`:

| stride | valor | de dónde sale |
|---|---:|---|
| `stride_head` | **1 KiB** | `2 * 256 * 2` (K\|V empaquetados) |
| `stride_token` | 16 KiB | `Hkv * stride_head` |
| `stride_page` | 256 KiB | `block_size * stride_token` |

**`S` no aparece en ningún stride.** El eje de secuencia es el block table. El
«1 MiB exacto, el peor caso posible» del §7.1 **no se asigna nunca**. Mis dos
primeros kernels (`kernel.hip`, `paged.hip`) midieron layouts que nadie usa;
los dejo en el repo porque acotan el fenómeno, pero **el resultado que cuenta
es el de `nhd.hip`**.

### En el layout real, padear no sirve

Barrido de `stride_head % 256`, con la fila real repetida al final como
control. Block table barajado (lo que produce un allocator real):

| pad | `stride_head` | `%256` | GiB/s | vs. real |
|---:|---:|---:|---:|---:|
| 0 | **1024** | 0 | **206.8** | — |
| +32 | 1056 | 32 | 184.7 | 0.893× |
| +64 | 1088 | 64 | 190.8 | 0.923× |
| +96 | 1120 | 96 | 187.7 | 0.908× |
| +128 | 1152 | **128** | 204.6 | **0.989×** |
| +160 | 1184 | 160 | 188.1 | 0.910× |
| +192 | 1216 | 192 | 192.2 | 0.929× |
| +224 | 1248 | 224 | 186.5 | 0.902× |
| 0 (control) | 1024 | 0 | **206.8** | 1.000× |

El control cierra en 206.8 contra 206.8: sin deriva. Idéntico con block table
identity (207.7 → 207.7), así que el orden de páginas no interviene.

**Esto refuta la predicción de F0 de que el óptimo está en `%256 == 128`**:
1152 B sale **por debajo** del valor actual. Lo que la tabla sí dice es que
`%256 == 0` es lo mejor y cualquier desalineación de 256 B cuesta 7–10 %:
**la alineación a 256 B importa, pero en el sentido contrario al propuesto** —
hay que *mantenerla*, no romperla.

Barriendo solo múltiplos de 256 B, el techo es ~1.01× y está en el ruido:

| `stride_head` | 1024 | 1280 | 1536 | 1792 | **2048** | 2304 | 2560 | 3072 | **4096** | 5120 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GiB/s | 206.8 | 209.3 | 209.1 | 209.5 | **109.1** | 208.9 | 208.8 | 203.2 | **55.4** | 201.2 |

### El acantilado: `stride_head % 2048 == 0`

Lo único de magnitud en toda la tabla. Confirmado en tres puntos con vecinos
sanos a un lado y otro:

| `stride_head` | 1024 | **2048** | 2304 | **4096** | 4352 | **8192** | 8448 |
|---|---:|---:|---:|---:|---:|---:|---:|
| GiB/s | 206.8 | **109.1** | 208.8 | **55.4** | 208.5 | **56.2** | 208.5 |
| caída | — | **1.90×** | — | **3.76×** | — | **3.68×** | — |

Un solo paso de 256 B fuera del múltiplo de 2 KiB lo cura por completo.

**Atribución del acantilado**: fijando `stride_token` independientemente de
`stride_head` (el kernel los toma como argumentos separados), con
`stride_head = 2048`:

| `stride_token` | 32 KiB (= 16·2048) | 33 KiB | 36 KiB | 48 KiB |
|---|---:|---:|---:|---:|
| GiB/s | **109.1** | 204.0 | **108.5** | **108.3** |

Y fijando `stride_page` entre 256 KiB y 1 MiB con `stride_head = 1024`: 206.3–207.1
GiB/s, rango 0.3 % — **`stride_page` es irrelevante**. El acantilado pertenece
a `stride_head`; `stride_token` solo lo hereda por ser su múltiplo.

### El mecanismo: conflicto de sets en GL1, NO interleave de canal

Las dos hipótesis hacen predicciones opuestas sobre una variable: **cuántas
columnas de kv-head se tocan a la vez**. Un conflicto de sets necesita varias
direcciones compitiendo por el mismo set; un efecto de canal DDR no depende de
eso. Añadí `heads_live` al kernel manteniendo **los bytes leídos constantes**
(cada WG conserva su rango disjunto de páginas; solo cambia cuántos múltiplos
distintos de `stride_head` están vivos):

| `heads_live` | `sh=1024` GiB/s | `sh=2048` GiB/s | **caída del acantilado** |
|---:|---:|---:|---:|
| 1 | 63.7 | 63.5 | **1.00× — no hay acantilado** |
| 2 | 129.4 | 113.9 | 1.14× |
| 4 | 226.7 | 111.1 | 2.04× |
| 8 | 222.3 | 109.9 | 2.02× |
| 16 | 206.7 | 107.7 | 1.92× |

**Con una sola head el acantilado desaparece por completo** (63.7 vs 63.5,
0.3 % de diferencia). Aparece al llegar a 2 y satura en 4. Esto es
incompatible con interleave de canal DDR —que no sabe cuántas columnas hay
vivas— y es la firma exacta de un conflicto de sets: hacen falta ≥ 2
direcciones mapeando al mismo set, y con 4 el set (de asociatividad ~4) ya
está lleno.

Contadores de hardware, que cierran el argumento (cada grupo en su propia
pasada; gfx1151 no puede armar `FETCH_SIZE` y `GL2C_*` a la vez — error 38):

| caso | FETCH_SIZE | vs. pedido | GL2C hit | **GL1C_STALL_GL2_GL1** |
|---|---:|---:|---:|---:|
| `sh=1024`, hl=16 (rápido) | 31.25 MiB | ×1.000 | 0.04 % | **177** |
| `sh=2048`, hl=16 (**acantilado**) | 31.25 MiB | ×1.000 | 0.04 % | **1294** |
| `sh=2304`, hl=16 (rápido) | 31.25 MiB | ×1.000 | 0.04 % | **119** |
| `sh=1024`, hl=1 | 18.22 MiB | ×0.583 | 41.56 % | 11276 |
| `sh=2048`, hl=1 (sin acantilado) | 18.54 MiB | ×0.593 | 40.46 % | 10442 |

Las tres primeras filas son la prueba: **mismos bytes de DRAM al 0.0 %, mismo
hit rate de L2 al 0.0 %, y `GL1C_STALL_GL2_GL1` multiplicado por 7.3** (177 →
1294) exactamente en el acantilado, volviendo a 119 en cuanto se desalinea
256 B. El único contador que se mueve mide stalls en la ruta **GL1←GL2**:
dentro de la jerarquía de caché, no en el DDR.

Esto **reproduce de forma independiente** el hallazgo de F0
(`GL1C_STALL_GL2_GL1` −61 % con `FETCH_SIZE` sin cambio) y además lo
**explica**: el número de heads concurrentes es la variable causal.

Las dos últimas filas son el control que valida la construcción del
experimento: con `heads_live=1` los 80 WG comparten una sola columna, así que
se solapan en caché — `FETCH_SIZE` baja a 0.583× y el hit de L2 sube a 41 %.
Menos tráfico y sin embargo **3× menos ancho de banda** (63.7 vs 206.8): con
una sola columna no hay paralelismo de bancos suficiente. Es el recordatorio
de que en este régimen **el cuello no son los bytes**.

### Recomendaciones que salen de aquí

1. **NO padear el stride del KV.** El valor actual (1 KiB) ya es óptimo dentro
   del 1 %. La fila 1 del §0 («stride del KV, 2.55×») debe **retirarse**, no
   solo reducirse.
2. **Sí añadir una guarda**: rechazar/padear cualquier configuración cuyo
   `stride_head` sea múltiplo de 2048 B. Se alcanza con formas plausibles:
   `D=512 fp16` da `stride_head = 2 KiB` exacto → **1.90× de pérdida
   silenciosa**; `D=1024 fp16` da 4 KiB → **3.76×**. Es una comprobación de
   una línea en `get_kv_cache_shape` que evita un acantilado invisible.
3. **Mantener la alineación a 256 B.** Romperla cuesta 7–10 % (filas +32…+224).
4. El padding **nunca** debe elegirse por `% 256`; el predictor correcto es
   `stride_head % 2048 != 0` **y** `stride_head % 256 == 0`.

---

## Experimento 4 — profundidad de pipeline (§7.2)

Mismo kernel, `DH` como macro ⇒ `2*DH` `global_load_b128` en vuelo (K y V son
dos streams). 80 WG × 1 wave = 1.0 wave/SIMD, el punto de operación de NSEG=5.

| DH | en vuelo | VGPR | GiB/s (pad +256) | % de 232 | GiB/s (sin pad) |
|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 21 | 166.7 | 72 % | 150.1 |
| **2** | **4** | **37** | **206.4** | **89 %** | 182.0 |
| 3 | 6 | 53 | 208.3 | 90 % | 192.6 |
| 4 | 8 | 69 | 205.6 | 89 % | 189.9 |
| 5 | 10 | 85 | 208.0 | 90 % | 193.7 |
| 6 | 12 | 93 | 205.1 | 88 % | 188.0 |
| 8 | 16 | 93 | 207.0 | 89 % | 199.3 |
| 10 | 20 | 101 | **175.1** | 75 % | 177.2 |

1. **Satura en 4 en vuelo, no en 6.** De 4 a 16 la curva es plana dentro del
   ±1.5 %. Ahorrarse 2 niveles son **32 VGPR/lane**, que con D=256 es
   exactamente el margen para bajar de los 96 VGPR objetivo del §0.
2. **El coste por nivel es 16 VGPR, no 4.** El §7.2 dice «4 VGPR por nivel»,
   cierto para un `load_b128` de un stream. Aquí cada nivel añade un load de K
   y otro de V más su acumulador: 21 → 37 → 53 → 69 → 85, **+16 por nivel**,
   leído del ELF. Con D=256 esto convierte la profundidad en un gasto de
   registros de primer orden.
3. **Pasarse penaliza: 20 en vuelo cuesta 15 %** (207 → 175). Coherente con el
   tope de ~10 loads/wave del VMEM return buffer (§2.6). El aviso del §7.3
   («pasarse de 4–6 penaliza») vale también con accesos coalescidos, pero el
   umbral está en 16–20, no en 6.
4. **El 0.7 % del §7.2 no se reproduce**: el peor caso construible (2 en vuelo)
   da **72 %**. Ese 0.7 % es el régimen de 1 wave; aquí hay 80 waves en 80 SIMD.

---

## Evidencia de ASM

### El bucle tiene la profundidad pedida

`kv_stream_d3_w1`, `.vgpr_count: 53`, `.vgpr_spill_count: 0`:

```
.LBB0_2:                                ; =>This Inner Loop Header: Depth=1
	global_load_b128 v[29:32], v[27:28], off slc dlc
	global_load_b128 v[33:36], v[25:26], off slc dlc
	global_load_b128 v[37:40], v[27:28], off offset:512 slc dlc
	global_load_b128 v[41:44], v[25:26], off offset:512 slc dlc
	global_load_b128 v[45:48], v[27:28], off offset:1024 slc dlc
	global_load_b128 v[49:52], v[25:26], off offset:1024 slc dlc
	s_waitcnt vmcnt(5) ... vmcnt(0)
	s_cbranch_scc1 .LBB0_2
```

`slc dlc` = `__builtin_nontemporal_load` produce el bypass de L1/L2 que §8.2
pide para K/V, y el contador lo confirma (GL2C hit 0.04 %).

### Trampa de scheduling: «rolling» no da profundidad

La forma natural del prefetch software (mantener `DH` buffers y reemplazar uno
por iteración) **no** produce la profundidad pedida — HIPCC la reordena a
`s_waitcnt vmcnt(1)` con 2 loads efectivos:

```
	s_waitcnt vmcnt(1)          <-- solo 2 en vuelo, no 6
	v_xor3_b32 ...
	global_load_b128 ...
	...
	s_waitcnt vmcnt(5)
	s_waitcnt vmcnt(3)          <-- dos waitcnt seguidos
```

Hay que escribirlo como **batch** (emitir los `DH` loads, consumirlos todos)
con **un acumulador independiente por slot**. Es una restricción de escritura
para el kernel HIP final.

### VGPR reales de las tres rutas de `P@V` (§5.3, §6.1)

`D_v=256`, `TILE=16`, `BLOCK_M=8` (M×R=8), wave32, operandos acotados a
`N_BUF=2` para que ninguna variante spillee:

| ruta | `.vgpr_count` | spills | acc teórico (§6.1) |
|---|---:|---:|---:|
| WMMA (BLOCK_M=16, forzado por el fragmento) | **153** | 0 | 128 |
| `__builtin_amdgcn_fdot2` (BLOCK_M=8) | **129** | 0 | 64 |
| `v_pk_fma_f16` (acc fp16) | **50** | 0 | 32 |

El dato que decide es otro: manteniendo residentes los 16 fragmentos de V (el
caso «todo en registros»), **WMMA sube a 249 VGPR con 24 spills** mientras
fdot2 se queda en 129 sin spills. Con D=256 el presupuesto de registros es el
cuello (§6.4) y WMMA lo agota. Concuerda con lo que midió F4.

---

## Qué NO se pudo probar y por qué

**Experimento 2 (throughput de `P@V`) — bloqueado por un problema real.** El
emparejado VOPD desde `__builtin_amdgcn_fdot2` sale a **7
`v_dual_dot2acc_f32_f16` de 505 `v_dot2acc_f32_f16`** (1.4 %) en mi bucle,
frente al 10/10 de `rdna35-expert/microbench/vopd_promotion`. La diferencia no
es «≥2 acumuladores» (yo tengo 64): con 64 acumuladores vivos el pairer casi
nunca encuentra pareja válida bajo la restricción del puerto SRC2 (§2.3).
Medir así compararía WMMA contra dot2 **single-issue** y daría un veredicto
falso. **Hace falta caracterizar la restricción de emparejado —cuántos
acumuladores admite antes de dejar de emparejar— antes de que la medida
signifique algo.** Es un experimento acotado y de valor; lo dejo planteado.

**Experimentos 3 y 5**: no iniciados. El tiempo de GPU se fue en cerrar el
mecanismo del stride, que era la prioridad. La infraestructura queda lista.

**Desglose de stalls `vmcnt` vs `lgkmcnt`** (punto 5 del §5 del protocolo): no
es medible en gfx1151, solo existe `SQ_WAIT_CNT_ANY`. Lo suplí con
`GL1C_STALL_GL2_GL1`, que resultó ser el contador informativo.

**Frecuencia**: todo a boost (2890–2900 MHz). El kernel es memory-bound y §2.8
dice que el plateau de DRAM es independiente del reloj shader, así que los
GiB/s son trasladables a la frecuencia sostenida de ~2100 MHz; **las cifras
por ciclo no lo serían**.

---

## Avisos a la flota

### 1. Un bug de medida que puede afectar a cualquiera

Si un harness carga **varios `.so` compilados del mismo fuente** en el mismo
proceso, el runtime de HIP resuelve todos los lanzamientos al **primer**
kernel registrado con ese nombre: ejecuta el código equivocado con los
argumentos correctos, **en silencio y sin error**.

Me dio un DH=3 «a 496 GiB/s» (214 % del roofline) — exactamente 2.9× lo
correcto, porque corría el cuerpo de DH=1 con el contador de vueltas de DH=3.
Lo delató que superase el roofline; con un factor menor habría pasado por
buena. **Mitigación**: meter el nombre del kernel, del `launch_` y del
`describe` detrás de una macro que concatene los parámetros de compilación
(`kv_stream_d3_w1`). Hecho en `kv_stride_alias/kernel.hip` y `nhd.hip`.

### 2. `mblib.HipRuntime.memcpy_h2d` corrompe datos

Construye un `bytearray` temporal y toma su dirección con
`ctypes.c_char.from_buffer(bytearray(view))`; el temporal puede liberarse
antes de que la copia se complete. Me produjo un block table de basura y un
`HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`. Copiar desde un buffer ctypes
que se mantenga vivo (ver `driver_nhd.py`). **No lo he arreglado en `mblib`**
porque es repo de otro proyecto; queda avisado.

### 3. rocprofv3 en gfx1151

- Falla con «Request exceeds the capabilities of the hardware to collect»
  (error 38) si se arman `FETCH_SIZE` y `GL2C_*` a la vez → una pasada por
  grupo de contadores.
- Falla igual si el proceso padre ya inicializó la GPU vía torch → la pasada
  de contadores tiene que vivir en su propio script torch-free.

---

## Artefactos

| ruta | qué es |
|---|---|
| `fleet/micro/README.md` | cómo reproducir cada medida |
| `fleet/micro/run.sh` | envoltorio `quiet-lock measure amd-gpu-lock` |
| `fleet/micro/common.py` | protocolo de medida + extracción de VGPR del ELF de dispositivo |
| **`fleet/micro/kv_stride_alias/nhd.hip`** | **el layout NHD real; `heads_live` es el knob discriminante** |
| `fleet/micro/kv_stride_alias/harness_nhd.py` | barrido de stride con `--heads-live`, `--token-stride`, `--page-stride` |
| `fleet/micro/kv_stride_alias/counters_nhd.py` | contadores: FETCH_SIZE, GL2C, GL1C_STALL_GL2_GL1 |
| `fleet/micro/kv_stride_alias/results_nhd.csv` | barrido `%256`, identity + shuffled |
| `fleet/micro/kv_stride_alias/results_nhd_rule.csv` | la regla `%2048`, tres acantilados |
| **`fleet/micro/kv_stride_alias/results_headslive.csv`** | **el experimento discriminante** |
| `fleet/micro/kv_stride_alias/results_counters_nhd.csv` | contadores del acantilado |
| `fleet/micro/kv_stride_alias/kernel.hip`, `paged.hip` | layouts contiguos (no usados por vLLM; acotan el fenómeno) |
| `fleet/micro/pipeline_depth/results.csv` | curva de profundidad D1..D10 |
| `fleet/micro/pv_matmul/kernel.hip` | `P@V` en WMMA / fdot2 / pk_fma_f16 |

No he tocado ningún fichero compartido del repo: todo vive bajo `fleet/`.
