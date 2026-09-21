# Protocolo de la flota — kernel de atención decode en gfx1151

Reglas operativas **obligatorias** para todo agente que trabaje en este
proyecto. Léelas enteras antes de ejecutar nada.

Documento de diseño: `attention_kernel_design.md` (hipótesis a probar).

---

## 1. Regla de oro: el mutex compilación ↔ medición

El problema: `amd-gpu-lock` serializa los trabajos *de GPU*, pero **no impide
que otro agente compile mientras tú mides**. Una compilación satura los cores,
mueve la frecuencia del SoC y contamina cualquier medida.

**`quiet-lock` resuelve esto. Es obligatorio.**

```bash
# MEDIR (lock exclusivo — nada más corre a la vez)
quiet-lock measure amd-gpu-lock python benchmark.py ...

# COMPILAR (lock compartido — varias builds a la vez, pero nunca durante una medición)
quiet-lock build hipcc -O3 --offload-arch=gfx1151 ...
quiet-lock build pip install -e .
```

Matriz garantizada (verificada):

| | build | measure |
| --- | --- | --- |
| **build** | concurrentes | serializados |
| **measure** | serializados | serializados |

Reglas:

- **Toda** medición va con `quiet-lock measure amd-gpu-lock`. Sin excepción.
- **Toda** compilación va con `quiet-lock build`. Incluye `pip install -e .`,
  `hipcc`, `ninja`, `cargo`.
- Si `quiet-lock` devuelve **exit 75** (timeout), otro agente tiene el recurso.
  **Reporta y para. NUNCA reintentes sin el lock.**
- Trabajo que no toca ni GPU ni compilador (leer código, escribir análisis)
  **no** necesita lock. No lo tomes: bloquearías a los demás.
- `quiet-lock status` dice si está libre. `/tmp/rogarcia-quiet.lock.log` registra
  quién lo tomó.

Sube el timeout en trabajos largos: `QUIET_LOCK_TIMEOUT=7200 quiet-lock ...`

### Contención y timeouts de agente (importante)

Con varios agentes activos la GPU está muy disputada. Un comando bloqueado
esperando el lock **sin emitir salida** hace que el runner del agente lo dé por
colgado y lo mate (ya ocurrió dos veces).

Mitigaciones:

- `quiet-lock` emite un **heartbeat cada 20 s** mientras espera, indicando cuánto
  lleva y quién tomó el lock por última vez. Ajustable con
  `QUIET_LOCK_HEARTBEAT`.
- **Usa timeouts cortos y reintenta**: `QUIET_LOCK_TIMEOUT=240` es preferible a
  una espera de una hora en silencio. Exit 75 no es un fallo tuyo.
- Cuando recibas exit 75: **escribe tu progreso al informe**, sigue con trabajo
  que no necesite GPU (escribir código, compilar, analizar ASM — el lock `build`
  es compartido y casi nunca bloquea) y reintenta después.
- **Trocea las tandas largas de medición.** Diez comandos de 30 s son más
  robustos que uno de 5 min.

---

## 2. Entorno verificado

| Recurso | Ruta / valor |
| --- | --- |
| venv | `/scratch/rogarcia/vllm-build/.venv` |
| Python | `$VENV/bin/python` (**nunca** `python3` del sistema) |
| torch | 2.13.0+rocm10.1.0a20260822 |
| vLLM | editable → **`/scratch/rogarcia/vllm-build/vllm`** ⚠️ **655 commits ATRASADO** |
| hipcc | `$VENV/bin/hipcc` |
| rocprofv3 | `$VENV/bin/rocprofv3` |
| llvm-objdump / llvm-mc | `$VENV/lib/python3.12/site-packages/_rocm_sdk_core/lib/llvm/bin/` |
| GPU | `gfx1151`, 20 WGP, L2 2 MB, 96 GB |

**Ojo:** `torch.cuda` y `amd-smi` reportan `multi_processor_count=20`. Son
**WGPs**, no CUs. La board tiene 40 CU = 20 WGP = 80 SIMD.

Comprobado end-to-end: `hipcc` compila para gfx1151, `llvm-mc` ensambla
`v_wmma_f32_16x16x16_f16`, `llvm-objdump` desensambla.

### ⚠️ Tres fallos SILENCIOSOS del entorno (verificados)

**1. El editable-install apunta a un árbol obsoleto.** Sin `PYTHONPATH`,
`import vllm` carga `/scratch/rogarcia/vllm-build/vllm`, que está **655 commits
por detrás y NO tiene el path 3D para M>1**. Medir sin `PYTHONPATH` benchmarkea
el kernel equivocado sin dar ningún error.

**2. Falta el módulo `amdsmi`.** Sin él vLLM cae a `UnspecifiedPlatform` y el
benchmark falla de forma opaca. Está en
`$VENV/lib/python3.12/site-packages/_rocm_sdk_core/share/amd_smi`.

**3. `global_load_lds` NO existe en gfx1151** (verificado: `llvm-mc` lo rechaza
en gfx1151 y lo acepta en gfx942). Es CDNA-only. El §9 del documento de diseño
lo listaba como palanca para D=256: **esa palanca no existe**. Global→LDS tiene
que pasar por VGPRs.

Invocación correcta (las tres correcciones):

```bash
PYTHONPATH=/scratch/rogarcia/<tu-worktree>:$VENV/lib/python3.12/site-packages/_rocm_sdk_core/share/amd_smi \
  QUIET_LOCK_TIMEOUT=240 quiet-lock measure amd-gpu-lock $VENV/bin/python ...
```

Verifica SIEMPRE antes de medir:

```bash
PYTHONPATH=... $VENV/bin/python -c "import vllm; print(vllm.__file__)"
```

---

## 3. Ramas

| Rama | Rol |
| --- | --- |
| `rogarcia.gfx1151-3d-attn-tuning` | **Baseline y competencia práctica.** Kernel Triton 3D con soporte MTP. |
| `rogarcia.attn-bench-cache-residency` | **Herramienta de medida.** Rotación de buffers para evitar residencia en MALL. |

Ambas hacen falta a la vez: mide **siempre** el baseline con el benchmark de
residencia de caché.

### Por qué importa la residencia

`benchmark_fn` recorre un KV por capa y divide por el número de capas, así que
el bucle de capas hace de rotación. Si `num_layers × kv_bytes_per_layer` cae por
debajo de los 32 MiB de MALL, la medida es **optimista** — CUDA-graph replay
nunca vacía MALL. Medido en esta board con TRITON_ATTN decode:

| working set (10 capas) | celdas | sesgo mediano | peor |
| --- | --- | --- | --- |
| < 32 MiB | 6 | **+35 %** | **+64 %** |
| > 32 MiB | 18 | +0 % | +3 % |

**Usa siempre `--min-working-set-mb 96`** salvo que midas deliberadamente el
caso caliente. Solo sube `--num-layers`, nunca lo baja.

---

## 4. La competencia

Dos referencias, no una:

1. **Práctica:** el kernel Triton 3D de `rogarcia.gfx1151-3d-attn-tuning`.
2. **Teórica:** el **roofline de la board: 230 GiB/s**.

Para el caso de estudio (`Hq=32, Hkv=16, M=4, D=256, S=2048`), KV de una capa =
32.0 MiB → **límite inferior ≈ 141 µs por paso de decode**.

**Todo informe debe dar las tres cifras**: tu tiempo, el del baseline Triton, y
el % del roofline alcanzado. Un kernel que bate al baseline pero se queda al
40 % del roofline no está terminado.

---

## 5. Definición de "terminado"

Una implementación **no** está terminada hasta que:

1. **Correcta** — el criterio de `max_abs` por sí solo **NO BASTA** (verificado
   con controles negativos):

   - **`max_rel`**, no solo `max_abs`. Un bug de máscara causal de **una sola
     key** da `max_abs = 1.2e-03` a S=2048 y **PASA** un umbral de 2e-2; el
     `max_rel` lo delata (1.065 contra 2.0e-05 del kernel correcto). El error de
     una key escala como ~1/S mientras la tolerancia es fija, así que **cuanto
     mayor el contexto, más bugs esconde**.
   - **Un caso de S pequeño** (S≈48). El mismo bug falla limpio ahí (5.2e-02).
   - **Controles negativos**: muta el kernel a propósito (máscara desplazada una
     key, kv head equivocada en GQA) y comprueba que tu test los **detecta**. Un
     test que no falla ante un bug conocido no prueba nada.
   - Umbrales: `max_rel ≤ 1e-3` y `max_abs ≤ 2e-2` (fp16), en **todas** las
     formas del barrido.

   `fleet/hip/ref_attn.py` ya reporta ambas métricas.
2. **Medida**: con `quiet-lock measure amd-gpu-lock` y el benchmark de
   residencia de caché, ≥ 5 repeticiones, reportando mediana y dispersión.
3. **Comparada**: contra el baseline Triton **y** contra el roofline.
4. **Inspeccionada en ASM**: `llvm-objdump` del binario final, comprobando al
   menos:
   - nº de VGPR/lane reales (`.vgpr_count` en los metadatos)
   - loads en vuelo entre `s_waitcnt` (profundidad de pipeline efectiva)
   - que el `global_load` siguiente va **inmediatamente** tras el `s_waitcnt`
     (meter ALU en medio cuesta 10 %)
   - bancos de los operandos WMMA (`vgpr_start % 4`; HIPCC los pone todos en
     banco 0 y paga 34 cyc en vez de 32)
5. **Perfilada**: `rocprofv3` con contadores, reportando al menos
   `GL2C_HIT/MISS`, `FETCH_SIZE` y el desglose de stalls por `vmcnt` vs
   `lgkmcnt`. Un `s_waitcnt` total indiferenciado **no** permite decir
   "memory-bound".

6. **De configuración única** — una sola configuración de compilación gana en
   **todo** el barrido de contexto. Está prohibido elegir `NSEG`, `KPW`,
   `BLOCK`, `TILE` o el layout en función de `S`.

   No es una preferencia estética, es implementabilidad: `S` crece en cada paso
   de decode y con CUDA graphs el grid se captura, así que una configuración
   indexada por `S` no puede materializarse en tiempo de ejecución. Los
   parámetros sí pueden depender de propiedades estáticas —
   `(H_kv, D, M, board)` — porque son fijas al capturar el grafo.

   Una tabla con la mejor configuración por cada `S` no mide un kernel: mide la
   envolvente inferior de una familia de kernels, y esa envolvente no existe
   como binario. Si una configuración gana a `S` corto y pierde a `S` largo, el
   resultado que hay que reportar es el de la configuración única elegida, con
   su pérdida en el extremo malo explícita.

   Esto **no** alcanza a los parámetros del harness (`ncopy`, `iters`), que
   existen para fijar el régimen de memoria y el ruido estadístico, no para
   cambiar el código medido. Varían con `S` por diseño.

Sin los seis puntos, el trabajo se reporta como **en progreso**, no como hecho.

---

## 6. Higiene de medición

- **Cronometra con el realtime clock**, no con `SHADER_CYCLES`: este se apaga
  durante stalls de L2+ y hace wrap a 2²⁰.
- **Frecuencia**: boost ~2900 MHz, sostenida bajo carga ~2100 MHz. Reporta a qué
  reloj mediste; las cifras por ciclo a boost sobreestiman ~27 %.
- **Warmup**: la primera lectura sale ~4 % baja. Descarta las primeras
  iteraciones.
- **Rota buffers** entre repeticiones (lo hace el benchmark de residencia).
- Reporta **mediana y rango**, no solo la media.

---

## 7. Formato de informe

Todo agente termina con un fichero en `/scratch/rogarcia/vllm/fleet/<tu-id>.md`:

```markdown
# <id> — <título>

## Resultado en una línea
<qué se probó y qué salió>

## Cifras
| variante | tiempo (µs) | vs baseline | % roofline |
|---|---|---|---|

## Hipótesis probadas
| # | hipótesis (§ del doc de diseño) | veredicto | evidencia |
|---|---|---|---|
- veredicto ∈ {CONFIRMADA, REFUTADA, NO CONCLUYENTE}

## Evidencia de ASM / profiler
<extractos concretos, no resúmenes>

## Qué NO se pudo probar y por qué

## Artefactos
<rutas a código, logs, CSVs>
```

**Honestidad por encima de todo.** Si una hipótesis del documento de diseño
resulta falsa, dilo con la evidencia. Si no pudiste medir algo, dilo. Un informe
que dice "no concluyente, aquí está por qué" vale más que uno que inventa una
cifra.

---

## 8. Disciplina de git

- **No** hagas commit en `main` ni en las dos ramas de referencia.
- Trabaja en tu propia rama: `rogarcia.fleet-<tu-id>`.
- No hagas push ni abras PR salvo petición explícita del usuario.
- Si tocas ficheros compartidos, avisa en tu informe.

---

## 9. Aislamiento entre agentes (CRÍTICO — añadido tras incidente)

**El venv es compartido y su editable-install apunta a `/scratch/rogarcia/vllm`.**
Si dos agentes editan ese árbol a la vez, o uno mide mientras otro tiene cambios
sin commitear, **las medidas cargan código ajeno**. Ya ocurrió: un agente cambió
la rama del árbol compartido con modificaciones vivas en los ficheros de Triton.

### Regla

Cada agente trabaja en **su propio git worktree**, nunca en
`/scratch/rogarcia/vllm`:

```bash
cd /scratch/rogarcia/vllm
git worktree add /scratch/rogarcia/<tu-id> -b rogarcia.fleet-<tu-id>
cd /scratch/rogarcia/<tu-id>
```

Y **toda** invocación de Python que importe vLLM lleva `PYTHONPATH` a tu
worktree (verificado: anula el editable-install):

```bash
PYTHONPATH=/scratch/rogarcia/<tu-id> \
  quiet-lock measure amd-gpu-lock $VENV/bin/python benchmark.py ...
```

Comprueba siempre de dónde importa antes de medir:

```bash
PYTHONPATH=/scratch/rogarcia/<tu-id> $VENV/bin/python -c \
  "import vllm.v1.attention.ops.triton_unified_attention as m; print(m.__file__)"
```

Si imprime `/scratch/rogarcia/vllm/...` en vez de tu worktree, **para**: estarías
midiendo el árbol de otro.

### Qué NO hacer

- `git checkout` / `git switch` en `/scratch/rogarcia/vllm` — cambia la rama bajo
  los pies de los demás.
- Dejar cambios sin commitear en el árbol compartido.
- Medir sin `PYTHONPATH` apuntando a tu worktree.

`/scratch/rogarcia/vllm` queda como árbol **de solo lectura** para documentos
(`attention_*.md`, `fleet/`). Los informes sí van ahí.
