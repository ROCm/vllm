# F7 — VOPD pairing arreglado, y el veredicto de `P@V`

**Veredicto: `P@V` va con dot2/VOPD, no con WMMA.** Gana por las dos razones a
la vez: **1.57× más rápido** por trabajo útil y **83 VGPR frente a 153**.

El pairing se arregló: **0 % → 100 %** de `v_dual_dot2acc_f32_f16`. La causa no
era el número de acumuladores (la pista de F4 era un síntoma, no la causa): era
la **regla de bancos de VGPR del operando `src0`**, y `P@V` la viola por
construcción.

Workdir: `/scratch/rogarcia/f7-vopd/`.

---

## 1. Por qué no emparejaba (la causa real)

VOPD en RDNA3 exige que `srcX0` y `srcY0` de las dos mitades estén en **bancos
de VGPR distintos** (`banco = índice % 4`).

En `P@V` el operando `P` es la probabilidad de softmax **broadcast a
`__half2`**, y es **invariante en el bucle interno sobre las columnas de `V`**:

```
acc[j] = fdot2(p_bcast, v[j], acc[j])   // j = 0..NACC
```

Todas las `fdot2` leen **el mismo VGPR** como `src0`. El mismo registro está
trivialmente en el mismo banco, así que **ningún par de `dot2acc` puede
emparejarse jamás**, tenga el bucle 2 acumuladores o 64.

Eso es exactamente lo que se ve en el ASM que ya existía (`fleet/asm/out/`):

```
q3b_dot2_dv256_sp1_st1.s:165   v_dot2acc_f32_f16 v66, v96, v81
q3b_dot2_dv256_sp1_st1.s:171   v_dot2acc_f32_f16 v65, v96, v82
q3b_dot2_dv256_sp1_st1.s:172   v_dot2acc_f32_f16 v68, v96, v79
                                                     ^^^ src0 = v96 siempre
```

Los pocos `v_dual` que F2 contaba **no eran dos dot2 emparejadas**: eran
`v_dual_mul_f32 :: v_dual_dot2acc_f32_f16`, el rescale `acc[j] *= corr` del
softmax emparejado con un dot2. De ahí el 1.4 %.

**Por qué el sintético de F4 sí emparejaba**: `q1_builtin*.hip` lee `pp[2*i]` y
`pp[2*i+1]`, dos operandos **distintos** que caen en registros distintos. Lo que
F4 leyó como "hacen falta ≥2 acumuladores" era en realidad "hacen falta ≥2
`src0` en bancos distintos"; en su bucle las dos cosas venían juntas y no se
podían separar.

### La prueba directa de que es el banco

`bank.hip` fija las dos copias de `P` a VGPR físicos concretos con
`asm volatile("" : "+{v20}"(p0))`, sin cambiar nada más:

| copias de P | registros | banco (`idx%4`) | `v_dual_dot2acc` | sueltas | pairing |
|---|---|---|---:|---:|---:|
| `v20`, `v24` | distintos | **0 y 0 — iguales** | 0 | 8 | **0 %** |
| `v20`, `v21` | distintos | **0 y 1 — distintos** | **8** | 0 | **100 %** |

Mismo código, mismo número de acumuladores, misma presión de registros. Lo
único que cambia es el banco. **Registros distintos no bastan; tienen que ser
bancos distintos.**

## 2. El barrido de acumuladores que pedía el brief

`pv.hip`, forma exacta del bucle real (`acc[j] *= corr; acc[j] = fdot2(p, v[j], acc[j])`),
direccionamiento per-lane, `NPREP` = copias independientes de `P`:

| NACC | pairing con `NPREP=1` | pairing con `NPREP=2` | VGPR |
|---:|---:|---:|---:|
| 2 | 0 % | — | 12 |
| 4 | 25 % | — | 16 |
| 8 | 12 % | **87 %** | 25→26 |
| 16 | 37 % | **81 %** | 41→42 |
| 32 | 62 % | **78 %** | 73→74 |
| 64 | 39 % | **68 %** | 93 |

**No hay acantilado por número de acumuladores.** La hipótesis del brief
("demasiados acumuladores agotan los registros y rompen el pairing") queda
**refutada**: con 64 acumuladores y `NPREP=2` el pairing sube a 68 %, y no hay
ni un spill en toda la tabla (93 VGPR, holgado). El ruido de la columna
`NPREP=1` (25 %, 12 %, 37 %, 62 %, 39 %) es el pairer emparejando dot2 con los
`v_mul_f32` del rescale, no dot2 con dot2.

Quitando el rescale del softmax, que es lo que compite por los slots, el
resultado es binario y limpio:

| NACC | `NPREP=1` | `NPREP=2` |
|---:|---:|---:|
| 8 | **0 %** (0 duales / 8 sueltas) | **100 %** (8 / 0) |
| 16 | **0 %** (0 / 16) | **100 %** (16 / 0) |
| 32 | **0 %** (0 / 32) | **100 %** (32 / 0) |

`NPREP=4` no mejora sobre `NPREP=2` (y cuesta +2 VGPR): dos bancos bastan porque
el pairer solo empareja de dos en dos.

## 3. La receta

```cpp
__device__ __forceinline__ __half2 opaque(__half2 x) {
    asm volatile("" : "+v"(x));   // impide que el CSE vuelva a fundir las copias
    return x;
}

// dos copias independientes del operando P broadcast
__half2 p0 = opaque(__float2half2_rn(pe));
__half2 p1 = opaque(__float2half2_rn(pe));

#pragma unroll
for (int j = 0; j < NACC; ++j)
    acc[j] = __builtin_amdgcn_fdot2((j & 1) ? p1 : p0, v[j], acc[j], false);
```

Dos detalles que no son opcionales:

- **El `asm volatile("" : "+v"(x))` es obligatorio.** Sin él el compilador hace
  CSE de las dos copias y vuelve al caso de un solo registro: medido,
  `MODE=0 NPREP=2` da **12 %** de pairing y `MODE=1 NPREP=2` da **87 %**, con el
  mismo código fuente. El coste es **+1 VGPR**.
- **El alternado `j & 1` importa**, no basta con tener dos copias vivas: el
  pairer empareja instrucciones **adyacentes**, así que las dos mitades de cada
  par tienen que leer copias distintas.

Cuando el bucle ya tiene ≥4 operandos `P` distintos de verdad (p. ej.
`BLOCK_M=8`, una `P` por fila), **no hace falta nada**: el planificador
intercala filas y sale 100 % solo. Medido: `BLOCK_M=8, NPREP=1` → 512 duales,
0 sueltas. El problema es específico del **broadcast de una sola `P`**, que es
el caso `BLOCK_M=1`.

## 4. La medida de `P@V` (ahora que el pairing es 100 %)

`D_v=256`, `TILE=16`, `BLOCK_M=8` (`M×R=8`), wave32, `N_BUF=2`, 80 WG,
`N_OUTER=512`. Normalizado a **MACs útiles**: las tres rutas hacen 1024
MAC/lane/iteración de trabajo útil, así que el 2× de padding de WMMA (8 de 16
filas vacías) ya está dentro de su número.

| ruta | µs | **cyc/iter** | **vs WMMA** | **VGPR** | spill | pairing |
|---|---:|---:|---:|---:|---:|---:|
| WMMA | 121.8 | 640.9 | 1.00× | **153** | 0 | — |
| **dot2/VOPD** (`NPREP=1`) | 84.6 | 434.4 | **1.48×** | **75** | 0 | 100 % |
| **dot2/VOPD** (`NPREP=2`) | **78.4** | **407.8** | **1.57×** | **83** | 0 | 100 % |
| `v_pk_fma_f16` | 123.5 | 647.6 | 0.99× | **50** | 0 | — |

Tres corridas independientes bajo el lock; los µs coinciden en <2 % entre
corridas. La tabla es la pasada con telemetría aceptada (`accepted=True` en las
cuatro filas, sclk 2629–2694 MHz, spread 0.9–1.5 %).

Lecturas:

- **dot2/VOPD gana 1.57×** con la mitad de registros. El 1.57× ≈ el 2× teórico
  del padding de WMMA menos el límite del puerto SRC2 (F4: VOPD alcanza 73.8 %
  de su pico, WMMA 96 %): `2.0 × 0.738/0.96 = 1.54`. Concuerda.
- **`v_pk_fma_f16` no es competitivo**: empata con WMMA en ciclos aunque use 50
  VGPR. Hace 2 MAC/instrucción single-issue, igual que dot2 sin emparejar.
  Solo tendría sentido si el acumulador fp16 fuese aceptable numéricamente
  *y* los 33 VGPR de diferencia compraran una oleada más, que no es el caso.
- El `NPREP=2` del caso real (broadcast) cuesta **+8 VGPR sobre `NPREP=1`** y
  compra **+6 % de velocidad**. Con `BLOCK_M=8` sale 100 % sin `NPREP`, así que
  en nuestro kernel probablemente ni haga falta.

### Lo que de verdad decide: registros

Barrido de operandos residentes de `V` (`N_BUF`), compile-only:

| `N_BUF` | WMMA VGPR | WMMA spills | dot2 VGPR | dot2 spills |
|---:|---:|---:|---:|---:|
| 2 | 153 | 0 | **83** | 0 |
| 4 | 169 | 0 | **85** | 0 |
| 8 | 201 | 0 | **89** | 0 |
| 16 (V entero residente) | **249** | **24** | **96** | 0 |

WMMA crece **+6 VGPR por operando** y **spillea** al intentar tener `V`
residente; dot2 crece **+0.9 VGPR por operando** y no spillea nunca. Reproduce
lo que midieron F2 y F4 y lo extiende: no es un punto, es una pendiente 6.7×
peor.

## 5. Qué significa para el kernel

**Importa por los registros, no por el tiempo.** El kernel de decode es
**18–90× memory-bound**, así que ahorrar 200 ciclos de VALU por iteración de
`P@V` no va a mover el tiempo de pared. Lo que sí lo mueve es que **153 → 83
VGPR** permite subir la ocupación y el `NSTAGE` del prefetch, y eso sí ataca el
cuello real, que es esconder la latencia de DRAM. F4 ya lo había medido por el
otro lado: la ruta dot2 aguanta `NSTAGE=8` sin spill donde WMMA no pasa de 4.

El 1.57× de VALU es el bonus, no el argumento.

## 6. Correcciones a informes previos

- **F4 R1, "hacen falta ≥2 acumuladores independientes"**: es correcto como
  observación pero **la causa está mal atribuida**. La condición real es ≥2
  `src0` en **bancos de VGPR distintos**. En el bucle sintético de F4 ambas
  cosas coincidían. En `P@V` **se separan**, y por eso el consejo no transfirió:
  F2 siguió la receta al pie de la letra, tenía 64 acumuladores, y sacó 1.4 %.
- **F2, "con 64 acumuladores vivos el pairer casi nunca encuentra pareja bajo la
  restricción del puerto SRC2"**: el diagnóstico apuntaba al sitio correcto
  (una restricción de operandos) pero al mecanismo equivocado. No es presión de
  registros ni el puerto SRC2 — con 64 acumuladores y dos copias de `P` el
  pairing sube a 68 % sin un solo spill. F2 hizo bien en no medir.
- **`fleet/micro/pv_matmul/kernel.hip` (VARIANT=1) mide algo que no es el
  kernel.** Su `P` es uniforme por wave, así que hipcc la escalariza a SGPR:
  `v_dual_dot2acc_f32_f16 v66, s4, v1 :: v_dual_dot2acc_f32_f16 v65, s4, v2`.
  Un operando SGPR **no tiene banco**, así que empareja al 100 % gratis — pero
  en el kernel real cada lane tiene su propia probabilidad de softmax y la `P`
  es un VGPR. Mi `pvgemm.hip` la carga per-lane (`p[tid*BLOCK_M + m]`), que es
  lo que hace aparecer el problema. Quien reutilice ese fichero, ojo.

## 7. Ficheros

- `/scratch/rogarcia/f7-vopd/pv.hip` — sonda del bucle real; barrido NACC × NPREP × rescale.
- `/scratch/rogarcia/f7-vopd/bank.hip` — prueba aislada de la regla de bancos (`{v20}` vs `{v24}`/`{v21}`).
- `/scratch/rogarcia/f7-vopd/pvgemm.hip` — las tres rutas de `P@V`, `P` per-lane, nombres derivados de los parámetros de compilación.
- `/scratch/rogarcia/f7-vopd/harness.py` — medida (dos pasadas; la primera ramping deja la telemetría inválida).
- `/scratch/rogarcia/f7-vopd/sweep.sh` — compile-only, cuenta duales/sueltas y saca el % de pairing.
- `/scratch/rogarcia/f7-vopd/results_final.csv` — la tabla de §4.
- `/scratch/rogarcia/f7-vopd/out/*.s` — todo el ASM del barrido.

### Nota de método

Dos trampas del brief se confirmaron en vivo:

1. **Nombres derivados de los parámetros.** El `assert_describe` cazó un
   `--n-outer 512` contra `.so` compilados a 256 en vez de medir en silencio el
   binario viejo. La mitigación funciona; usadla.
2. **Compilar entre medidas invalida la telemetría.** Un `hipcc` entre dos
   ventanas deja caer el reloj a idle y la siguiente ventana se toma en plena
   rampa (stdev >300 MHz, el protocolo la rechaza). Hay que compilar todo antes
   y medir después — y aun así descartar la primera pasada.
