# F11 — las 50 formas embarcadas, y el bucle que las mide

`M=4`, fp16, HND, `block_size=16`, batch = 1 secuencia, **S=128**.
Roofline del protocolo: 230 GiB/s → `t_roof = 2·S·Hkv·D·2 / 230GiB/s`.

**Estado: COMPLETO** para la cobertura. **ABIERTO** para la optimización: el
bucle ya ha encontrado dos cosas que valen hasta 2× y aún no están arregladas.

> **Corrección.** Todo lo que sigue se mide a **S=128**, el único contexto que
> `roofline.py` recorre. "Ganamos en las 49" es cierto ahí y falso en el rango
> completo: el geomean sobre 7 contextos es **0.933×** y perdemos en 15 de 27
> configuraciones a partir de S=4096. La tabla completa está en f12 y el
> catálogo por configuración en HANDOFF.md §1. Medir un solo contexto y creerlo
> es el error; está anotado como trampa en HANDOFF.md §5.1.
>
> Este informe **sustituye las tablas de f10**. Aquellas se midieron antes de
> la reducción fusionada (`be0f2083d9`), que se llevó ~1.2 µs planos de cada
> celda, y antes de que el kernel sirviera más de un tamaño de cabeza.

## Resumen en seis líneas

1. **49 de 50 formas servidas**, frente a 10 antes. La que falta es D=96, fuera
   de alcance por decisión explícita.
2. **Ganamos en las 49**, de 1.11× a 2.79× contra el kernel que vLLM usa hoy.
   Mediana 1.34×, media 1.53×.
3. **La ganancia crece con D**: 1.13× a D=64, 1.33× a 128, 1.83× a 256, 2.35×
   a 512. Triton se degrada más rápido que nosotros según crece la fila.
4. **El objetivo de workgroups está mal** y vale hasta 2×. `_TARGET_WORKGROUPS`
   es una constante y el óptimo se mueve con D.
5. **La carga estrecha cuesta ~40% del ancho de banda marginal**: 58% del pico
   a b128 contra 35% a b64, misma forma, solo cambiando D.
6. **El % de roofline engaña como métrica de prioridad**: las peores formas
   están cerca de su techo alcanzable, no lejos.

## El bucle

```text
roofline.py  ->  peor forma  ->  ASM / wiki medido / primeros principios
     ^                                        |
     +------------  commit  <-  validar  <----+
```

`tools/roofline.py` es las dos mitades en una orden. Es la puerta de regresión,
porque comprueba las 50 formas contra una referencia en float y vigila los
contadores `kernel_calls`/`fallback_calls` del backend; y es el detector de
oportunidades, porque ordena por distancia al roofline.

Solo mide S=128, a propósito. El coste fijo es el mismo a cualquier contexto y
el presupuesto del roofline es mínimo ahí, así que es donde una forma está más
lejos del bus — y un contexto mantiene un pase de 50 formas en minutos, que es
lo que lo hace usable como puerta y no como informe.

La columna `ran` es el motivo de que exista. El backend cae a Triton en
silencio ante cualquier forma que no sabe servir, y un arnés que no lo mira
reporta Triton como si fuera nuestro; en este proyecto ha pasado dos veces.

## Tabla principal

Defaults de ambos backends. `vs` es Triton dividido por nosotros.

| modelo | Hq | Hkv | D | µs | %roof | Triton | vs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-0.5B-Instruct | 14 | 2 | 64 | 5.77 | 4.6 % | 6.52 | 1.13× |
| MiniCPM-V-0.53B-bosch | 16 | 2 | 64 | 5.79 | 4.6 % | 6.46 | 1.12× |
| geely-vlm-0320-ckpt14000 | 16 | 2 | 64 | 5.79 | 4.6 % | 6.49 | 1.12× |
| MiniCPM-V-custom-checkpoint-2400 | 32 | 2 | 128 | 7.55 | 7.0 % | 9.16 | 1.21× |
| Qwen2.5-3B-Instruct | 16 | 2 | 128 | 6.11 | 8.7 % | 8.22 | 1.35× |
| Qwen2.5-VL-3B-Instruct | 16 | 2 | 128 | 6.13 | 8.7 % | 8.23 | 1.34× |
| Qwen3-Omni-30B-A3B-talker | 16 | 2 | 128 | 6.12 | 8.7 % | 8.22 | 1.34× |
| gemma-2b-it | 8 | 1 | 256 | 5.62 | 9.5 % | 10.14 | 1.81× |
| gemma-4-12b-it | 16 | 1 | 512 | 9.90 | 10.7 % | 27.66 | 2.79× |
| Qwen2.5-7B-Instruct | 28 | 4 | 128 | 8.63 | 12.3 % | 9.79 | 1.13× |
| DeepSeek-R1-Distill-Qwen-7B | 28 | 4 | 128 | 8.64 | 12.3 % | 9.81 | 1.14× |
| MiniCPM-o-2_6 | 28 | 4 | 128 | 8.65 | 12.3 % | 9.82 | 1.14× |
| MiniCPM-V-2_6 | 28 | 4 | 128 | 8.65 | 12.3 % | 9.81 | 1.13× |
| Qwen2.5-VL-7B-Instruct | 28 | 4 | 128 | 8.65 | 12.3 % | 9.83 | 1.14× |
| Qwen3-30B-A3B-Instruct-2507 | 32 | 4 | 128 | 8.64 | 12.3 % | 11.22 | 1.30× |
| Qwen3-Omni-30B-A3B-thinker | 32 | 4 | 128 | 8.65 | 12.3 % | 11.22 | 1.30× |
| gemma-4-E2B-it | 8 | 1 | 512 | 8.16 | 13.0 % | 17.38 | 2.13× |
| Llama-3.2-1B-Instruct | 32 | 8 | 64 | 8.02 | 13.2 % | 9.65 | 1.20× |
| Qwen3.5-35B-A3B | 16 | 2 | 256 | 6.80 | 15.6 % | 12.43 | 1.83× |
| Qwen3.6-35B-A3B | 16 | 2 | 256 | 6.81 | 15.6 % | 12.43 | 1.83× |
| Qwen3.5-0.8B | 8 | 2 | 256 | 6.11 | 17.4 % | 11.77 | 1.93× |
| Qwen3.5-2B | 8 | 2 | 256 | 6.10 | 17.4 % | 11.79 | 1.93× |
| gemma-4-26B-A4B-it | 16 | 2 | 512 | 10.32 | 20.6 % | 24.15 | 2.34× |
| Qwen3.6-27B | 24 | 4 | 256 | 9.57 | 22.2 % | 17.13 | 1.79× |
| Ministral-3-8B-Instruct-2512 | 32 | 8 | 128 | 9.06 | 23.4 % | 12.02 | 1.33× |
| Qwen3-VL-4B-Instruct | 32 | 8 | 128 | 9.08 | 23.4 % | 12.01 | 1.32× |
| Cosmos-Reason2-8B-LLM | 32 | 8 | 128 | 9.06 | 23.4 % | 12.01 | 1.33× |
| Qwen3-4B | 32 | 8 | 128 | 9.04 | 23.5 % | 12.01 | 1.33× |
| Qwen3-8B | 32 | 8 | 128 | 9.05 | 23.5 % | 12.01 | 1.33× |
| Llama-3.1-8B-Instruct | 32 | 8 | 128 | 9.05 | 23.5 % | 12.00 | 1.33× |
| granite-4.1-8b | 32 | 8 | 128 | 9.05 | 23.5 % | 12.01 | 1.33× |
| Ministral-3-3B-Instruct-2512 | 32 | 8 | 128 | 9.05 | 23.5 % | 12.00 | 1.33× |
| NVIDIA-Nemotron-3-Nano-4B | 40 | 8 | 128 | 9.03 | 23.5 % | 11.04 | 1.22× |
| gemma-4-E4B-it | 8 | 2 | 512 | 8.98 | 23.6 % | 21.57 | 2.40× |
| Llama-3.2-3B-Instruct | 24 | 8 | 128 | 8.87 | 23.9 % | 9.85 | 1.11× |
| Phi-4-multimodal-instruct | 24 | 8 | 128 | 8.87 | 23.9 % | 9.86 | 1.11× |
| Qwen3.5-9B | 16 | 4 | 256 | 7.22 | 29.4 % | 15.23 | 2.11× |
| gemma-4-31B-it-assistant | 32 | 4 | 512 | 13.33 | 31.8 % | 31.36 | 2.35× |
| gemma-4-31B-it-AWQ | 32 | 4 | 512 | 13.31 | 31.9 % | 31.37 | 2.36× |
| Qwen3-Omni-30B-A3B-code-predictor | 16 | 8 | 128 | 6.61 | 32.1 % | 9.06 | 1.37× |
| Qwen3-1.7B | 16 | 8 | 128 | 6.64 | 32.0 % | 9.04 | 1.36× |
| Cosmos-Reason2-2B-LLM | 16 | 8 | 128 | 6.64 | 32.0 % | 9.03 | 1.36× |
| paligemma2-3b-mix-448-LLM | 8 | 4 | 256 | 6.43 | 33.0 % | 11.54 | 1.80× |
| gemma-3-4b-it | 8 | 4 | 256 | 6.42 | 33.1 % | 11.53 | 1.79× |
| deepseek-vl2-tiny | 10 | 10 | 128 | 7.15 | 37.1 % | 9.51 | 1.33× |
| SmolLM2-1.7B-Instruct | 32 | 32 | 64 | 8.88 | 47.8 % | 12.64 | 1.42× |
| gemma-3-12b-it | 16 | 8 | 256 | 8.82 | 48.1 % | 16.80 | 1.90× |
| Llama-2-7B | 32 | 32 | 128 | 13.11 | 64.8 % | 17.99 | 1.37× |
| cogagent-chat-hf | 32 | 32 | 128 | 13.10 | 64.8 % | 17.99 | 1.37× |
| **Phi-3.5-vision-instruct** | 32 | 32 | 96 | 14.95 | 42.6 % | 14.96 | **1.00×** |

La última fila no la servimos: D=96 son 3 fp16 por lane, que no es potencia de
dos y no tiene un ancho de carga único. Ahí *somos* Triton.

### Por tamaño de cabeza

| D | formas | speedup mediano | rango de %roofline | ancho de carga |
| --- | --- | --- | --- | --- |
| 64 | 5 | 1.13× | 4.6–47.8 % | b32 |
| 128 | 28 | 1.33× | 7.0–64.8 % | b64 |
| 256 | 10 | 1.83× | 9.5–48.1 % | b128 |
| 512 | 6 | 2.35× | 10.7–31.9 % | 2× b128 |

Que la ventaja crezca con D no es mérito nuestro subiendo: es Triton bajando.
Nuestro peor %roofline está en D=64 y el suyo también.

## Qué llevó de 10 formas a 49

`DPL` siempre fue `HEAD_DIM/32`; el `static_assert(DPL == 8)` era la única
razón de que solo existiera D=256. La generalización es sobre todo tipado: el
vector de carga pasa a `DPL/2` floats, el producto punto sigue ese límite, y el
epílogo estridea la salida por `BLOCK` en vez de asumir un hilo por elemento —
D=512 tiene más elementos que hilos y D=64 menos.

| D | instrucciones | LDS | VGPR | spills |
| --- | --- | --- | --- | --- |
| 64 | 1099 | 8448 | 66 | 0 |
| 128 | 1157 | 16640 | 82 | 0 |
| 256 | 1263 | 33024 | 114 | 0 |
| 512 | 1055 | 32896 | 132 | 0 |

D=512 destapó un bug latente que fue el primero en necesitar: la regla de
respaldo de `MSPLIT` leía `NWAVE` antes de que el preprocesador lo hubiera
visto. Un identificador desconocido vale 0 dentro de `#if`, en silencio, así
que todas las ramas comparaban `0 <= 65536` y siempre elegía `MSPLIT=1`. Nunca
había funcionado; lo tapaba que el loader pasa `-DMSPLIT` explícito.

## Abierto: dos cosas que el bucle ya encontró

### 1. El objetivo de workgroups es constante y no debería serlo

`_TARGET_WORKGROUPS = 32`. Midiendo `Hq=32 Hkv=8` a S=4096, variando solo el
número de workgroups:

| D | 32 WGs | 64 WGs | 128 WGs | 256 WGs |
| --- | --- | --- | --- | --- |
| 64 | 317.58 | 142.85 | **89.42** | 96.95 |
| 128 | 333.50 | 136.92 | **97.16** | 105.06 |
| 256 | 372.91 | 164.89 | **153.92** | 191.50 |
| 512 | 421.86 | **333.63** | 482.90 | 541.91 |

32 workgroups es el peor valor de las cuatro filas. Para `Hq=32` la heurística
elige NSEG=1 y deja entre 1.6× y 3.7× sin reclamar. El óptimo se mueve con D,
lo que sugiere que el invariante son bytes en vuelo y no workgroups.

**Pero no basta para escribir la regla.** `Hq=8/Hkv=4/D=256` y
`Hq=32/Hkv=8/D=128` tienen los mismos bytes por token (4096) y óptimos
distintos (64 contra 128 WGs), así que entra `Hq` o el GQA. Ajustar una fórmula
con los puntos disponibles sería repetir la regla del alineamiento con WGPs,
refutada ya tres veces. Falta un barrido por clase de forma.

### 2. La carga estrecha cuesta ~40% del ancho de banda marginal

Misma forma (`Hq=32 Hkv=8`), ajuste lineal sobre S, variando solo D:

| D | carga | fijo | marginal | % del pico |
| --- | --- | --- | --- | --- |
| 256 | b128 | ~3.3 µs | 0.0566 µs/token | 58 % |
| 128 | b64 | ~3.0 µs | 0.0473 µs/token | **35 %** |

La solución identificada es dividir la wave en grupos (`LPR = D/DPL` carriles
por fila), que mantiene b128 a cualquier D dando a cada grupo su propio token.
Es el trabajo estructural que queda.

## El % de roofline engaña como prioridad

Las tres peores formas de la tabla están al 4.6 %, y eso parece el sitio donde
atacar. No lo es. `Hq=14/Hkv=2/D=64` tiene un roofline de **0.27 µs** a S=128
contra un coste fijo de ~4 µs: aunque el marginal fuera exactamente cero, el
techo sería ~6 %. Quedan 1.7 puntos, no 95.

La métrica útil para priorizar es el exceso absoluto sobre el roofline, no la
fracción. Ordenar por fracción dirige el esfuerzo a las formas que menos
pueden mejorar.

## Reproducir

```bash
cd <worktree>
export PATH=<venv>/bin:$PATH PYTHONPATH=$PWD VLLM_KV_CACHE_LAYOUT=HND

# tabla principal, y puerta de regresión
amd-gpu-lock python benchmarks/kernels/gfx1151_decode_attn/tools/roofline.py --triton

# una clase de forma en detalle
amd-gpu-lock python benchmarks/kernels/gfx1151_decode_attn/tools/sweep.py \
    --hq 32 --hkv 8 --head-dim 128 --nseg 4 --msplit 4 --contexts 128 1024 4096
```
