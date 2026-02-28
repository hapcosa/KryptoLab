# CryptoLab v0.5 — Manual de Referencia Completo

---

## Índice

1. Instalación y requisitos
2. Opciones globales del CLI
3. Datos: descargar, listar, validar
4. Backtest: ejecutar y analizar
5. Parámetros de estrategias
6. Validación anti-overfitting (4 capas)
7. Optimización (3 métodos × 4 objetivos)
   - 7.1 Grid Search
   - 7.2 Bayesian (Optuna TPE)
   - 7.3 Genetic (DEAP)
   - 7.4 JSON export/import de parámetros
8. Análisis avanzado
   - 8.1 Detección de régimen
   - 8.2 Ensemble multi-estrategia
   - 8.3 Objetivos temporales
   - 8.4 Búsqueda combinatorial de portfolios
9. Pipeline completo paso a paso
10. Interpretación de resultados
11. Referencia rápida de comandos

---

## 1. Instalación y Requisitos

### Dependencias core

```bash
pip install numpy pandas scipy aiohttp certifi pyarrow
pip install scikit-learn    # clustering de régimen
```

### Dependencias opcionales (optimización avanzada)

```bash
pip install optuna          # --method bayesian
pip install deap            # --method genetic
pip install hmmlearn        # régimen HMM (opcional)
```

Si no están instaladas, el CLI cae automáticamente a grid search con un mensaje.

### Estructura de carpetas

```
cryptolab/
├── cli.py                  ← Punto de entrada principal (v0.5)
├── core/
│   └── engine.py           ← Motor de backtesting
├── strategies/
│   ├── base.py             ← Clase base IStrategy + Signal
│   ├── cybercycle.py       ← CyberCycle v6.2
│   ├── gaussbands.py       ← Gaussian Bands
│   └── smartmoney.py       ← Smart Money Concepts
├── indicators/
│   ├── ehlers.py           ← Indicadores Ehlers (cybercycle, itrend, fisher)
│   ├── common.py           ← EMA, SMA, ATR, crossover, etc.
│   ├── gaussian.py         ← Filtros gaussianos
│   └── structure.py        ← BOS/CHoCH/FVG/OB (SMC)
├── data/
│   ├── bitget_client.py    ← Cliente API Bitget + cache Parquet
│   ├── data_manager.py     ← Gestión de datos con validación
│   └── cache/              ← Archivos .parquet descargados
├── optimize/
│   ├── anti_overfit.py     ← Pipeline de 4 capas
│   ├── walk_forward.py     ← Walk-Forward Analysis
│   ├── purged_kfold.py     ← Purged K-Fold CV
│   ├── deflated_sharpe.py  ← Deflated Sharpe Ratio
│   ├── monte_carlo.py      ← Monte Carlo Permutation + Bootstrap
│   ├── grid_search.py      ← Grid Search (4 objetivos)
│   ├── bayesian.py         ← Optuna TPE + importancias
│   └── genetic.py          ← DEAP genético + elitismo
├── ml/
│   ├── regime_detector.py  ← Detector de régimen de mercado
│   ├── ensemble.py         ← Ensemble multi-estrategia
│   ├── temporal_targets.py ← Objetivos temporales (3 presets)
│   └── combinatorial.py    ← Búsqueda combinatorial
└── output/                 ← Resultados exportados
    ├── trades_*.csv
    ├── equity_*.csv
    └── params_*.json       ← Parámetros optimizados (JSON)
```

---

## 2. Opciones Globales del CLI

Todas las opciones están disponibles para la mayoría de comandos:

```
--strategy    STR     Estrategia: cybercycle (cc), gaussbands (gb), smartmoney (smc)
--symbol      STR     Par de trading: BTCUSDT, ETHUSDT, SOLUSDT, etc.
--tf          STR     Timeframe: 1m, 5m, 15m, 1h, 4h, 1d
--leverage    FLOAT   Apalancamiento: 1.0 - 125.0
--capital     FLOAT   Capital inicial en USDT (default: 1000 backtest, 10000 validate)
--start       DATE    Fecha inicio: YYYY-MM-DD
--end         DATE    Fecha fin: YYYY-MM-DD
--sample               Usar datos sintéticos (no necesita API)
--batch       STR     Batch: crypto, stocks, all, o SYM1,SYM2,...
--detail-tf   STR     Override detail TF (e.g. 1m, 5m)
--no-detail            Desactivar carga de detail data
```

### Opciones nuevas en v0.5

```
--params-file PATH    Cargar parámetros desde archivo JSON
--objective   STR     Objetivo: sharpe, return, calmar, composite
--method      STR     Método de optimización: grid, bayesian, genetic
--targets     STR     Preset de targets: conservative, aggressive, consistency
--n-trials    INT     Número de trials para bayesian (default: 100)
```

**Importante:** `--leverage` debe pasarse siempre que se quiera un valor distinto de 3.0x. Si no se pasa, todos los comandos usan leverage=3.0 (el default de la estrategia). Esto aplica a `backtest`, `validate`, `optimize`, `regime` y `targets`.

---

## 3. DATOS: Descargar, Listar, Validar

### 3.1 Descargar datos de Bitget

El API de Bitget es público — no necesitas API key para datos de velas.

```bash
# Un solo par
python cli.py download --symbol BTCUSDT --tf 4h --start 2023-01-01 --end 2025-06-01
python cli.py download --symbol SOLUSDT --tf 1h --start 2024-01-01 --end 2025-06-01

# Batch
python cli.py download --batch crypto --tf 4h --start 2024-01-01 --end 2025-06-01
python cli.py download --batch stocks --tf 4h --start 2024-06-01 --end 2025-06-01
python cli.py download --batch BTCUSDT,ETHUSDT,SOLUSDT --tf 1h --start 2024-06-01 --end 2025-06-01
```

Los datos se guardan en `data/cache/{SYMBOL}_{TF}.parquet` y se reutilizan automáticamente.

### 3.2 Gestionar datos en caché

```bash
python cli.py data list                                    # Listar datasets
python cli.py data info --symbol BTCUSDT --tf 4h           # Info detallada
python cli.py data validate --symbol BTCUSDT --tf 4h       # Validar integridad
python cli.py data delete --symbol BTCUSDT --tf 4h         # Eliminar dataset
python cli.py data clear                                   # Eliminar todo
```

---

## 4. BACKTEST: Ejecutar y Analizar

### 4.1 Backtest básico

```bash
# Con datos descargados
python cli.py backtest --strategy cybercycle --symbol SOLUSDT --tf 1h \
  --start 2024-06-01 --end 2025-06-01 --leverage 10.0

# Con datos sintéticos
python cli.py backtest --strategy cybercycle --sample

# Con parámetros optimizados desde JSON
python cli.py backtest --strategy cybercycle --symbol SOLUSDT --tf 1h \
  --params-file output/params_cybercycle_SOLUSDT_1h.json --leverage 10.0
```

Cuando usas `--params-file`, los parámetros del JSON se aplican a la estrategia. Si además pasas `--leverage`, el leverage del CLI tiene prioridad sobre el del JSON (a menos que el JSON contenga `leverage` en `best_params`). El orden de aplicación es: defaults → leverage CLI → params-file → params extra.

### 4.2 Estrategias disponibles

```
cybercycle   (cc, cyber)     Ehlers Adaptive CyberCycle v6.2
gaussbands   (gb, gaussian)  Gaussian Bands con filtros
smartmoney   (smc, sm)       Smart Money Concepts (order blocks, FVG, etc.)
```

### 4.3 Archivos de salida

Cada backtest genera automáticamente:

```
output/trades_{strategy}_{symbol}_{tf}.csv    ← Log de cada trade
output/equity_{strategy}_{symbol}_{tf}.csv    ← Curva de equity + drawdown
```

### 4.4 Métricas del reporte

```
Total Return      Retorno total del período (%)
Annual Return     Retorno anualizado (%)
Sharpe Ratio      Retorno ajustado por riesgo (annualizado dinámicamente por TF)
Sortino Ratio     Como Sharpe pero solo penaliza volatilidad negativa
Max Drawdown      Mayor caída desde un pico (%)
Calmar Ratio      Annual Return / Max Drawdown
Win Rate          Porcentaje de trades ganadores
Profit Factor     Ganancias brutas / Pérdidas brutas
Avg Win / Loss    PnL promedio de ganadores / perdedores
Total Trades      Longs + Shorts
Avg Duration      Duración promedio en barras
```

**Annualización dinámica (v0.5):** El Sharpe y Sortino usan `√(bars_per_year)` donde `bars_per_year = 365.25 × 86400 / seconds_per_bar`. Esto hace que las métricas sean comparables entre timeframes:

| Timeframe | Factor | bars/año |
|-----------|--------|----------|
| 1m | 725.2 | 525,960 |
| 5m | 324.3 | 105,192 |
| 15m | 187.3 | 35,064 |
| 1h | 93.6 | 8,766 |
| 4h | 46.8 | 2,191 |
| 1d | 19.1 | 365 |

### 4.5 Cómo funciona el motor internamente

**Orden de ejecución por barra:**
1. Check exits (SL, TP1, TP2, trailing, liquidación)
2. Aplicar funding rate (cada 8h)
3. Generar señal nueva (si no hay posición)
4. Actualizar trailing stops
5. Registrar equity

**Prioridad de exits:** Liquidación > SL > Break-Even > TP1 > TP2 > Trailing

**Detail data:** Para TFs ≥ 1h, el motor carga automáticamente datos de 5m para simular exits intra-barra (SL/TP). Esto emula el comportamiento real donde un SL puede activarse a mitad de una vela de 4h.

---

## 5. PARÁMETROS DE ESTRATEGIAS

### 5.1 Listar parámetros

```bash
python cli.py params --strategy cybercycle
python cli.py params --strategy gaussbands
python cli.py params --strategy smartmoney
```

### 5.2 Parámetros de CyberCycle v6.2

**Señal:**

```
alpha_method       categorical   mama       [homodyne, mama, autocorrelation, kalman, manual]
manual_alpha       float         0.35       [0.05 → 0.80]
itrend_alpha       float         0.07       [0.01 → 0.30]
trigger_ema        int           14         [3 → 30]
min_bars           int           24         [5 → 50]
confidence_min     float         80.0       [30.0 → 95.0]
ob_level           float         1.5        [0.3 → 3.0]
os_level           float         -1.5       [-3.0 → -0.3]
```

**Filtros:**

```
use_trend          bool          True       Filtro de tendencia iTrend
use_volume         bool          True       Filtro de volumen
volume_mult        float         2.0        [0.5 → 5.0]  Multiplicador mínimo
use_htf            bool          False      Filtro HTF (proxy 4h)
```

**Risk Management:**

```
leverage           float         3.0        [1.0 → 25.0]
sl_atr_mult        float         2.0        [0.5 → 4.0]   SL = ATR × este valor
tp1_rr             float         1.8        [0.5 → 5.0]   TP1 en risk × este valor
tp1_size           float         0.6        [0.1 → 0.9]   Fracción cerrada en TP1
tp2_rr             float         3.0        [1.0 → 10.0]  TP2 en risk × este valor
be_pct             float         0.5        [0.0 → 5.0]   BE activa a este % de move
use_trailing       bool          True       Activar trailing stop
trail_activate_pct float         1.0        [0.0 → 10.0]  Trailing activa a este %
trail_pullback_pct float         0.5        [0.1 → 5.0]   Retroceso máximo del trailing
```

**Ejemplo numérico (LONG, entry=$150, ATR=$3):**

```
SL  = 150 - 3×2.0 = $144.00  (-4.0%)
TP1 = 150 + 6×1.8 = $160.80  (+7.2%) → cierra 60% de la posición
TP2 = 150 + 6×3.0 = $168.00  (+12.0%) → cierra 40% restante
BE  = 150 + 150×0.5% = $150.75 → SL se mueve a $150 (entry)
Trail activa a +1.0% ($151.50), retroceso máximo 0.5% ($0.75)
```

---

## 6. VALIDACIÓN ANTI-OVERFITTING (4 capas)

### 6.1 Ejecutar validación

```bash
# Con datos reales
python cli.py validate --strategy cybercycle --symbol SOLUSDT --tf 1h \
  --start 2024-06-01 --end 2025-06-01 --leverage 10.0

# Con parámetros optimizados
python cli.py validate --strategy cybercycle --symbol SOLUSDT --tf 1h \
  --params-file output/params_cybercycle_SOLUSDT_1h.json --leverage 10.0

# Con datos sintéticos
python cli.py validate --strategy cybercycle --sample
```

### 6.2 Las 4 capas de validación

**Layer 1 — Walk-Forward Analysis (WFA)**

Divide los datos en ventanas temporales. En cada ventana: optimiza parámetros In-Sample (IS), luego prueba Out-of-Sample (OOS). Mide si los parámetros optimizados funcionan en datos futuros.

```
Métrica:   WFE (Walk-Forward Efficiency) = Avg_OOS_Sharpe / Avg_IS_Sharpe
Threshold: WFE > 0.3  →  PASS
Falla si:  Los parámetros solo funcionan en el período donde se optimizaron
```

**Layer 2 — Purged K-Fold Cross-Validation**

Divide datos en K segmentos. Entrena en K-1, prueba en 1 (con gap de purga para evitar data leakage). Repite K veces.

```
Métrica:   Degradación = (Train_Sharpe - Test_Sharpe) / Train_Sharpe × 100
Threshold: Degradación < 40%  →  PASS
Falla si:  La estrategia solo funciona en datos de entrenamiento
```

**Layer 3 — Deflated Sharpe Ratio (DSR)**

Penaliza el Sharpe por el número de combinaciones probadas (n_trials). Si pruebas 27 combinaciones, por azar el mejor tendrá Sharpe alto — DSR descuenta eso. La kurtosis se Winsoriza a 30 para evitar que leverage alto infle artificialmente el error estándar. La annualización usa el factor correcto del timeframe.

```
Métrica:   DSR = P(SR_observado > E[max SR | n_trials])
Threshold: DSR > 0.5  →  PASS
Falla si:  El Sharpe es bajo relativo al número de trials
```

**Layer 4 — Monte Carlo Permutation Test**

Tres sub-tests. Tanto la métrica observada como las simuladas se computan con la misma metodología (trade PnLs → equity → returns → Sharpe), evitando comparaciones de métricas incompatibles.

```
Trade Shuffle:  Permuta el ORDEN de trades. Si shuffle ≥ real → timing es ruido
Return Shuffle: Permuta bloques de returns. Si shuffle ≥ real → señales no aportan
Bootstrap CI:   Resamplea trades para intervalos de confianza (90% y 98%)

Métrica:   p-value = fracción de permutaciones con Sharpe ≥ observado
Threshold: p-value < 0.05  →  PASS
```

### 6.3 Interpretar el veredicto

```
4/4 PASS   →  Estrategia robusta, lista para paper trading
3/4 PASS   →  Prometedora pero revisar la capa fallida
2/4 PASS   →  Necesita mejoras significativas
1/4 PASS   →  Sin edge demostrable
0/4 PASS   →  Completamente aleatorio o bugueado
```

El `Robustness Score` (0-100) pondera las 4 capas.

---

## 7. OPTIMIZACIÓN (3 métodos × 4 objetivos)

### 7.1 Grid Search (default)

Prueba todas las combinaciones de parámetros en un grid automático. El grid se construye desde `parameter_defs()` seleccionando los 3 parámetros más impactantes con 3 valores cada uno (min, default, max) = 27 combinaciones.

```bash
python cli.py optimize --strategy cybercycle --symbol SOLUSDT --tf 1h --leverage 10
python cli.py optimize --strategy cybercycle --symbol SOLUSDT --tf 1h \
  --objective composite --leverage 10
```

### 7.2 Bayesian (Optuna TPE)

Explora el espacio de parámetros de forma inteligente usando Tree-structured Parzen Estimators. Necesita menos evaluaciones que grid para encontrar buenos params. Reporta **importancias de parámetros** (qué param afecta más al resultado).

```bash
python cli.py optimize --strategy cybercycle --symbol SOLUSDT --tf 1h \
  --method bayesian --objective sharpe --leverage 10 --n-trials 150

# Con más trials para exploración más profunda
python cli.py optimize --strategy cybercycle --symbol SOLUSDT --tf 1h \
  --method bayesian --objective composite --n-trials 300
```

**Requiere:** `pip install optuna`. Si no está instalado, cae a grid automáticamente.

Las importancias de parámetros se guardan en el JSON exportado:

```json
{
  "param_importances": {
    "confidence_min": 0.42,
    "alpha_method": 0.31,
    "sl_atr_mult": 0.15,
    "leverage": 0.08,
    "tp1_rr": 0.04
  }
}
```

### 7.3 Genetic (DEAP)

Algoritmo genético con crossover BLX-α, mutación gaussiana y elitismo. Bueno para espacios de búsqueda grandes con muchos parámetros.

```bash
python cli.py optimize --strategy cybercycle --symbol SOLUSDT --tf 1h \
  --method genetic --objective calmar --leverage 10
```

**Requiere:** `pip install deap`. Si no está instalado, cae a grid automáticamente.

### 7.4 Objetivos disponibles

| Flag | Nombre | Qué maximiza |
|------|--------|-------------|
| `--objective sharpe` | Sharpe Ratio | Retorno ajustado por riesgo (default) |
| `--objective return` | Total Return | Retorno total bruto |
| `--objective calmar` | Calmar Ratio | Retorno vs drawdown |
| `--objective composite` | Composite | SR(40%) + PF(20%) + Calmar(20%) + WR(20%) |

**`composite` es el más equilibrado** — no maximiza un solo número sino el balance entre rendimiento, consistencia y riesgo.

### 7.5 JSON Export/Import de Parámetros

**Export automático:** Cada `optimize` guarda automáticamente un archivo JSON:

```
output/params_{strategy}_{symbol}_{tf}.json
```

**Contenido del JSON:**

```json
{
  "strategy": "cybercycle",
  "symbol": "SOLUSDT",
  "timeframe": "1h",
  "objective": "composite",
  "method": "bayesian",
  "best_params": {
    "alpha_method": "mama",
    "confidence_min": 72.0,
    "sl_atr_mult": 1.8
  },
  "metrics": {
    "sharpe": 1.45,
    "return": 18.3,
    "win_rate": 62.1,
    "max_drawdown": 4.8,
    "profit_factor": 1.65,
    "trades": 185
  },
  "top_5": [ ... ],
  "param_importances": { ... }
}
```

**Import en cualquier comando:**

```bash
# Backtest con params guardados
python cli.py backtest --strategy cybercycle --symbol SOLUSDT --tf 1h \
  --params-file output/params_cybercycle_SOLUSDT_1h.json

# Validar esos params
python cli.py validate --strategy cybercycle --symbol SOLUSDT --tf 1h \
  --params-file output/params_cybercycle_SOLUSDT_1h.json

# Ver régimen con esos params
python cli.py regime --strategy cybercycle --symbol SOLUSDT --tf 1h \
  --params-file output/params_cybercycle_SOLUSDT_1h.json

# Targets agresivos con esos params
python cli.py targets --strategy cybercycle --symbol SOLUSDT --tf 1h \
  --params-file output/params_cybercycle_SOLUSDT_1h.json --targets aggressive
```

El JSON acepta tres formatos de entrada:
- Nuestro formato de export (con `best_params` key)
- Formato simple con `params` key
- Dict directo de parámetros

---

## 8. ANÁLISIS AVANZADO

### 8.1 Detección de Régimen de Mercado

Clasifica cada barra en un régimen y muestra el rendimiento de la estrategia en cada uno.

```bash
python cli.py regime --strategy cybercycle --symbol SOLUSDT --tf 1h \
  --start 2024-06-01 --end 2025-06-01 --leverage 10.0

# Con params optimizados
python cli.py regime --strategy cybercycle --symbol SOLUSDT --tf 1h \
  --params-file output/params_cybercycle_SOLUSDT_1h.json --leverage 10.0
```

**Los 5 regímenes:**

```
TREND_UP       Tendencia alcista, volatilidad baja-media
TREND_DOWN     Tendencia bajista, volatilidad baja-media
RANGING        Consolidación lateral, baja volatilidad
HIGH_VOL_UP    Rally explosivo, alta volatilidad
HIGH_VOL_DOWN  Crash/pánico, alta volatilidad
```

Esto te dice inmediatamente en qué condiciones tu estrategia gana y pierde — la señal más útil para mejorar filtros.

### 8.2 Ensemble Multi-Estrategia

Ejecuta las 3 estrategias simultáneamente y combina sus señales.

```bash
python cli.py ensemble --symbol SOLUSDT --tf 1h \
  --start 2024-06-01 --end 2025-06-01
```

**Métodos de combinación (vía código):**

```python
from ml.ensemble import EnsembleBuilder

builder = EnsembleBuilder()
builder.add('CyberCycle', get_strategy('cybercycle'))
builder.add('GaussBands', get_strategy('gaussbands'))
builder.add('SmartMoney', get_strategy('smartmoney'))

result = builder.evaluate(data, engine_factory, method='confidence_vote')  # default CLI
result = builder.evaluate(data, engine_factory, method='static_blend')
result = builder.evaluate(data, engine_factory, method='regime_switch')
result = builder.evaluate(data, engine_factory, method='dynamic_blend')
```

### 8.3 Objetivos Temporales

Evalúa si la estrategia cumple objetivos de rendimiento consistentes a lo largo del tiempo — no solo en promedio.

```bash
# Targets conservadores (default)
python cli.py targets --strategy cybercycle --symbol SOLUSDT --tf 1h --leverage 10

# Targets agresivos
python cli.py targets --strategy cybercycle --symbol SOLUSDT --tf 1h \
  --targets aggressive --leverage 10

# Targets de consistencia
python cli.py targets --strategy cybercycle --symbol SOLUSDT --tf 1h \
  --targets consistency --leverage 10

# Con params optimizados
python cli.py targets --strategy cybercycle --symbol SOLUSDT --tf 1h \
  --params-file output/params_cybercycle_SOLUSDT_1h.json --targets aggressive
```

**Presets disponibles:**

| Preset | WR semanal | Profit mensual | Max DD diario | Trades |
|--------|-----------|----------------|---------------|--------|
| `conservative` | ≥50% en 75% semanas | ≥2% en 60% meses | <3% en 90% días | ≥2/semana en 70% |
| `aggressive` | ≥55% en 70% semanas | ≥5% en 60% meses | <5% en 85% días | Sharpe ≥0.5/mes en 60% |
| `consistency` | ≥45% en 85% semanas | ≥1% en 80% meses | <2% en 95% días | ≥3/semana en 80% |

**Formato personalizado vía código:**

```python
from ml.temporal_targets import evaluate_targets

custom_targets = [
    "win_rate:weekly:55:0.7",       # WR ≥ 55% en 70% de semanas
    "profit:monthly:3:0.6",         # +3% en 60% de meses
    "sharpe:monthly:0.5:0.5",       # Sharpe ≥ 0.5 en 50% de meses
    "max_drawdown:daily:-3:0.9",    # DD < 3% en 90% de días
    "n_trades:weekly:2:0.8",        # ≥ 2 trades en 80% de semanas
]

result = evaluate_targets(custom_targets, trades, timestamps, equity_curve)
```

### 8.4 Búsqueda Combinatorial de Portfolios

Prueba combinaciones de estrategias y encuentra el portfolio óptimo.

```bash
python cli.py portfolio --symbol SOLUSDT --tf 1h --start 2024-06-01 --end 2025-06-01
```

Internamente: crea 4 configs (CC-mama, CC-homodyne, GB-default, SMC-default), prueba todas las combinaciones de 2 y 3 estrategias, calcula correlación de returns, rechaza pares con correlación > 0.7, y ordena por Sharpe combinado. Un `synergy_score > 0` significa diversificación real.

---

## 9. PIPELINE COMPLETO PASO A PASO

### Paso 1: Descargar datos

```bash
python cli.py download --symbol SOLUSDT --tf 1h --start 2024-01-01 --end 2025-06-01
python cli.py data list
python cli.py data validate --symbol SOLUSDT --tf 1h
```

### Paso 2: Backtest exploratorio

```bash
python cli.py backtest --strategy cybercycle --symbol SOLUSDT --tf 1h \
  --start 2024-01-01 --end 2025-06-01 --leverage 10.0
```

¿Win Rate > 50%? ¿Sharpe > 0.5? ¿Profit Factor > 1.2?

### Paso 3: Analizar régimen

```bash
python cli.py regime --strategy cybercycle --symbol SOLUSDT --tf 1h --leverage 10.0
```

Identifica en qué regímenes tu estrategia gana y pierde.

### Paso 4: Optimizar parámetros

```bash
# Bayesian con objetivo composite para balance SR+PF+Calmar+WR
python cli.py optimize --strategy cybercycle --symbol SOLUSDT --tf 1h \
  --method bayesian --objective composite --leverage 10.0 --n-trials 150

# → Guarda output/params_cybercycle_SOLUSDT_1h.json automáticamente
```

### Paso 5: Backtest con params optimizados

```bash
python cli.py backtest --strategy cybercycle --symbol SOLUSDT --tf 1h \
  --params-file output/params_cybercycle_SOLUSDT_1h.json --leverage 10.0
```

Verifica que las métricas mejoraron respecto al paso 2.

### Paso 6: Validar anti-overfitting

```bash
python cli.py validate --strategy cybercycle --symbol SOLUSDT --tf 1h \
  --params-file output/params_cybercycle_SOLUSDT_1h.json --leverage 10.0
```

Necesitas ≥ 3/4 layers. Si DSR falla, es el más estricto. Si WFA o Monte Carlo fallan, la estrategia tiene problemas reales.

### Paso 7: Evaluar consistencia temporal

```bash
python cli.py targets --strategy cybercycle --symbol SOLUSDT --tf 1h \
  --params-file output/params_cybercycle_SOLUSDT_1h.json \
  --targets aggressive --leverage 10.0
```

¿Es rentable la mayoría de semanas/meses? ¿O depende de un solo período lucky?

### Paso 8: Test final con datos frescos

```bash
# Descargar datos que NO usaste en optimización
python cli.py download --symbol SOLUSDT --tf 1h --start 2025-06-01 --end 2025-12-01

# Backtest out-of-sample verdadero
python cli.py backtest --strategy cybercycle --symbol SOLUSDT --tf 1h \
  --params-file output/params_cybercycle_SOLUSDT_1h.json \
  --start 2025-06-01 --end 2025-12-01 --leverage 10.0
```

Si el rendimiento OOS es similar al IS → estrategia validada. Si colapsa → volver al paso 3.

---

## 10. INTERPRETACIÓN DE RESULTADOS

### 10.1 Sharpe Ratio

```
< 0.0    →  Pierde dinero ajustado por riesgo
0.0-0.5  →  Bajo, probablemente no es significativo
0.5-1.0  →  Aceptable para crypto (alta vol base)
1.0-2.0  →  Bueno
> 2.0    →  Excelente (o posible overfitting — validar)
```

### 10.2 Profit Factor

```
< 1.0    →  Pierde dinero (gross loss > gross profit)
1.0-1.5  →  Marginal (comisiones pueden eliminarlo)
1.5-2.0  →  Sólido
> 2.0    →  Excelente
```

### 10.3 Win Rate

```
< 40%    →  Necesita que los ganadores sean mucho mayores que perdedores
40-55%   →  Normal para sistemas de momentum/tendencia
55-70%   →  Bueno (típico con TPs escalonados y trailing)
> 70%    →  Verificar que no hay look-ahead bias
```

### 10.4 Monte Carlo CI

```
90% CI: [a, b]  →  En 90% de escenarios, el Sharpe estará entre a y b
Si a < 0         →  Hay riesgo real de pérdida
Si a > 0         →  La estrategia tiene un floor positivo
```

### 10.5 Exit Reasons

```
SL         →  Stop Loss tocado (pérdida controlada)
TP1        →  Take Profit 1 (cierre parcial)
TP2        →  Take Profit 2 (cierre total)
trailing   →  Trailing stop tocado
liquidation→  Margen insuficiente
end_of_data→  Cierre forzado al final del período
```

---

## 11. REFERENCIA RÁPIDA

```bash
# ── DATOS ──
python cli.py download --symbol BTCUSDT --tf 4h --start 2023-01-01 --end 2025-06-01
python cli.py download --batch crypto --tf 4h --start 2024-01-01 --end 2025-06-01
python cli.py data list
python cli.py data info --symbol BTCUSDT --tf 4h
python cli.py data validate --symbol BTCUSDT --tf 4h

# ── BACKTEST ──
python cli.py demo
python cli.py backtest --strategy cybercycle --symbol SOLUSDT --tf 1h --leverage 10
python cli.py backtest --strategy cybercycle --symbol SOLUSDT --tf 1h \
  --params-file output/params_cybercycle_SOLUSDT_1h.json --leverage 10
python cli.py params --strategy cybercycle

# ── VALIDACIÓN ──
python cli.py validate --strategy cybercycle --symbol SOLUSDT --tf 1h --leverage 10
python cli.py validate --strategy cybercycle --symbol SOLUSDT --tf 1h \
  --params-file output/params_cybercycle_SOLUSDT_1h.json --leverage 10

# ── OPTIMIZACIÓN ──
python cli.py optimize --strategy cybercycle --symbol SOLUSDT --tf 1h --leverage 10
python cli.py optimize --strategy cybercycle --symbol SOLUSDT --tf 1h \
  --method bayesian --objective composite --n-trials 150 --leverage 10
python cli.py optimize --strategy cybercycle --symbol SOLUSDT --tf 1h \
  --method genetic --objective calmar --leverage 10

# ── ANÁLISIS ──
python cli.py regime --strategy cybercycle --symbol SOLUSDT --tf 1h --leverage 10
python cli.py targets --strategy cybercycle --symbol SOLUSDT --tf 1h \
  --targets aggressive --leverage 10
python cli.py ensemble --symbol SOLUSDT --tf 1h
python cli.py portfolio --symbol SOLUSDT --tf 1h

# ── TODO CON SAMPLE (sin API) ──
python cli.py backtest --strategy cybercycle --sample
python cli.py optimize --strategy cybercycle --sample --method bayesian
python cli.py validate --strategy cybercycle --sample
python cli.py regime --strategy cybercycle --sample
python cli.py targets --strategy cybercycle --sample --targets aggressive
python cli.py ensemble --sample
python cli.py portfolio --sample
```