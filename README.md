# CryptoLab Engine v0.6 — Backtesting, Validación & Optimización

---

## 1. Visión General

**CryptoLab** es un framework de backtesting, validación anti-overfitting y optimización para **futuros perpetuos** en Bitget, soportando tanto **criptomonedas** como **acciones tokenizadas** (TSLA, NVDA, AAPL, etc.) bajo una única arquitectura unificada.

### Estrategias implementadas

| Estrategia | Señales | Estado |
|---|---|---|
| **CyberCycle v6.2** | buySignal/sellSignal con 5 métodos alpha + confidence scoring | ✅ |
| **Gaussian Bands** | long/short + TP multinivel (5 niveles) + trailing | ✅ |
| **Smart Money Concepts** | BOS/CHoCH + FVG + OB proximity oscillator + BSL/SSL targets | ✅ |

### Changelog v0.6

- **Conditional parameter spaces**: Optuna solo optimiza sub-params relevantes al `alpha_method` seleccionado (35 → 22-27 params efectivos)
- **Monthly breakdown en backtest**: Desglose mes a mes automático al final del reporte
- **Top 10 con monthly**: Optimización muestra top 10 trials con desglose mensual de cada uno
- **`--trial N`**: Seleccionar cualquier trial del top 10 para backtest/validate/regime/targets
- **`--optimize-params`**: Optimizar solo un subconjunto de parámetros
- **`--objective monthly`**: Optimización por consistencia mensual (monthly Sharpe × % meses positivos)
- **`--objective monthly_robust`**: Como monthly pero con hard gates (WR≥40%, PF≥1.0, penalty leverage/DD)
- **`--objective weekly`**: Optimización por consistencia semanal
- **`--objective weekly_robust`**: Como weekly con hard gates
- **`max_signals_per_day`**: Nuevo parámetro para limitar señales diarias (anti-overtrading)
- **`close_on_signal`**: Reversal automático en señal contraria
- **Ctrl+C graceful**: Interrumpir optimización muestra resultados parciales + guarda JSON
- **Verbose mejorado**: Métricas completas por trial durante optimización

---

## 2. Estructura del Proyecto

```
cryptolab/
├── cli.py                             # CLI principal v0.6
│
├── indicators/                        # Indicadores traducidos de Pine
│   ├── ehlers.py                      # 4 métodos alpha + CyberCycle + iTrend + Fisher
│   ├── gaussian.py                    # Gaussian filter + multi-trend + TP levels
│   ├── structure.py                   # BOS/CHoCH/FVG/OB/BSL/SSL
│   └── common.py                      # EMA, SMA, ATR, crossover, pivots, etc.
│
├── strategies/                        # Estrategias completas
│   ├── base.py                        # IStrategy + Signal + ParamDef
│   ├── cybercycle.py                  # CyberCycle v6.2 (36 params)
│   ├── gaussbands.py                  # Gaussian Bands + TP multinivel (16 params)
│   └── smartmoney.py                  # SMC Oscillator L1/L2/L3 (23 params)
│
├── core/                              # Motor de backtesting
│   └── engine.py                      # BacktestEngine + TimeframeDetail + Daily Signal Limit
│
├── data/                              # Capa de datos
│   ├── bitget_client.py               # API Bitget + cache Parquet + MarketConfig
│   └── data_manager.py               # Gestión de datos con validación
│
├── optimize/                          # Anti-Overfitting + Optimización
│   ├── anti_overfit.py                # Pipeline integrado (4 capas secuenciales)
│   ├── walk_forward.py                # Walk-Forward Analysis (Pardo 2008)
│   ├── purged_kfold.py                # Combinatorial Purged CV (López de Prado 2018)
│   ├── deflated_sharpe.py             # Deflated Sharpe Ratio (Bailey & LdP 2014)
│   ├── monte_carlo.py                 # Permutation Test + Bootstrap CI
│   ├── grid_search.py                 # Grid Search + 8 objetivos + monthly/weekly stats
│   ├── bayesian.py                    # Optuna TPE + conditional spaces + importancias
│   └── genetic.py                     # DEAP: crossover, mutación, elitismo
│
├── ml/                                # ML y análisis avanzado
│   ├── regime_detector.py             # Detección de régimen (VT + HMM)
│   ├── ensemble.py                    # Ensemble multi-estrategia (4 métodos)
│   ├── temporal_targets.py            # Objetivos temporales (3 presets)
│   └── combinatorial.py              # Búsqueda combinatorial de portfolios
│
└── output/                            # Resultados exportados
    ├── trades_*.csv                   # Log de trades
    ├── equity_*.csv                   # Curvas de equity
    └── params_*.json                  # Parámetros optimizados (top 10 seleccionables)
```

---

## 3. Pipeline Recomendado

### Flujo completo de optimización → producción

```bash
# ══════════════════════════════════════════════════════════
# PASO 1: Descargar datos
# ══════════════════════════════════════════════════════════
python cli.py download --symbol SOLUSDT --tf 1h --start 2023-01-01 --end 2026-02-01

# ══════════════════════════════════════════════════════════
# PASO 2: Optimizar en periodo IN-SAMPLE (ej: 2023-2025)
# ══════════════════════════════════════════════════════════
python cli.py optimize --strategy cybercycle --symbol SOLUSDT --tf 1h \
  --method bayesian --objective monthly_robust --n-trials 150 \
  --start 2023-01-01 --end 2025-01-01

# ══════════════════════════════════════════════════════════
# PASO 3: Revisar los 10 trials y elegir uno
#         → El JSON muestra métricas + monthly de cada trial
# ══════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════
# PASO 4: Backtest OUT-OF-SAMPLE con el trial elegido
# ══════════════════════════════════════════════════════════
python cli.py backtest --strategy cybercycle --symbol SOLUSDT --tf 1h \
  --params-file output/params_cybercycle_SOLUSDT_1h.json --trial 4 \
  --start 2025-01-01 --end 2026-02-01

# ══════════════════════════════════════════════════════════
# PASO 5: Validación anti-overfitting sobre datos OOS
# ══════════════════════════════════════════════════════════
python cli.py validate --strategy cybercycle --symbol SOLUSDT --tf 1h \
  --params-file output/params_*.json --trial 4 \
  --start 2025-01-01 --end 2026-02-01

# ══════════════════════════════════════════════════════════
# PASO 6: Análisis de régimen y targets temporales (OOS)
# ══════════════════════════════════════════════════════════
python cli.py regime --params-file output/params_*.json --trial 4 \
  --start 2025-01-01 --end 2026-02-01

python cli.py targets --params-file output/params_*.json --trial 4 \
  --start 2025-01-01 --end 2026-02-01
```

### ¿Qué datos usar para cada paso?

| Paso | Datos | Razón |
|---|---|---|
| **optimize** | In-sample (2023-2025) | Buscar parámetros óptimos |
| **backtest** | Out-of-sample (2025-2026) | Ver performance real en datos no vistos |
| **validate** | Out-of-sample (2025-2026) | Confirmar robustez en datos no vistos |
| **regime** | Out-of-sample (2025-2026) | Entender en qué condiciones funciona OOS |
| **targets** | Out-of-sample (2025-2026) | Verificar consistencia temporal OOS |

---

## 4. Anti-Overfitting Pipeline (4 capas)

### Capa 1: Walk-Forward Analysis (WFA)
¿Los parámetros optimizados son estables en el tiempo? Divide datos en N ventanas IS/OOS, optimiza en IS, valida en OOS. **Umbral: WFE < 0.3 → RECHAZAR**

### Capa 2: Combinatorial Purged K-Fold CV
¿El rendimiento se mantiene en segmentos no vistos? K-Fold con purga temporal y embargo post-test. **Umbral: degradación > 40% → RECHAZAR**

### Capa 3: Deflated Sharpe Ratio (DSR)
¿El Sharpe es estadísticamente significativo? Corrige por múltiple testing (N trials), no-normalidad y tamaño de muestra. **Umbral: DSR < 0.5 → RECHAZAR**

### Capa 4: Monte Carlo Permutation
¿Podría este resultado ocurrir por azar? Trade shuffle + block return shuffle + bootstrap CI. **Umbral: p-value > 0.05 → RECHAZAR**

---

## 5. Optimización (3 métodos × 8 objetivos)

### Métodos

| Método | Flag | Descripción |
|--------|------|-------------|
| Grid Search | `--method grid` | Exhaustivo, grid automático de 27 combos (default) |
| Bayesian (Optuna) | `--method bayesian` | TPE con conditional spaces por alpha_method |
| Genetic (DEAP) | `--method genetic` | Evolución con crossover, mutación, elitismo |

### Objetivos

| Objetivo | Flag | Descripción |
|----------|------|-------------|
| Sharpe Ratio | `--objective sharpe` | Sharpe annualizado (default) |
| Return | `--objective return` | Total return % |
| Calmar | `--objective calmar` | Annual return / Max drawdown |
| Composite | `--objective composite` | SR(40%) + PF(20%) + Calmar(20%) + WR(20%) |
| Monthly | `--objective monthly` | Monthly Sharpe × % meses positivos × worst penalty |
| Monthly Robust | `--objective monthly_robust` | Monthly + WR≥40% + PF≥1.0 + leverage/DD penalty |
| Weekly | `--objective weekly` | Weekly Sharpe × % semanas positivas |
| Weekly Robust | `--objective weekly_robust` | Weekly + WR≥40% + PF≥1.0 + leverage/DD penalty |

### Conditional Parameter Spaces (Bayesian)

Cuando Optuna selecciona un `alpha_method`, solo optimiza los sub-parámetros relevantes:

| Método | Params activos | Total efectivo |
|---|---|---|
| `manual` | manual_alpha | 22 |
| `homodyne` | hd_min_period, hd_max_period, alpha_floor | 24 |
| `mama` | mama_fast, mama_slow, alpha_floor | 24 |
| `autocorrelation` | ac_min_period, ac_max_period, ac_avg_length, alpha_floor | 25 |
| `kalman` | kal_process_noise, kal_meas_noise, kal_alpha_fast, kal_alpha_slow, kal_sensitivity, alpha_floor | 27 |

### JSON I/O con Trial Selection

```bash
# Optimizar → guarda top 10 en JSON
python cli.py optimize --strategy cybercycle ...

# Cargar trial #1 (default)
python cli.py backtest --params-file output/params_*.json

# Cargar trial #4
python cli.py backtest --params-file output/params_*.json --trial 4

# Optimizar solo ciertos params
python cli.py optimize --optimize-params confidence_min,sl_atr_mult,tp1_rr,be_pct ...
```

---

## 6. Parámetros de Control de Señal

| Parámetro | Tipo | Default | Descripción |
|---|---|---|---|
| `min_bars` | int | 24 | Mínimo de barras entre señales (throttle) |
| `confidence_min` | float | 80.0 | Umbral de confianza para generar señal |
| `close_on_signal` | bool | True | Cerrar posición y abrir contraria en señal opuesta |
| `max_signals_per_day` | int | 0 | Máximo de señales por día (0 = sin límite) |

---

## 7. Métricas del Motor

El Sharpe y Sortino se annualizan dinámicamente según el timeframe: `√(bars_per_year)`.

| Timeframe | Factor | bars/año |
|-----------|--------|----------|
| 1m | √525960 ≈ 725 | 525,960 |
| 5m | √105192 ≈ 324 | 105,192 |
| 15m | √35064 ≈ 187 | 35,064 |
| 1h | √8766 ≈ 94 | 8,766 |
| 4h | √2191 ≈ 47 | 2,191 |
| 1d | √365 ≈ 19 | 365 |

---

## 8. Stack Tecnológico

| Componente | Tecnología |
|---|---|
| Motor core | Python + NumPy vectorizado |
| Indicadores | NumPy + loops para estado |
| Anti-overfitting | SciPy stats + NumPy Monte Carlo |
| Cache de datos | Parquet (pyarrow) |
| API de datos | aiohttp async |
| Optimización | Optuna TPE + DEAP genético + Grid Search |
| ML/Régimen | scikit-learn + hmmlearn |
| CLI | argparse nativo |

---

## 9. Referencias Académicas

- **Ehlers, J.F.** (2004) "Cybernetic Analysis for Stocks and Futures" — CyberCycle, iTrend, Fisher
- **Ehlers, J.F.** (2001) "MESA and Trading Market Cycles" — Homodyne, MAMA
- **Pardo, R.** (2008) "The Evaluation and Optimization of Trading Strategies" — WFA
- **López de Prado, M.** (2018) "Advances in Financial Machine Learning" — Purged K-Fold
- **Bailey, D.H. & López de Prado, M.** (2014) "The Deflated Sharpe Ratio" — DSR, PSR
- **White, H.** (2000) "A Reality Check for Data Snooping" — Monte Carlo permutation
- **Bergstra et al.** (2011) "Algorithms for Hyper-Parameter Optimization" — TPE (Optuna)
- **Goldberg, D.E.** (1989) "Genetic Algorithms in Search, Optimization and Machine Learning" — DEAP