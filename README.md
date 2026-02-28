# CryptoLab Engine v0.5 — Backtesting, Validación & Optimización

---

## 1. Visión General

**CryptoLab** es un framework de backtesting, validación anti-overfitting y optimización para **futuros perpetuos** en Bitget, soportando tanto **criptomonedas** como **acciones tokenizadas** (TSLA, NVDA, AAPL, etc.) bajo una única arquitectura unificada.

### Estrategias implementadas

| Estrategia | Señales | Estado |
|---|---|---|
| **CyberCycle v6.2** | buySignal/sellSignal con 5 métodos alpha + confidence scoring | ✅ |
| **Gaussian Bands** | long/short + TP multinivel (5 niveles) + trailing | ✅ |
| **Smart Money Concepts** | BOS/CHoCH + FVG + OB proximity oscillator + BSL/SSL targets | ✅ |

### Changelog v0.5

- Annualización dinámica por timeframe (antes hardcodeado a √365 = solo diario)
- Monte Carlo: comparación consistente observed vs simulated (fix crítico)
- DSR: Winsorización de kurtosis extrema por leverage alto
- CLI: `--params-file` para cargar/guardar parámetros JSON
- CLI: `--objective` (sharpe, return, calmar, composite)
- CLI: `--method` (grid, bayesian, genetic)
- CLI: `--targets` (conservative, aggressive, consistency)
- Optimizador Bayesiano (Optuna TPE) conectado al CLI
- Algoritmo Genético (DEAP) conectado al CLI
- Export automático de JSON en `optimize`

---

## 2. Estructura del Proyecto

```
cryptolab/
├── cli.py                             # CLI principal v0.5
│
├── indicators/                        # Indicadores traducidos de Pine
│   ├── ehlers.py                      # 4 métodos alpha + CyberCycle + iTrend + Fisher
│   ├── gaussian.py                    # Gaussian filter + multi-trend + TP levels
│   ├── structure.py                   # BOS/CHoCH/FVG/OB/BSL/SSL
│   └── common.py                      # EMA, SMA, ATR, crossover, pivots, etc.
│
├── strategies/                        # Estrategias completas
│   ├── base.py                        # IStrategy + Signal + ParamDef
│   ├── cybercycle.py                  # CyberCycle v6.2 (30 params)
│   ├── gaussbands.py                  # Gaussian Bands + TP multinivel (16 params)
│   └── smartmoney.py                  # SMC Oscillator L1/L2/L3 (23 params)
│
├── core/                              # Motor de backtesting
│   └── engine.py                      # BacktestEngine + TimeframeDetail + Order Mgmt
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
│   ├── grid_search.py                 # Grid Search con 4 objetivos
│   ├── bayesian.py                    # Optuna TPE + importancias + warm start
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
    └── params_*.json                  # Parámetros optimizados
```

---

## 3. Anti-Overfitting Pipeline (4 capas)

Cada capa ataca un tipo distinto de overfitting:

### Capa 1: Walk-Forward Analysis (WFA)
¿Los parámetros optimizados son estables en el tiempo? Divide datos en N ventanas IS/OOS, optimiza en IS, valida en OOS. **Umbral: WFE < 0.3 → RECHAZAR**

### Capa 2: Combinatorial Purged K-Fold CV
¿El rendimiento se mantiene en segmentos no vistos? K-Fold con purga temporal y embargo post-test. **Umbral: degradación > 40% → RECHAZAR**

### Capa 3: Deflated Sharpe Ratio (DSR)
¿El Sharpe es estadísticamente significativo? Corrige por múltiple testing (N trials), no-normalidad y tamaño de muestra. Kurtosis Winsorizada a 30 para leverage alto. **Umbral: DSR < 0.5 → RECHAZAR**

### Capa 4: Monte Carlo Permutation
¿Podría este resultado ocurrir por azar? Trade shuffle + block return shuffle + bootstrap CI. Observed y simulated se computan con la misma metodología. **Umbral: p-value > 0.05 → RECHAZAR**

---

## 4. Optimización (3 métodos × 4 objetivos)

### Métodos

| Método | Flag | Descripción |
|--------|------|-------------|
| Grid Search | `--method grid` | Exhaustivo, grid automático de 27 combos (default) |
| Bayesian (Optuna) | `--method bayesian` | TPE inteligente, param importances, warm start |
| Genetic (DEAP) | `--method genetic` | Evolución con crossover, mutación, elitismo |

### Objetivos

| Objetivo | Flag | Fórmula |
|----------|------|---------|
| Sharpe Ratio | `--objective sharpe` | Sharpe annualizado (default) |
| Return | `--objective return` | Total return % |
| Calmar | `--objective calmar` | Annual return / Max drawdown |
| Composite | `--objective composite` | SR(40%) + PF(20%) + Calmar(20%) + WR(20%) |

### JSON I/O

El optimize guarda automáticamente `output/params_{strategy}_{symbol}_{tf}.json`. Todos los comandos aceptan `--params-file` para cargar esos parámetros.

---

## 5. Uso Rápido

```bash
# Descargar datos
python cli.py download --symbol SOLUSDT --tf 1h --start 2024-01-01 --end 2025-06-01

# Backtest
python cli.py backtest --strategy cybercycle --symbol SOLUSDT --tf 1h --leverage 10

# Optimizar con bayesian buscando composite
python cli.py optimize --strategy cybercycle --symbol SOLUSDT --tf 1h \
  --method bayesian --objective composite --leverage 10 --n-trials 150

# Backtest con params optimizados
python cli.py backtest --strategy cybercycle --symbol SOLUSDT --tf 1h \
  --params-file output/params_cybercycle_SOLUSDT_1h.json --leverage 10

# Validar anti-overfitting
python cli.py validate --strategy cybercycle --symbol SOLUSDT --tf 1h \
  --params-file output/params_cybercycle_SOLUSDT_1h.json --leverage 10

# Régimen + targets agresivos
python cli.py regime --strategy cybercycle --symbol SOLUSDT --tf 1h --leverage 10
python cli.py targets --strategy cybercycle --symbol SOLUSDT --tf 1h \
  --targets aggressive --leverage 10
```

---

## 6. Métricas del Motor

El Sharpe y Sortino se annualizan dinámicamente según el timeframe: `√(bars_per_year)` donde `bars_per_year = 365.25 × 86400 / seconds_per_bar`. Esto asegura que las métricas son comparables entre timeframes (1m, 5m, 15m, 1h, 4h, 1d).

| Timeframe | Factor | bars/año |
|-----------|--------|----------|
| 1m | √525960 ≈ 725 | 525,960 |
| 5m | √105192 ≈ 324 | 105,192 |
| 15m | √35064 ≈ 187 | 35,064 |
| 1h | √8766 ≈ 94 | 8,766 |
| 4h | √2191 ≈ 47 | 2,191 |
| 1d | √365 ≈ 19 | 365 |

---

## 7. Stack Tecnológico

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

## 8. Referencias Académicas

- **Ehlers, J.F.** (2004) "Cybernetic Analysis for Stocks and Futures" — CyberCycle, iTrend, Fisher
- **Ehlers, J.F.** (2001) "MESA and Trading Market Cycles" — Homodyne, MAMA
- **Pardo, R.** (2008) "The Evaluation and Optimization of Trading Strategies" — WFA
- **López de Prado, M.** (2018) "Advances in Financial Machine Learning" — Purged K-Fold
- **Bailey, D.H. & López de Prado, M.** (2014) "The Deflated Sharpe Ratio" — DSR, PSR
- **White, H.** (2000) "A Reality Check for Data Snooping" — Monte Carlo permutation
- **Bergstra et al.** (2011) "Algorithms for Hyper-Parameter Optimization" — TPE (Optuna)
- **Goldberg, D.E.** (1989) "Genetic Algorithms in Search, Optimization and Machine Learning" — DEAP