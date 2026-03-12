#!/usr/bin/env python3
"""
CryptoLab — Full Pipeline Automation
Executes the complete optimize → backtest → validate → analyze cycle
for one or more symbols, evaluating all top trials.

Usage:
    python pipeline.py \
      --symbols ETHUSDT,BTCUSDT \
      --strategy cybercycle \
      --tf 1h \
      --capital 1000 \
      --leverage 3 \
      --opt-start 2024-06-01 --opt-end 2025-06-01 \
      --bt-start 2025-06-01 --bt-end 2026-03-08 \
      --method bayesian --objective monthly_robust \
      --n-trials 150 --n-jobs -1 \
      --top-n 10

    # Skip optimization (use existing params JSON):
    python pipeline.py \
      --symbols ETHUSDT \
      --strategy cybercycle --tf 1h \
      --bt-start 2025-06-01 --bt-end 2026-03-08 \
      --skip-optimize --top-n 5

Full options:
    --symbols         STR   Comma-separated symbols (required)
    --strategy        STR   cybercycle|gaussbands|smartmoney (default: cybercycle)
    --tf              STR   Timeframe (default: 1h)
    --capital         FLOAT Starting capital (default: 1000)
    --leverage        FLOAT Leverage (default: 3.0)
    --opt-start       DATE  Optimization (IS) start date
    --opt-end         DATE  Optimization (IS) end date
    --bt-start        DATE  Backtest (OOS) start date (required)
    --bt-end          DATE  Backtest (OOS) end date (required)
    --method          STR   grid|bayesian|genetic (default: bayesian)
    --objective       STR   Objective function (default: monthly_robust)
    --n-trials        INT   Bayesian/genetic trials (default: 150)
    --n-jobs          INT   Parallel workers (default: -1 = auto)
    --top-n           INT   Number of top trials to evaluate (default: 10)
    --no-intrabar           Disable intra-bar detail data
    --optimize-params STR   Subset of params to optimize (comma-sep)
    --skip-optimize         Skip optimization, use existing params JSON
    --targets         STR   conservative|aggressive|consistency (default: conservative)
    --mc-sims         INT   Monte Carlo simulations for validate (default: 500)
"""

import sys
import os
import json
import time
import copy
import traceback
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np


# ═══════════════════════════════════════════════════════════════
#  ARGUMENT PARSING
# ═══════════════════════════════════════════════════════════════

def parse_args():
    args = {
        'symbols': [],
        'strategy': 'cybercycle',
        'timeframe': '1h',
        'capital': 1000.0,
        'leverage': 3.0,
        'opt_start': None,
        'opt_end': None,
        'bt_start': None,
        'bt_end': None,
        'method': 'bayesian',
        'objective': 'monthly_robust',
        'n_trials': 150,
        'n_jobs': -1,
        'top_n': 10,
        'no_intrabar': False,
        'optimize_params': None,
        'skip_optimize': False,
        'targets': 'conservative',
        'mc_sims': 500,
    }

    i = 1
    argv = sys.argv
    while i < len(argv):
        arg = argv[i]
        if arg == '--symbols':
            args['symbols'] = [s.strip().upper() for s in argv[i+1].split(',')]
            i += 2
        elif arg == '--strategy':
            args['strategy'] = argv[i+1]; i += 2
        elif arg in ('--tf', '--timeframe'):
            args['timeframe'] = argv[i+1]; i += 2
        elif arg == '--capital':
            args['capital'] = float(argv[i+1]); i += 2
        elif arg == '--leverage':
            args['leverage'] = float(argv[i+1]); i += 2
        elif arg == '--opt-start':
            args['opt_start'] = argv[i+1]; i += 2
        elif arg == '--opt-end':
            args['opt_end'] = argv[i+1]; i += 2
        elif arg == '--bt-start':
            args['bt_start'] = argv[i+1]; i += 2
        elif arg == '--bt-end':
            args['bt_end'] = argv[i+1]; i += 2
        elif arg == '--method':
            args['method'] = argv[i+1]; i += 2
        elif arg == '--objective':
            args['objective'] = argv[i+1]; i += 2
        elif arg == '--n-trials':
            args['n_trials'] = int(argv[i+1]); i += 2
        elif arg == '--n-jobs':
            args['n_jobs'] = int(argv[i+1]); i += 2
        elif arg == '--top-n':
            args['top_n'] = int(argv[i+1]); i += 2
        elif arg == '--no-intrabar':
            args['no_intrabar'] = True; i += 1
        elif arg == '--optimize-params':
            args['optimize_params'] = argv[i+1]; i += 2
        elif arg == '--skip-optimize':
            args['skip_optimize'] = True; i += 1
        elif arg == '--targets':
            args['targets'] = argv[i+1]; i += 2
        elif arg == '--mc-sims':
            args['mc_sims'] = int(argv[i+1]); i += 2
        elif arg in ('-h', '--help'):
            print(__doc__); sys.exit(0)
        else:
            print(f"⚠️  Unknown arg: {arg}"); i += 1

    if not args['symbols']:
        print("❌ --symbols required (e.g. --symbols ETHUSDT,BTCUSDT)")
        sys.exit(1)
    if not args['skip_optimize'] and (not args['opt_start'] or not args['opt_end']):
        print("❌ --opt-start and --opt-end required (or use --skip-optimize)")
        sys.exit(1)
    if not args['bt_start'] or not args['bt_end']:
        print("❌ --bt-start and --bt-end required")
        sys.exit(1)

    # FIX: Validar formato YYYY-MM-DD antes de continuar.
    # Evita errores crípticos como '2025-010-01' (mes con 3 dígitos)
    # que solo se descubren profundo en la descarga de datos.
    def _validate_date(label, value):
        from datetime import datetime as _dt
        try:
            _dt.strptime(value, "%Y-%m-%d")
        except ValueError:
            print(f"❌ Fecha inválida en {label}: '{value}'")
            print(f"   Formato requerido: YYYY-MM-DD  (ej: 2025-10-01)")
            sys.exit(1)

    date_fields = [
        ('--bt-start',  args['bt_start']),
        ('--bt-end',    args['bt_end']),
    ]
    if not args['skip_optimize']:
        date_fields += [
            ('--opt-start', args['opt_start']),
            ('--opt-end',   args['opt_end']),
        ]
    for label, value in date_fields:
        if value:
            _validate_date(label, value)

    return args


# ═══════════════════════════════════════════════════════════════
#  PIPELINE CORE
# ═══════════════════════════════════════════════════════════════

def run_pipeline(args):
    from cli import (
        get_strategy, _load_data, _make_engine_factory,
        _load_params_file, _build_validation_grid,
        cmd_optimize,
    )
    from core.engine import format_result, result_to_dataframe
    from data.bitget_client import MarketConfig
    from optimize.grid_search import compute_monthly_stats

    symbols = args['symbols']
    strategy_name = args['strategy']
    tf = args['timeframe']
    capital = args['capital']
    leverage = args['leverage']
    top_n = args['top_n']

    # Output directory
    ts_tag = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_root = Path("output") / f"pipeline_{strategy_name}_{tf}_{ts_tag}"
    output_root.mkdir(parents=True, exist_ok=True)

    # Save pipeline config
    with open(output_root / "pipeline_config.json", 'w') as f:
        json.dump(args, f, indent=2, default=str)

    pipeline_t0 = time.time()
    all_results = {}

    _print_banner(args, output_root)

    for sym_idx, symbol in enumerate(symbols):
        sym_t0 = time.time()
        print(f"\n{'═' * 70}")
        print(f"  [{sym_idx+1}/{len(symbols)}] {symbol}")
        print(f"{'═' * 70}")

        sym_dir = output_root / symbol
        sym_dir.mkdir(exist_ok=True)

        sym_result = {
            'symbol': symbol,
            'strategy': strategy_name,
            'timeframe': tf,
            'capital': capital,
            'leverage': leverage,
            'optimization': None,
            'trials': [],
            'best_trial': None,
        }

        try:
            # ═══════════════════════════════════════════════
            # STEP 1: OPTIMIZATION (In-Sample)
            # ═══════════════════════════════════════════════
            params_file = Path("output") / f"params_{strategy_name}_{symbol}_{tf}.json"

            if not args['skip_optimize']:
                print(f"\n  📊 Step 1: Optimization ({args['opt_start']} → {args['opt_end']})")
                print(f"  {'─' * 60}")

                opt_args = {
                    'strategy': strategy_name,
                    'symbol': symbol,
                    'timeframe': tf,
                    'capital': capital,
                    'leverage': leverage,
                    'start': args['opt_start'],
                    'end': args['opt_end'],
                    'method': args['method'],
                    'objective': args['objective'],
                    'n_trials': args['n_trials'],
                    'n_jobs': args['n_jobs'],
                }
                if args['no_intrabar']:
                    opt_args['no_detail'] = True
                if args['optimize_params']:
                    opt_args['optimize_params'] = args['optimize_params']

                try:
                    cmd_optimize(opt_args)
                    sym_result['optimization'] = {
                        'status': 'OK',
                        'method': args['method'],
                        'objective': args['objective'],
                        'n_trials': args['n_trials'],
                    }
                except KeyboardInterrupt:
                    print("\n  ⚠️  Optimization interrupted — using partial results")
                    sym_result['optimization'] = {'status': 'INTERRUPTED'}
                except Exception as e:
                    print(f"  ❌ Optimization failed: {e}")
                    sym_result['optimization'] = {'status': 'FAILED', 'error': str(e)}
                    all_results[symbol] = sym_result
                    continue
            else:
                print(f"\n  ⏭️  Step 1: Skipped (--skip-optimize)")
                sym_result['optimization'] = {'status': 'SKIPPED'}

            # Load params JSON
            if not params_file.exists():
                print(f"  ❌ Params not found: {params_file}")
                all_results[symbol] = sym_result
                continue

            with open(params_file) as f:
                params_json = json.load(f)

            top_trials = params_json.get('top_trials', params_json.get('top_5', []))
            n_eval = min(top_n, len(top_trials))
            if n_eval == 0:
                print(f"  ❌ No trials in {params_file.name}")
                all_results[symbol] = sym_result
                continue

            print(f"\n  ✅ {len(top_trials)} trials available, evaluating top {n_eval}")

            # ═══════════════════════════════════════════════
            # STEP 2: LOAD OOS DATA (once for all trials)
            # ═══════════════════════════════════════════════
            print(f"\n  📡 Step 2: Loading OOS data ({args['bt_start']} → {args['bt_end']})")

            load_args = {
                'symbol': symbol,
                'timeframe': tf,
                'start': args['bt_start'],
                'end': args['bt_end'],
            }
            if args['no_intrabar']:
                load_args['no_detail'] = True

            data, detail_info, symbol_out, tf_out = _load_data(load_args, tf)
            market = MarketConfig.detect(symbol)
            no_intrabar = args.get('no_intrabar', False)
            engine_factory = _make_engine_factory(capital, detail_info, market,
                                                  no_intrabar=no_intrabar)

            # ═══════════════════════════════════════════════
            # STEP 3: EVALUATE EACH TRIAL (OOS)
            # ═══════════════════════════════════════════════
            print(f"\n  🔄 Step 3: Evaluating {n_eval} trials on OOS data")

            for trial_idx in range(1, n_eval + 1):
                trial_info = top_trials[trial_idx - 1]
                trial_params = trial_info.get('params', {})
                is_metrics = trial_info.get('metrics', trial_info)

                print(f"\n  {'━' * 66}")
                print(f"  Trial #{trial_idx}/{n_eval}  "
                      f"(IS: SR={is_metrics.get('sharpe',0):.2f} "
                      f"Ret={is_metrics.get('return',0):+.1f}% "
                      f"WR={is_metrics.get('win_rate',0):.1f}% "
                      f"DD={is_metrics.get('max_drawdown',0):.1f}%)")
                print(f"  {'━' * 66}")

                trial_result = _evaluate_single_trial(
                    trial_idx=trial_idx,
                    trial_params=trial_params,
                    is_metrics=is_metrics,
                    strategy_name=strategy_name,
                    leverage=leverage,
                    capital=capital,
                    data=data,
                    detail_info=detail_info,
                    engine_factory=engine_factory,
                    symbol=symbol,
                    tf=tf,
                    args=args,
                    sym_dir=sym_dir,
                )
                sym_result['trials'].append(trial_result)

            # ═══════════════════════════════════════════════
            # STEP 4: RANKING
            # ═══════════════════════════════════════════════
            sym_result['elapsed_seconds'] = round(time.time() - sym_t0, 1)
            sym_result['best_trial'] = _find_best_trial(sym_result['trials'])

            # Print mini-leaderboard
            _print_symbol_leaderboard(sym_result)

            # Save symbol JSON
            sym_json = sym_dir / f"results_{symbol}_{tf}.json"
            with open(sym_json, 'w') as f:
                json.dump(sym_result, f, indent=2, default=_json_default)
            print(f"\n  📁 Saved: {sym_json}")

            all_results[symbol] = sym_result

        except Exception as e:
            print(f"\n  ❌ Fatal error for {symbol}: {e}")
            traceback.print_exc()
            sym_result['error'] = str(e)
            all_results[symbol] = sym_result

    # ═══════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════
    total_elapsed = time.time() - pipeline_t0
    _print_global_summary(all_results, total_elapsed, output_root, args)

    return all_results


# ═══════════════════════════════════════════════════════════════
#  SINGLE TRIAL EVALUATION
# ═══════════════════════════════════════════════════════════════

def _evaluate_single_trial(trial_idx, trial_params, is_metrics,
                           strategy_name, leverage, capital,
                           data, detail_info, engine_factory,
                           symbol, tf, args, sym_dir):
    """
    Evaluate a single trial: backtest + conditional validate + regime + targets.

    Validation gate: validate/regime/targets only run if the backtest:
      1. Has positive return (total_return > 0)
      2. Has drawdown < return (max_drawdown < total_return)
      3. Has at least 10 trades (enough statistical significance)

    This saves ~30-60s per bad trial (validate is the most expensive step).
    """
    from cli import get_strategy, _build_validation_grid
    from core.engine import result_to_dataframe
    from optimize.grid_search import compute_monthly_stats

    trial_t0 = time.time()

    result = {
        'trial_rank': trial_idx,
        'params': trial_params,
        'in_sample': _sanitize_metrics(is_metrics),
        'backtest': None,
        'monthly': None,
        'validate': None,
        'regime': None,
        'targets': None,
        'skipped_reason': None,
    }

    # ── BACKTEST ──
    bt_result = None
    try:
        print(f"    🚀 Backtest...", end=" ", flush=True)
        strategy = get_strategy(strategy_name)
        strategy.set_params({'leverage': leverage})
        strategy.set_params(trial_params)

        engine = engine_factory()
        bt_result = engine.run(strategy, data, symbol, tf)

        bt = {
            'total_return': round(bt_result.total_return, 2),
            'annual_return': round(bt_result.annual_return, 2),
            'sharpe_ratio': round(bt_result.sharpe_ratio, 2),
            'sortino_ratio': round(bt_result.sortino_ratio, 2),
            'max_drawdown': round(bt_result.max_drawdown, 2),
            'calmar_ratio': round(bt_result.calmar_ratio, 2),
            'win_rate': round(bt_result.win_rate, 1),
            'profit_factor': round(bt_result.profit_factor, 2),
            'n_trades': bt_result.n_trades,
            'n_longs': bt_result.n_longs,
            'n_shorts': bt_result.n_shorts,
            'avg_win': round(bt_result.avg_win, 2),
            'avg_loss': round(bt_result.avg_loss, 2),
        }

        n_liq = sum(1 for t in bt_result.trades if t.exit_reason == 'liquidation')
        if n_liq > 0:
            bt['liquidations'] = n_liq

        result['backtest'] = bt

        print(f"SR={bt_result.sharpe_ratio:.2f} "
              f"Ret={bt_result.total_return:+.1f}% "
              f"WR={bt_result.win_rate:.1f}% "
              f"DD={bt_result.max_drawdown:.1f}% "
              f"T={bt_result.n_trades}"
              f"{f' ⚠️{n_liq}LIQ' if n_liq else ''}")

        # Monthly breakdown
        try:
            ms = compute_monthly_stats(bt_result.trades)
            if ms['n_months'] >= 2:
                result['monthly'] = {
                    'avg_return': round(ms['avg_monthly_return'], 2),
                    'pct_positive': round(ms['pct_positive'], 1),
                    'monthly_sharpe': round(ms['monthly_sharpe'], 2),
                    'worst_month': round(ms['worst_month'], 2),
                    'best_month': round(ms['best_month'], 2),
                    'n_months': ms['n_months'],
                    'months': ms['months'],
                }
        except Exception:
            pass

        # Save trades CSV
        trade_df = result_to_dataframe(bt_result)
        if len(trade_df) > 0:
            trade_df.to_csv(sym_dir / f"trades_trial{trial_idx}.csv", index=False)

        # ── VALIDATION GATE ──
        # Only run expensive steps if OOS backtest shows real promise
        ret    = bt_result.total_return
        dd     = bt_result.max_drawdown
        ntrades = bt_result.n_trades
        passes_gate = (ret > 0 and dd < ret and ntrades >= 10)

        if not passes_gate:
            result['skipped_reason'] = (
                f"gate: ret={ret:+.1f}% dd={dd:.1f}% t={ntrades}"
            )
        else:
            # ── VALIDATE (WFA + Monte Carlo) ──
            try:
                print(f"    🔬 Validate...", end=" ", flush=True)
                from optimize.anti_overfit import AntiOverfitPipeline

                strategy_val = get_strategy(strategy_name)
                strategy_val.set_params({'leverage': leverage})
                strategy_val.set_params(trial_params)

                param_grid = _build_validation_grid(strategy_val)

                pipeline = AntiOverfitPipeline(
                    wfa_windows=4,
                    mc_simulations=args.get('mc_sims', 500),
                    fail_fast=False,
                    verbose=False,
                )
                val_result = pipeline.run(
                    strategy_val, data, engine_factory,
                    param_grid, symbol, tf,
                )

                # Flatten result into serializable dict
                val = {
                    'all_passed':    getattr(val_result, 'all_passed', False),
                    'layers_passed': getattr(val_result, 'layers_passed', 0),
                    'wfa_passed':    getattr(val_result, 'wfa_passed', False),
                    'mc_passed':     getattr(val_result, 'mc_passed', False),
                    'sensitivity_passed': getattr(val_result, 'sensitivity_passed', False),
                    'overfit_passed':     getattr(val_result, 'overfit_passed', False),
                }
                result['validate'] = val
                passed = val['layers_passed']
                mark = "✅" if val['all_passed'] else "⚠️"
                print(f"{mark} {passed}/4 layers")

            except Exception as e:
                print(f"❌ {e}")
                result['validate'] = {'status': 'FAILED', 'error': str(e)}

            # ── REGIME ──
            try:
                print(f"    🔄 Regime...", end=" ", flush=True)
                from ml.regime_detector import detect_regime, strategy_regime_performance
                from data.bitget_client import MarketConfig

                strategy_reg = get_strategy(strategy_name)
                strategy_reg.set_params({'leverage': leverage})
                strategy_reg.set_params(trial_params)

                market = MarketConfig.detect(symbol)
                no_intrabar = args.get('no_intrabar', False)
                from cli import _make_engine_factory
                engine_factory_reg = _make_engine_factory(
                    capital, detail_info, market, no_intrabar=no_intrabar)

                rr   = detect_regime(data, method='vt', verbose=False)
                perf = strategy_regime_performance(
                    strategy_reg, data, engine_factory_reg, rr,
                    symbol=symbol, timeframe=tf, verbose=False)

                # Serialize regime result
                reg = {}
                if hasattr(perf, '__dict__'):
                    reg = {k: v for k, v in perf.__dict__.items()
                           if isinstance(v, (int, float, str, bool, list, dict))}
                elif isinstance(perf, dict):
                    reg = perf
                result['regime'] = reg
                print("✅")

            except Exception as e:
                print(f"❌ {e}")
                result['regime'] = {'status': 'FAILED', 'error': str(e)}

            # ── TARGETS ──
            try:
                print(f"    📅 Targets...", end=" ", flush=True)
                import numpy as np
                from ml.temporal_targets import (
                    evaluate_targets,
                    CONSERVATIVE_TARGETS, AGGRESSIVE_TARGETS, CONSISTENCY_TARGETS,
                )

                targets_map = {
                    'conservative': CONSERVATIVE_TARGETS,
                    'aggressive':   AGGRESSIVE_TARGETS,
                    'consistency':  CONSISTENCY_TARGETS,
                }
                target_preset = args.get('targets', 'conservative')
                targets = targets_map.get(target_preset, CONSERVATIVE_TARGETS)

                tt_result = evaluate_targets(
                    targets,
                    bt_result.trades,
                    data.get('timestamp', np.arange(len(data['close']))),
                    bt_result.equity_curve,
                    initial_capital=capital,
                    verbose=False,
                )

                tgt = {
                    'all_passed': getattr(tt_result, 'all_passed', False),
                    'n_passed':   getattr(tt_result, 'n_passed', 0),
                    'n_targets':  getattr(tt_result, 'n_targets', 0),
                }
                result['targets'] = tgt
                mark = "✅" if tgt['all_passed'] else "⚠️"
                print(f"{mark} {tgt['n_passed']}/{tgt['n_targets']} targets")

            except Exception as e:
                print(f"❌ {e}")
                result['targets'] = {'status': 'FAILED', 'error': str(e)}

    except Exception as e:
        print(f"❌ {e}")
        result['backtest'] = {'status': 'FAILED', 'error': str(e)}

    result['elapsed_seconds'] = round(time.time() - trial_t0, 1)
    return result


# ═══════════════════════════════════════════════════════════════
#  PRINTING HELPERS
# ═══════════════════════════════════════════════════════════════

def _print_banner(args, output_root):
    symbols_str = ', '.join(args['symbols'])
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║              CryptoLab — Full Pipeline                      ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Strategy:    {args['strategy']}")
    print(f"║  Symbols:     {symbols_str}")
    print(f"║  Timeframe:   {args['timeframe']}")
    print(f"║  Capital:     ${args['capital']:,.0f}  |  Leverage: {args['leverage']}x")
    if not args['skip_optimize']:
        print(f"║  Optimize:    {args['opt_start']} → {args['opt_end']}")
        print(f"║  Method:      {args['method']} / {args['objective']}")
        print(f"║  Trials:      {args['n_trials']}  |  Jobs: {args['n_jobs']}")
    print(f"║  Backtest:    {args['bt_start']} → {args['bt_end']}")
    print(f"║  Top trials:  {args['top_n']}")
    print(f"║  Intra-bar:   {'disabled' if args['no_intrabar'] else 'enabled'}")
    print(f"║  Output:      {output_root}")
    print("╚══════════════════════════════════════════════════════════════╝")


def _print_symbol_leaderboard(sym_result):
    trials = sym_result.get('trials', [])
    if not trials:
        return

    print(f"\n  {'─' * 66}")
    print(f"  {'#':>3} {'SR':>6} {'Ret%':>8} {'WR%':>6} {'DD%':>6} "
          f"{'PF':>5} {'T':>4} {'Val':>5} {'Tgt':>5}")
    print(f"  {'─' * 66}")

    for t in trials:
        bt = t.get('backtest', {})
        val = t.get('validate', {})
        tgt = t.get('targets', {})

        if not isinstance(bt, dict) or 'sharpe_ratio' not in bt:
            print(f"  #{t['trial_rank']:>2}  — failed —")
            continue

        val_str = f"{val.get('layers_passed','?')}/4" if isinstance(val, dict) and 'layers_passed' in val else '—'
        tgt_str = f"{tgt.get('n_passed','?')}/{tgt.get('n_targets','?')}" if isinstance(tgt, dict) and 'n_passed' in tgt else '—'

        best = sym_result.get('best_trial', {})
        marker = " ★" if t.get('trial_rank') == best.get('trial_rank') else ""

        print(f"  #{t['trial_rank']:>2} "
              f"{bt.get('sharpe_ratio',0):>5.2f} "
              f"{bt.get('total_return',0):>+7.1f}% "
              f"{bt.get('win_rate',0):>5.1f}% "
              f"{bt.get('max_drawdown',0):>5.1f}% "
              f"{bt.get('profit_factor',0):>4.2f} "
              f"{bt.get('n_trades',0):>4} "
              f"{val_str:>5} "
              f"{tgt_str:>5}{marker}")


def _print_global_summary(all_results, total_elapsed, output_root, args):
    summary = {
        'pipeline_version': '1.0',
        'timestamp': datetime.now().isoformat(),
        'config': args,
        'total_elapsed_seconds': round(total_elapsed, 1),
        'symbols': {},
    }

    print(f"\n\n{'═' * 70}")
    print(f"  PIPELINE SUMMARY — {len(all_results)} symbol(s) in {total_elapsed:.0f}s")
    print(f"{'═' * 70}")
    print(f"  {'Symbol':<12} {'Best#':>5} {'OOS SR':>7} {'OOS Ret':>8} {'WR%':>6} "
          f"{'DD%':>6} {'Val':>5} {'Tgt':>5}")
    print(f"  {'─' * 66}")

    for symbol, sr in all_results.items():
        best = sr.get('best_trial')
        if best and isinstance(best.get('backtest'), dict) and 'sharpe_ratio' in best['backtest']:
            bt = best['backtest']
            val = best.get('validate', {})
            tgt = best.get('targets', {})
            v_str = f"{val.get('layers_passed','?')}/4" if isinstance(val, dict) and 'layers_passed' in val else '—'
            t_str = f"{tgt.get('n_passed','?')}/{tgt.get('n_targets','?')}" if isinstance(tgt, dict) and 'n_passed' in tgt else '—'

            print(f"  {symbol:<12} #{best.get('trial_rank','?'):>4} "
                  f"{bt['sharpe_ratio']:>6.2f} "
                  f"{bt['total_return']:>+7.1f}% "
                  f"{bt['win_rate']:>5.1f}% "
                  f"{bt['max_drawdown']:>5.1f}% "
                  f"{v_str:>5} "
                  f"{t_str:>5}")

            summary['symbols'][symbol] = {
                'best_trial': best.get('trial_rank'),
                'oos_sharpe': bt['sharpe_ratio'],
                'oos_return': bt['total_return'],
                'oos_win_rate': bt['win_rate'],
                'oos_max_dd': bt['max_drawdown'],
                'validated': val.get('all_passed', False) if isinstance(val, dict) else False,
                'targets_passed': tgt.get('all_passed', False) if isinstance(tgt, dict) else False,
            }
        else:
            print(f"  {symbol:<12}  — no valid results —")
            summary['symbols'][symbol] = {'status': 'FAILED'}

    print(f"  {'─' * 66}")
    print(f"  Output: {output_root}")
    print(f"{'═' * 70}")

    summary_file = output_root / "pipeline_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=_json_default)
    print(f"\n📁 Summary: {summary_file}")


# ═══════════════════════════════════════════════════════════════
#  UTILITY
# ═══════════════════════════════════════════════════════════════

def _find_best_trial(trials):
    """Pick best trial: prefer validated + positive return, then highest OOS SR."""
    if not trials:
        return None

    # FIX: filtrar Nones que aparecen si _evaluate_single_trial no retornaba valor
    valid_trials = [t for t in trials if t is not None]
    if not valid_trials:
        return None

    def score(t):
        bt = t.get('backtest', {})
        val = t.get('validate', {})
        if not isinstance(bt, dict) or 'sharpe_ratio' not in bt:
            return (-999, -999)
        validated = 1 if (isinstance(val, dict) and val.get('all_passed')) else 0
        return (validated, bt['sharpe_ratio'])

    return max(valid_trials, key=score)


def _sanitize_metrics(m):
    safe = {}
    for k, v in m.items():
        if isinstance(v, (np.integer,)):
            safe[k] = int(v)
        elif isinstance(v, (np.floating,)):
            safe[k] = float(v)
        elif isinstance(v, (int, float, str, bool)):
            safe[k] = v
    return safe


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print(__doc__)
        sys.exit(0)
    args = parse_args()
    run_pipeline(args)