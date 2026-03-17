#!/usr/bin/env python3
"""
CryptoLab — Full Pipeline v2 (Dual Test + Quality Gates + Incremental Retry)
==============================================================================
Drop-in replacement for pipeline.py with:
  - TEST-2 verification backtest (--bt2-start/--bt2-end)
  - Quality gates: --min-validate, --min-win-rate
  - Incremental retry: --max-retries, --retry-trials-delta (same Optuna study)
  - Dynamic min_trades (--min-trades-per-week)

Usage:
    # Same as pipeline.py (100% backward compatible):
    python pipeline_v2.py \\
      --symbols ETHUSDT,BTCUSDT --strategy cybercycle --tf 1h \\
      --opt-start 2024-06-01 --opt-end 2025-06-01 \\
      --bt-start 2025-06-01 --bt-end 2025-12-01 \\
      --method bayesian --n-trials 150 --top-n 10

    # With T2 verification:
    python pipeline_v2.py \\
      --symbols ETHUSDT --strategy cybercycle --tf 1h \\
      --opt-start 2024-06-01 --opt-end 2025-06-01 \\
      --bt-start 2025-06-01 --bt-end 2025-12-01 \\
      --bt2-start 2025-12-01 --bt2-end 2026-03-01 \\
      --n-trials 150 --top-n 10

    # With quality gates + incremental retry:
    python pipeline_v2.py \\
      --symbols ETHUSDT --strategy cybercycle --tf 1h \\
      --opt-start 2024-06-01 --opt-end 2025-06-01 \\
      --bt-start 2025-06-01 --bt-end 2025-12-01 \\
      --bt2-start 2025-12-01 --bt2-end 2026-03-01 \\
      --n-trials 200 --top-n 10 \\
      --min-validate 2 --min-win-rate 40 \\
      --max-retries 3 --retry-trials-delta 100

Full options:
    --symbols         STR   Comma-separated symbols (required)
    --strategy        STR   cybercycle|gaussbands|smartmoney (default: cybercycle)
    --tf              STR   Timeframe (default: 1h)
    --capital         FLOAT Starting capital (default: 1000)
    --leverage        FLOAT Leverage (default: 3.0)
    --opt-start       DATE  Optimization (IS) start date
    --opt-end         DATE  Optimization (IS) end date
    --bt-start        DATE  Backtest T1 (OOS selection) start date (required)
    --bt-end          DATE  Backtest T1 end date (required)
    --bt2-start       DATE  Backtest T2 (verification) start date (optional)
    --bt2-end         DATE  Backtest T2 end date (optional)
    --method          STR   grid|bayesian|genetic (default: bayesian)
    --objective       STR   Objective function (default: monthly_robust)
    --n-trials        INT   Trials (default: 150)
    --n-jobs          INT   Parallel workers (default: -1)
    --top-n           INT   Top trials to evaluate (default: 10)
    --no-intrabar           Disable intra-bar detail data
    --optimize-params STR   Subset of params (comma-sep)
    --skip-optimize         Use existing params JSON
    --targets         STR   conservative|aggressive|consistency
    --mc-sims         INT   Monte Carlo sims (default: 500)
    --min-validate    INT   Min validation layers (0-4, default: 0)
    --min-win-rate    FLOAT Min win rate % (default: 0)
    --max-retries     INT   Retry rounds if gate fails (default: 0)
    --retry-trials-delta INT Extra trials per retry, same study (default: 100)
    --min-trades-per-week FLOAT Dynamic min trades (default: 1.5)
"""

import sys
import os
import json
import time
import traceback
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np


# ═══════════════════════════════════════════════════════════════
#  ARGS
# ═══════════════════════════════════════════════════════════════

def parse_args():
    args = {
        'symbols': [], 'strategy': 'cybercycle', 'timeframe': '1h',
        'capital': 1000.0, 'leverage': 3.0,
        'opt_start': None, 'opt_end': None,
        'bt_start': None, 'bt_end': None,
        'bt2_start': None, 'bt2_end': None,
        'method': 'bayesian', 'objective': 'monthly_robust',
        'n_trials': 150, 'n_jobs': -1, 'top_n': 10,
        'no_intrabar': False, 'optimize_params': None,
        'skip_optimize': False,
        'targets': 'conservative', 'mc_sims': 500,
        'min_validate': 0, 'min_win_rate': 0.0,
        'max_retries': 0, 'retry_trials_delta': 100,
        'min_trades_per_week': 1.5,
    }

    i = 1
    argv = sys.argv
    while i < len(argv):
        a = argv[i]
        if a == '--symbols':
            args['symbols'] = [s.strip().upper() for s in argv[i+1].split(',')]; i += 2
        elif a == '--strategy':      args['strategy'] = argv[i+1]; i += 2
        elif a in ('--tf', '--timeframe'): args['timeframe'] = argv[i+1]; i += 2
        elif a == '--capital':       args['capital'] = float(argv[i+1]); i += 2
        elif a == '--leverage':      args['leverage'] = float(argv[i+1]); i += 2
        elif a == '--opt-start':     args['opt_start'] = argv[i+1]; i += 2
        elif a == '--opt-end':       args['opt_end'] = argv[i+1]; i += 2
        elif a == '--bt-start':      args['bt_start'] = argv[i+1]; i += 2
        elif a == '--bt-end':        args['bt_end'] = argv[i+1]; i += 2
        elif a == '--bt2-start':     args['bt2_start'] = argv[i+1]; i += 2
        elif a == '--bt2-end':       args['bt2_end'] = argv[i+1]; i += 2
        elif a == '--method':        args['method'] = argv[i+1]; i += 2
        elif a == '--objective':     args['objective'] = argv[i+1]; i += 2
        elif a == '--n-trials':      args['n_trials'] = int(argv[i+1]); i += 2
        elif a == '--n-jobs':        args['n_jobs'] = int(argv[i+1]); i += 2
        elif a == '--top-n':         args['top_n'] = int(argv[i+1]); i += 2
        elif a == '--no-intrabar':   args['no_intrabar'] = True; i += 1
        elif a == '--optimize-params': args['optimize_params'] = argv[i+1]; i += 2
        elif a == '--skip-optimize': args['skip_optimize'] = True; i += 1
        elif a == '--targets':       args['targets'] = argv[i+1]; i += 2
        elif a == '--mc-sims':       args['mc_sims'] = int(argv[i+1]); i += 2
        elif a == '--min-validate':  args['min_validate'] = int(argv[i+1]); i += 2
        elif a == '--min-win-rate':  args['min_win_rate'] = float(argv[i+1]); i += 2
        elif a == '--max-retries':   args['max_retries'] = int(argv[i+1]); i += 2
        elif a == '--retry-trials-delta': args['retry_trials_delta'] = int(argv[i+1]); i += 2
        elif a == '--min-trades-per-week': args['min_trades_per_week'] = float(argv[i+1]); i += 2
        elif a in ('-h', '--help'):  print(__doc__); sys.exit(0)
        else:
            print(f"⚠️  Unknown: {a}"); i += 1

    if not args['symbols']:
        print("❌ --symbols required"); sys.exit(1)
    if not args['skip_optimize'] and (not args['opt_start'] or not args['opt_end']):
        print("❌ --opt-start and --opt-end required (or --skip-optimize)"); sys.exit(1)
    if not args['bt_start'] or not args['bt_end']:
        print("❌ --bt-start and --bt-end required"); sys.exit(1)
    if args['bt2_start'] and not args['bt2_end']:
        print("❌ --bt2-end required when --bt2-start is set"); sys.exit(1)

    def _vd(label, value):
        if value:
            try:
                datetime.strptime(value, "%Y-%m-%d")
            except ValueError:
                print(f"❌ Bad date {label}: '{value}'"); sys.exit(1)
    for l, v in [('--opt-start', args['opt_start']), ('--opt-end', args['opt_end']),
                 ('--bt-start', args['bt_start']), ('--bt-end', args['bt_end']),
                 ('--bt2-start', args['bt2_start']), ('--bt2-end', args['bt2_end'])]:
        _vd(l, v)

    return args


# ═══════════════════════════════════════════════════════════════
#  FIND BEST TRIAL (with quality gates)
# ═══════════════════════════════════════════════════════════════

def _find_best_trial(trials, min_validate_layers=0, min_win_rate=0.0):
    """Pick best trial with quality gate filters."""
    valid = [t for t in trials if t is not None]
    if not valid:
        return None

    def _layers(t):
        val = t.get('validate', {})
        return val.get('layers_passed', 0) if isinstance(val, dict) else 0

    def _wr(t):
        bt = t.get('backtest', {})
        return bt.get('win_rate', 0) if isinstance(bt, dict) else 0

    if min_validate_layers > 0 or min_win_rate > 0:
        qualified = [
            t for t in valid
            if isinstance(t.get('backtest'), dict)
            and 'sharpe_ratio' in t['backtest']
            and _layers(t) >= min_validate_layers
            and _wr(t) >= min_win_rate
        ]
        if not qualified:
            return None
        valid = qualified

    def score(t):
        bt = t.get('backtest', {})
        val = t.get('validate', {})
        if not isinstance(bt, dict) or 'sharpe_ratio' not in bt:
            return (-999, -999)
        v = 1 if (isinstance(val, dict) and val.get('all_passed')) else 0
        return (v, bt['sharpe_ratio'])

    return max(valid, key=score)


# ═══════════════════════════════════════════════════════════════
#  EVALUATE SINGLE TRIAL
# ═══════════════════════════════════════════════════════════════

def _evaluate_single_trial(trial_idx, trial_params, is_metrics,
                           strategy_name, leverage, capital,
                           data, detail_info, engine_factory,
                           symbol, tf, args, sym_dir,
                           min_trades=10):
    """
    Evaluate: backtest + conditional validate + regime + targets.
    Gate: ret>0, dd<ret, trades>=min_trades.
    FIX: prints explicit feedback when gate blocks validation.
    """
    from cli import get_strategy, _build_validation_grid
    from core.engine import result_to_dataframe
    from optimize.grid_search import compute_monthly_stats

    trial_t0 = time.time()

    result = {
        'trial_rank': trial_idx,
        'params': trial_params,
        'in_sample': _sanitize(is_metrics),
        'backtest': None, 'monthly': None,
        'validate': None, 'regime': None, 'targets': None,
        'skipped_reason': None,
    }

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

        # Monthly
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
        try:
            trade_df = result_to_dataframe(bt_result)
            if len(trade_df) > 0 and sym_dir:
                trade_df.to_csv(sym_dir / f"trades_trial{trial_idx}.csv",
                                index=False)
        except Exception:
            pass

        # ── Gate check (with explicit feedback) ──
        ret = bt_result.total_return
        dd = bt_result.max_drawdown
        nt = bt_result.n_trades

        gate_fails = []
        if ret <= 0:
            gate_fails.append(f"ret={ret:+.1f}%≤0")
        if dd >= ret:
            gate_fails.append(f"dd={dd:.1f}%≥ret={ret:+.1f}%")
        if nt < min_trades:
            gate_fails.append(f"trades={nt}<{min_trades}")

        if gate_fails:
            reason = ", ".join(gate_fails)
            result['skipped_reason'] = reason
            print(f"    ⏭️  Skip validation ({reason})")
        else:
            # ── VALIDATE ──
            try:
                print(f"    🔬 Validate...", end=" ", flush=True)
                from optimize.anti_overfit import AntiOverfitPipeline
                sv = get_strategy(strategy_name)
                sv.set_params({'leverage': leverage})
                sv.set_params(trial_params)
                pg = _build_validation_grid(sv)
                vp = AntiOverfitPipeline(
                    wfa_windows=4,
                    mc_simulations=args.get('mc_sims', 500),
                    fail_fast=False, verbose=False)
                vr = vp.run(sv, data, engine_factory, pg, symbol, tf)
                val = {
                    'all_passed': getattr(vr, 'all_passed', False),
                    'layers_passed': getattr(vr, 'layers_passed', 0),
                    'wfa_passed': getattr(vr, 'wfa_passed', False),
                    'mc_passed': getattr(vr, 'mc_passed', False),
                    'sensitivity_passed': getattr(vr, 'sensitivity_passed', False),
                    'overfit_passed': getattr(vr, 'overfit_passed', False),
                }
                result['validate'] = val
                m = "✅" if val['all_passed'] else "⚠️"
                print(f"{m} {val['layers_passed']}/4 layers")
            except Exception as e:
                print(f"❌ {e}")
                result['validate'] = {'status': 'FAILED', 'error': str(e)}

            # ── REGIME ──
            try:
                print(f"    🔄 Regime...", end=" ", flush=True)
                from ml.regime_detector import detect_regime, strategy_regime_performance
                sr = get_strategy(strategy_name)
                sr.set_params({'leverage': leverage})
                sr.set_params(trial_params)
                _ita = sr.get_param('itrend_alpha', 0.07)
                rr = detect_regime(data, method='vt', verbose=False,
                                   itrend_alpha=_ita)
                perf = strategy_regime_performance(
                    sr, data, engine_factory, rr,
                    symbol=symbol, timeframe=tf, verbose=False)
                reg = {str(k): v for k, v in perf.items()
                       if isinstance(v, dict)} if isinstance(perf, dict) else {}
                result['regime'] = reg
                print("✅")
            except Exception as e:
                print(f"❌ {e}")
                result['regime'] = {'status': 'FAILED', 'error': str(e)}

            # ── TARGETS ──
            try:
                print(f"    📅 Targets...", end=" ", flush=True)
                from ml.temporal_targets import (
                    evaluate_targets,
                    CONSERVATIVE_TARGETS, AGGRESSIVE_TARGETS, CONSISTENCY_TARGETS)
                tmap = {
                    'conservative': CONSERVATIVE_TARGETS,
                    'aggressive': AGGRESSIVE_TARGETS,
                    'consistency': CONSISTENCY_TARGETS}
                tspecs = tmap.get(args.get('targets', 'conservative'),
                                  CONSERVATIVE_TARGETS)
                tt = evaluate_targets(
                    tspecs, bt_result.trades,
                    data.get('timestamp', np.arange(len(data['close']))),
                    bt_result.equity_curve,
                    initial_capital=capital, verbose=False)
                tgt = {
                    'all_passed': getattr(tt, 'all_passed', False),
                    'n_passed': getattr(tt, 'n_passed', 0),
                    'n_targets': getattr(tt, 'n_targets', 0)}
                result['targets'] = tgt
                m = "✅" if tgt['all_passed'] else "⚠️"
                print(f"{m} {tgt['n_passed']}/{tgt['n_targets']} targets")
            except Exception as e:
                print(f"❌ {e}")
                result['targets'] = {'status': 'FAILED', 'error': str(e)}

    except Exception as e:
        print(f"❌ {e}")
        result['backtest'] = {'status': 'FAILED', 'error': str(e)}

    result['elapsed_seconds'] = round(time.time() - trial_t0, 1)
    return result


# ═══════════════════════════════════════════════════════════════
#  PIPELINE CORE
# ═══════════════════════════════════════════════════════════════

def run_pipeline(args):
    from cli import (
        get_strategy, _load_data, _make_engine_factory,
        _load_params_file, _build_validation_grid, cmd_optimize,
    )
    from core.engine import format_result, result_to_dataframe
    from data.bitget_client import MarketConfig

    symbols = args['symbols']
    strategy_name = args['strategy']
    tf = args['timeframe']
    capital = args['capital']
    leverage = args['leverage']
    top_n = args['top_n']
    has_t2 = bool(args.get('bt2_start') and args.get('bt2_end'))
    has_gate = (args['min_validate'] > 0 or args['min_win_rate'] > 0)
    # When retries are enabled: use _optimize_increment so Optuna study stays alive
    use_incremental = (args['max_retries'] > 0 and not args['skip_optimize'])

    ts_tag = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_root = Path("output") / f"pipeline_{strategy_name}_{tf}_{ts_tag}"
    output_root.mkdir(parents=True, exist_ok=True)

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
            'symbol': symbol, 'strategy': strategy_name,
            'timeframe': tf, 'capital': capital, 'leverage': leverage,
            'optimization': None, 'trials': [], 'best_trial': None,
            't2_result': None,
        }

        try:
            params_file = Path("output") / f"params_{strategy_name}_{symbol}_{tf}.json"

            # ═══════════════════════════════════════════════
            # STEP 1: OPTIMIZATION (In-Sample)
            # ═══════════════════════════════════════════════
            is_data = None
            is_detail = None
            opt_state = None  # Kept alive for retries

            if not args['skip_optimize']:
                print(f"\n  📊 Step 1: Optimization "
                      f"({args['opt_start']} → {args['opt_end']})")

                if use_incremental:
                    # ── INCREMENTAL MODE ──
                    # Use _optimize_increment so the Optuna study stays alive.
                    # If gate fails later, retries CONTINUE the same study
                    # from where it left off (e.g. 200 → 300 → 400).
                    from optimize.rolling_wfo import (
                        _OptimizerState, _optimize_increment)

                    is_load = {
                        'symbol': symbol, 'timeframe': tf,
                        'start': args['opt_start'], 'end': args['opt_end']}

                    print(f"  📡 Loading IS data...")
                    is_data, is_detail, _, _ = _load_data(is_load, tf)
                    print(f"     {len(is_data['close'])} bars loaded")

                    opt_state = _OptimizerState()
                    param_subset = None
                    if args['optimize_params']:
                        param_subset = [p.strip()
                                        for p in args['optimize_params'].split(',')]

                    print(f"  🔧 Optimizing ({args['method']}, "
                          f"{args['n_trials']} trials, "
                          f"obj={args['objective']})...")

                    top_trials_raw = _optimize_increment(
                        opt_state, strategy_name,
                        is_data, is_detail, capital, leverage,
                        args['method'], args['objective'],
                        args['n_trials'], args['n_jobs'],
                        symbol, tf, args.get('no_intrabar', False),
                        param_subset, top_n, verbose=True)

                    sym_result['optimization'] = {
                        'status': 'OK', 'method': args['method'],
                        'n_trials': opt_state.total_trials}

                    # Save params JSON (compatible with cmd_optimize format)
                    _save_params_json(
                        params_file, strategy_name, symbol, tf,
                        args['objective'], args['method'], top_trials_raw)

                else:
                    # ── STANDARD MODE (no retries, use cmd_optimize) ──
                    opt_args = {
                        'strategy': strategy_name, 'symbol': symbol,
                        'timeframe': tf, 'capital': capital,
                        'leverage': leverage,
                        'start': args['opt_start'], 'end': args['opt_end'],
                        'method': args['method'],
                        'objective': args['objective'],
                        'n_trials': args['n_trials'],
                        'n_jobs': args['n_jobs'],
                    }
                    if args['no_intrabar']:
                        opt_args['no_intrabar'] = True
                    if args['optimize_params']:
                        opt_args['optimize_params'] = args['optimize_params']

                    try:
                        cmd_optimize(opt_args)
                        sym_result['optimization'] = {
                            'status': 'OK', 'method': args['method'],
                            'n_trials': args['n_trials']}
                    except KeyboardInterrupt:
                        print("\n  ⚠️ Interrupted — using partial")
                        sym_result['optimization'] = {'status': 'INTERRUPTED'}
                    except Exception as e:
                        print(f"  ❌ Optimization failed: {e}")
                        sym_result['optimization'] = {
                            'status': 'FAILED', 'error': str(e)}
                        all_results[symbol] = sym_result
                        continue
            else:
                sym_result['optimization'] = {'status': 'SKIPPED'}

            # Load params JSON
            if not params_file.exists():
                print(f"  ❌ Params not found: {params_file}")
                all_results[symbol] = sym_result
                continue

            with open(params_file) as f:
                params_json = json.load(f)

            top_trials_json = params_json.get('top_trials',
                                              params_json.get('top_5', []))
            n_eval = min(top_n, len(top_trials_json))
            if n_eval == 0:
                print(f"  ❌ No trials in {params_file.name}")
                all_results[symbol] = sym_result
                continue

            print(f"\n  ✅ {len(top_trials_json)} trials available, "
                  f"evaluating top {n_eval}")

            # ═══════════════════════════════════════════════
            # STEP 2: LOAD T1 DATA
            # ═══════════════════════════════════════════════
            print(f"\n  📡 Step 2: Loading T1 data "
                  f"({args['bt_start']} → {args['bt_end']})")

            load_args = {
                'symbol': symbol, 'timeframe': tf,
                'start': args['bt_start'], 'end': args['bt_end']}

            t1_data, t1_detail, _, _ = _load_data(load_args, tf)
            market = MarketConfig.detect(symbol)
            no_intrabar = args.get('no_intrabar', False)
            t1_ef = _make_engine_factory(
                capital, t1_detail, market, no_intrabar=no_intrabar)

            # Dynamic min trades
            bt_start_dt = datetime.strptime(args['bt_start'], "%Y-%m-%d")
            bt_end_dt = datetime.strptime(args['bt_end'], "%Y-%m-%d")
            bt_days = (bt_end_dt - bt_start_dt).days
            bt_weeks = max(1.0, bt_days / 7.0)
            dyn_min_trades = max(2, int(bt_weeks * args['min_trades_per_week']))
            print(f"     {len(t1_data['close'])} bars loaded")
            print(f"     Min trades: {dyn_min_trades} "
                  f"({args['min_trades_per_week']}/week × {bt_weeks:.1f} weeks)")

            # ═══════════════════════════════════════════════
            # STEP 3+4: EVALUATE + QUALITY GATE + RETRY
            # ═══════════════════════════════════════════════
            # FIX: Track evaluated params to avoid re-evaluation
            evaluated_params = set()
            all_trial_results = []
            display_rank = 0  # FIX: Clean sequential numbering

            def _params_key(params):
                """Hashable key for a param dict."""
                return frozenset(
                    (k, round(v, 6) if isinstance(v, float) else v)
                    for k, v in sorted(params.items()))

            def _evaluate_batch(trials_list, label=""):
                """Evaluate a batch, skip already-evaluated, sequential numbering."""
                nonlocal display_rank
                new_results = []

                for ti in trials_list:
                    tparams = ti.get('params', {})
                    pk = _params_key(tparams)

                    # FIX: Skip params already evaluated in prior round
                    if pk in evaluated_params:
                        continue
                    evaluated_params.add(pk)

                    # FIX: Sequential rank 1, 2, 3... (not optimizer trial ID)
                    display_rank += 1
                    ism = ti.get('metrics', ti)

                    print(f"\n  {'━' * 66}")
                    print(f"  Trial #{display_rank}/{top_n}  "
                          f"{label}"
                          f"(IS: SR={ism.get('sharpe',0):.2f} "
                          f"Ret={ism.get('return',0):+.1f}% "
                          f"WR={ism.get('win_rate',0):.1f}% "
                          f"DD={ism.get('max_drawdown',0):.1f}%)")
                    print(f"  {'━' * 66}")

                    tr = _evaluate_single_trial(
                        display_rank, tparams, ism,
                        strategy_name, leverage, capital,
                        t1_data, t1_detail, t1_ef,
                        symbol, tf, args, sym_dir,
                        min_trades=dyn_min_trades)
                    new_results.append(tr)
                    all_trial_results.append(tr)

                return new_results

            # ── Round 0: Evaluate initial top-N ──
            print(f"\n  🔄 Step 3: Evaluating {n_eval} trials on T1")
            _evaluate_batch(top_trials_json[:n_eval])

            # ── Quality gate check ──
            best = _find_best_trial(
                all_trial_results,
                args['min_validate'], args['min_win_rate'])

            # ── Incremental retry loop ──
            # Only when use_incremental=True (Optuna study is alive in opt_state)
            if (best is None and use_incremental
                    and opt_state is not None and is_data is not None):
                from optimize.rolling_wfo import _optimize_increment

                gates = []
                if args['min_validate'] > 0:
                    gates.append(f"val≥{args['min_validate']}/4")
                if args['min_win_rate'] > 0:
                    gates.append(f"WR≥{args['min_win_rate']:.0f}%")
                gate_str = " + ".join(gates) if gates else ""
                if gate_str:
                    print(f"\n  ⚠️ No trial passed gate ({gate_str})")

                extra = abs(args['retry_trials_delta'])
                param_subset = None
                if args['optimize_params']:
                    param_subset = [p.strip()
                                    for p in args['optimize_params'].split(',')]

                for retry in range(1, args['max_retries'] + 1):
                    prev_total = opt_state.total_trials
                    print(f"\n  🔄 RETRY {retry}/{args['max_retries']}: "
                          f"+{extra} trials (study: {prev_total} → "
                          f"{prev_total + extra})")

                    new_top = _optimize_increment(
                        opt_state, strategy_name,
                        is_data, is_detail, capital, leverage,
                        args['method'], args['objective'],
                        extra, args['n_jobs'],
                        symbol, tf, no_intrabar,
                        param_subset, top_n, verbose=True)

                    if not new_top:
                        continue

                    # Count truly new params
                    new_count = sum(
                        1 for ti in new_top
                        if _params_key(ti.get('params', {}))
                        not in evaluated_params)
                    print(f"\n  ✅ {len(new_top)} top trials, "
                          f"{new_count} new → evaluating on T1...")

                    if new_count == 0:
                        print(f"  ⚠️ All top params already evaluated, "
                              f"retrying with more trials...")
                        continue

                    # FIX: Only evaluate NEW params (skip duplicates)
                    _evaluate_batch(new_top, label=f"[R{retry}] ")

                    # Update params JSON
                    _save_params_json(
                        params_file, strategy_name, symbol, tf,
                        args['objective'], args['method'], new_top)

                    # Check gate on ALL evaluated results
                    best = _find_best_trial(
                        all_trial_results,
                        args['min_validate'], args['min_win_rate'])

                    if best is not None:
                        layers = best.get('validate', {}).get(
                            'layers_passed', 0)
                        wr = best.get('backtest', {}).get('win_rate', 0)
                        print(f"\n  ✅ Qualified trial on retry {retry} "
                              f"({layers}/4, WR={wr:.1f}%) "
                              f"[study: {opt_state.total_trials} trials]")
                        break

                    if gate_str:
                        print(f"\n  ⚠️ Still no trial passed ({gate_str})")

            # ── Fallback ──
            if best is None:
                best = _find_best_trial(all_trial_results, 0, 0.0)
                if best and has_gate:
                    print(f"\n  ⚠️ Fallback: best unqualified trial "
                          f"(#{best.get('trial_rank','?')})")

            sym_result['trials'] = all_trial_results
            sym_result['best_trial'] = best

            _print_leaderboard(sym_result)

            # ═══════════════════════════════════════════════
            # STEP 5: T2 VERIFICATION + ANALYSIS
            # ═══════════════════════════════════════════════
            if has_t2 and best and isinstance(best.get('backtest'), dict):
                # ── Skip T2 if best is fallback (no trial passed gates) ──
                is_fallback = best.get('skipped_reason') is not None
                if is_fallback:
                    print(f"\n  ⏭️  Skip T2: no trial passed validation gates "
                          f"(best #{best.get('trial_rank', '?')} is fallback)")
                    sym_result['t2_result'] = {
                        'total_return': 0.0, 'sharpe_ratio': 0.0,
                        'win_rate': 0.0, 'max_drawdown': 0.0,
                        'profit_factor': 0.0, 'n_trades': 0,
                        'n_longs': 0, 'n_shorts': 0,
                        'skipped': True, 'reason': 'fallback_trial',
                    }
                else:
                    print(f"\n  🎯 Step 5: T2 Verification "
                          f"({args['bt2_start']} → {args['bt2_end']})")

                    t2_load = {
                        'symbol': symbol, 'timeframe': tf,
                        'start': args['bt2_start'], 'end': args['bt2_end']}

                    t2_data, t2_detail, _, _ = _load_data(t2_load, tf)
                    t2_ef = _make_engine_factory(
                        capital, t2_detail, market, no_intrabar=no_intrabar)

                    print(f"     {len(t2_data['close'])} bars loaded")
                    print(f"    🚀 Backtest...", end=" ", flush=True)

                    s2 = get_strategy(strategy_name)
                    s2.set_params({'leverage': leverage})
                    s2.set_params(best.get('params', {}))

                    e2 = t2_ef()
                    r2 = e2.run(s2, t2_data, symbol, tf)

                    from optimize.grid_search import compute_monthly_stats as _cms

                    t2 = {
                        'total_return': round(r2.total_return, 2),
                        'annual_return': round(r2.annual_return, 2),
                        'sharpe_ratio': round(r2.sharpe_ratio, 2),
                        'sortino_ratio': round(r2.sortino_ratio, 2),
                        'max_drawdown': round(r2.max_drawdown, 2),
                        'calmar_ratio': round(r2.calmar_ratio, 2),
                        'win_rate': round(r2.win_rate, 1),
                        'profit_factor': round(r2.profit_factor, 2),
                        'n_trades': r2.n_trades,
                        'n_longs': r2.n_longs,
                        'n_shorts': r2.n_shorts,
                        'avg_win': round(r2.avg_win, 2),
                        'avg_loss': round(r2.avg_loss, 2),
                    }

                    n_liq = sum(1 for t in r2.trades
                                if t.exit_reason == 'liquidation')
                    if n_liq > 0:
                        t2['liquidations'] = n_liq

                    print(f"SR={r2.sharpe_ratio:.2f} "
                          f"Ret={r2.total_return:+.1f}% "
                          f"WR={r2.win_rate:.1f}% "
                          f"DD={r2.max_drawdown:.1f}% "
                          f"T={r2.n_trades}"
                          f"{f' ⚠️{n_liq}LIQ' if n_liq else ''}")

                    # ── T2 Validate ──
                    try:
                        print(f"    🔬 T2 Validate...", end=" ", flush=True)
                        from optimize.anti_overfit import AntiOverfitPipeline
                        sv2 = get_strategy(strategy_name)
                        sv2.set_params({'leverage': leverage})
                        sv2.set_params(best.get('params', {}))
                        pg2 = _build_validation_grid(sv2)
                        vp2 = AntiOverfitPipeline(
                            wfa_windows=4,
                            mc_simulations=args.get('mc_sims', 500),
                            fail_fast=False, verbose=False)
                        vr2 = vp2.run(sv2, t2_data, t2_ef, pg2, symbol, tf)
                        t2['validate'] = {
                            'all_passed': getattr(vr2, 'all_passed', False),
                            'layers_passed': getattr(vr2, 'layers_passed', 0),
                            'wfa_passed': getattr(vr2, 'wfa_passed', False),
                            'mc_passed': getattr(vr2, 'mc_passed', False),
                            'sensitivity_passed': getattr(vr2, 'sensitivity_passed', False),
                            'overfit_passed': getattr(vr2, 'overfit_passed', False),
                        }
                        m2v = "✅" if t2['validate']['all_passed'] else "⚠️"
                        print(f"{m2v} {t2['validate']['layers_passed']}/4 layers")
                    except Exception as e:
                        print(f"❌ {e}")
                        t2['validate'] = {'status': 'FAILED', 'error': str(e)}

                    # ── T2 Regime ──
                    try:
                        print(f"    🔄 T2 Regime...", end=" ", flush=True)
                        from ml.regime_detector import detect_regime, strategy_regime_performance
                        sr2 = get_strategy(strategy_name)
                        sr2.set_params({'leverage': leverage})
                        sr2.set_params(best.get('params', {}))
                        _ita2 = sr2.get_param('itrend_alpha', 0.07)
                        rr2 = detect_regime(t2_data, method='vt', verbose=False,
                                            itrend_alpha=_ita2)
                        perf2 = strategy_regime_performance(
                            sr2, t2_data, t2_ef, rr2,
                            symbol=symbol, timeframe=tf, verbose=False)
                        t2['regime'] = {str(k): v for k, v in perf2.items()
                                        if isinstance(v, dict)} if isinstance(perf2, dict) else {}
                        print("✅")
                    except Exception as e:
                        print(f"❌ {e}")
                        t2['regime'] = {'status': 'FAILED', 'error': str(e)}

                    # ── T2 Targets ──
                    try:
                        print(f"    📅 T2 Targets...", end=" ", flush=True)
                        from ml.temporal_targets import (
                            evaluate_targets,
                            CONSERVATIVE_TARGETS, AGGRESSIVE_TARGETS, CONSISTENCY_TARGETS)
                        tmap2 = {
                            'conservative': CONSERVATIVE_TARGETS,
                            'aggressive': AGGRESSIVE_TARGETS,
                            'consistency': CONSISTENCY_TARGETS}
                        tspecs2 = tmap2.get(args.get('targets', 'conservative'),
                                            CONSERVATIVE_TARGETS)
                        tt2 = evaluate_targets(
                            tspecs2, r2.trades,
                            t2_data.get('timestamp', np.arange(len(t2_data['close']))),
                            r2.equity_curve,
                            initial_capital=capital, verbose=False)
                        t2['targets'] = {
                            'all_passed': getattr(tt2, 'all_passed', False),
                            'n_passed': getattr(tt2, 'n_passed', 0),
                            'n_targets': getattr(tt2, 'n_targets', 0)}
                        m2t = "✅" if t2['targets']['all_passed'] else "⚠️"
                        print(f"{m2t} {t2['targets']['n_passed']}/{t2['targets']['n_targets']} targets")
                    except Exception as e:
                        print(f"❌ {e}")
                        t2['targets'] = {'status': 'FAILED', 'error': str(e)}

                    # Monthly stats for T2
                    try:
                        ms2 = _cms(r2.trades)
                        if ms2['n_months'] >= 1:
                            t2['monthly'] = {
                                'avg_return': round(ms2['avg_monthly_return'], 2),
                                'pct_positive': round(ms2['pct_positive'], 1),
                                'monthly_sharpe': round(ms2['monthly_sharpe'], 2),
                                'worst_month': round(ms2['worst_month'], 2),
                                'best_month': round(ms2['best_month'], 2),
                                'n_months': ms2['n_months'],
                                'months': ms2['months'],
                            }
                    except Exception:
                        pass

                    # Degradation
                    t1_sr = best['backtest'].get('sharpe_ratio', 0)
                    t1_ret = best['backtest'].get('total_return', 0)
                    if abs(t1_sr) > 0.01:
                        t2['degradation_sharpe'] = round(
                            (t1_sr - r2.sharpe_ratio) / abs(t1_sr) * 100, 1)
                    if abs(t1_ret) > 0.01:
                        t2['degradation_return'] = round(
                            (t1_ret - r2.total_return) / abs(t1_ret) * 100, 1)

                    sym_result['t2_result'] = t2

            # ═══════════════════════════════════════════════
            # DETAILED BEST TRIAL REPORT
            # ═══════════════════════════════════════════════
            _print_best_trial_report(sym_result, args)

            # Save
            sym_result['elapsed_seconds'] = round(time.time() - sym_t0, 1)
            sym_json = sym_dir / f"results_{symbol}_{tf}.json"
            with open(sym_json, 'w') as f:
                json.dump(sym_result, f, indent=2, default=_jd)
            print(f"\n  📁 Saved: {sym_json}")

            all_results[symbol] = sym_result

        except Exception as e:
            print(f"\n  ❌ Fatal: {e}")
            traceback.print_exc()
            sym_result['error'] = str(e)
            all_results[symbol] = sym_result

    total_elapsed = time.time() - pipeline_t0
    _print_summary(all_results, total_elapsed, output_root, args)
    return all_results


# ═══════════════════════════════════════════════════════════════
#  SAVE PARAMS JSON (compatible format)
# ═══════════════════════════════════════════════════════════════

def _save_params_json(params_file, strategy_name, symbol, tf,
                      objective, method, top_trials):
    """Save params JSON in same format as cmd_optimize."""
    best = top_trials[0] if top_trials else {}
    export = {
        'strategy': strategy_name,
        'symbol': symbol,
        'timeframe': tf,
        'objective': objective,
        'method': method,
        'best_params': best.get('params', {}),
        'metrics': best.get('metrics', {}),
        'top_trials': [
            {
                'rank': i + 1,
                'params': t.get('params', {}),
                'metrics': t.get('metrics', {}),
            }
            for i, t in enumerate(top_trials)
        ],
        'top_5': [
            {
                'rank': i + 1,
                'params': t.get('params', {}),
                'metrics': t.get('metrics', {}),
            }
            for i, t in enumerate(top_trials[:5])
        ],
    }

    params_file.parent.mkdir(parents=True, exist_ok=True)
    with open(params_file, 'w') as f:
        json.dump(export, f, indent=2, default=str)
    print(f"\n  📁 Params saved: {params_file}")


# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════

def _sanitize(m):
    safe = {}
    for k, v in m.items():
        if isinstance(v, (np.integer,)):   safe[k] = int(v)
        elif isinstance(v, (np.floating,)): safe[k] = float(v)
        elif isinstance(v, (int, float, str, bool)): safe[k] = v
    return safe


def _jd(obj):
    if isinstance(obj, (np.integer,)):  return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray):     return obj.tolist()
    return str(obj)


def _print_banner(args, output_root):
    syms = ', '.join(args['symbols'])
    has_t2 = bool(args.get('bt2_start'))
    has_gate = args['min_validate'] > 0 or args['min_win_rate'] > 0

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║         CryptoLab — Pipeline v2 (Dual Test + Retry)        ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Strategy:   {args['strategy']}")
    print(f"║  Symbols:    {syms}")
    print(f"║  Timeframe:  {args['timeframe']}")
    print(f"║  Capital:    ${args['capital']:,.0f}  |  Leverage: {args['leverage']}x")
    if not args['skip_optimize']:
        print(f"║  Optimize:   {args['opt_start']} → {args['opt_end']}")
        print(f"║  Method:     {args['method']} / {args['objective']}")
        print(f"║  Trials:     {args['n_trials']}  |  Jobs: {args['n_jobs']}")
    print(f"║  T1 (select): {args['bt_start']} → {args['bt_end']}")
    if has_t2:
        print(f"║  T2 (verify): {args['bt2_start']} → {args['bt2_end']}")
    print(f"║  Top trials: {args['top_n']}  |  "
          f"Min trades: {args['min_trades_per_week']}/week")
    if has_gate:
        gates = []
        if args['min_validate'] > 0:
            gates.append(f"val≥{args['min_validate']}/4")
        if args['min_win_rate'] > 0:
            gates.append(f"WR≥{args['min_win_rate']:.0f}%")
        extra = abs(args['retry_trials_delta'])
        print(f"║  Quality:    {' + '.join(gates)} | "
              f"retries: {args['max_retries']} (+{extra}/retry, same study)")
    print(f"║  Output:     {output_root}")
    print("╚══════════════════════════════════════════════════════════════╝")


def _print_leaderboard(sym_result):
    trials = sym_result.get('trials', [])
    if not trials:
        return
    best = sym_result.get('best_trial', {})

    print(f"\n  {'─' * 76}")
    print(f"  {'#':>3} {'SR':>6} {'Ret%':>8} {'WR%':>6} {'DD%':>6} "
          f"{'PF':>5} {'T':>4} {'L/S':>7} {'Val':>5} {'Tgt':>5}")
    print(f"  {'─' * 76}")

    for t in trials:
        bt = t.get('backtest', {})
        val = t.get('validate', {})
        tgt = t.get('targets', {})

        if not isinstance(bt, dict) or 'sharpe_ratio' not in bt:
            print(f"  #{t.get('trial_rank','?'):>2}  — failed —")
            continue

        vs = (f"{val.get('layers_passed','?')}/4"
              if isinstance(val, dict) and 'layers_passed' in val else '—')
        ts = (f"{tgt.get('n_passed','?')}/{tgt.get('n_targets','?')}"
              if isinstance(tgt, dict) and 'n_passed' in tgt else '—')
        mk = (" ★" if best and t.get('trial_rank') == best.get('trial_rank')
               else "")
        ls = f"{bt.get('n_longs',0)}/{bt.get('n_shorts',0)}"
        liq = bt.get('liquidations', 0)
        liq_s = f" ⚠️{liq}L" if liq else ""

        print(f"  #{t['trial_rank']:>2} "
              f"{bt.get('sharpe_ratio',0):>5.2f} "
              f"{bt.get('total_return',0):>+7.1f}% "
              f"{bt.get('win_rate',0):>5.1f}% "
              f"{bt.get('max_drawdown',0):>5.1f}% "
              f"{bt.get('profit_factor',0):>4.2f} "
              f"{bt.get('n_trades',0):>4} "
              f"{ls:>7} "
              f"{vs:>5} {ts:>5}{mk}{liq_s}")


def _print_best_trial_report(sym_result, args):
    """Print detailed comparison of best trial: IS → T1 → T2."""
    best = sym_result.get('best_trial')
    if not best or not isinstance(best.get('backtest'), dict):
        return

    bt = best['backtest']
    ism = best.get('in_sample', {})
    val = best.get('validate', {})
    tgt = best.get('targets', {})
    mo = best.get('monthly', {})
    t2 = sym_result.get('t2_result', {})
    has_t2 = isinstance(t2, dict) and 'sharpe_ratio' in t2

    w = 76
    print(f"\n  ╔{'═' * w}╗")
    print(f"  ║  🏆 BEST TRIAL: #{best.get('trial_rank','?')}"
          f"{' ' * (w - 22 - len(str(best.get('trial_rank','?'))))}║")
    print(f"  ╠{'═' * w}╣")

    # ── Metrics comparison table ──
    def _row(label, t1_val, t2_val=None, fmt="+.1f", suffix="%"):
        t1s = f"{t1_val:{fmt}}{suffix}" if t1_val is not None else "—"
        if has_t2 and t2_val is not None:
            t2s = f"{t2_val:{fmt}}{suffix}" if t2_val is not None else "—"
            print(f"  ║  {label:<22} {'T1':>6}: {t1s:>10}   "
                  f"{'T2':>6}: {t2s:>10}{'':>{w-64}}║")
        else:
            print(f"  ║  {label:<22} {t1s:>10}{'':>{w-36}}║")

    header = "T1 (select)" if has_t2 else "OOS Backtest"
    periods = f"  ║  {header}: {args['bt_start']} → {args['bt_end']}"
    if has_t2:
        periods += f"   T2 (verify): {args['bt2_start']} → {args['bt2_end']}"
    periods += " " * max(0, w - len(periods) + 4) + "║"
    print(periods)
    print(f"  ║{'─' * w}║")

    _row("Return",      bt.get('total_return'), t2.get('total_return'))
    _row("Sharpe",       bt.get('sharpe_ratio'), t2.get('sharpe_ratio'), fmt=".2f", suffix="")
    _row("Sortino",      bt.get('sortino_ratio'), t2.get('sortino_ratio'), fmt=".2f", suffix="")
    _row("Max Drawdown", bt.get('max_drawdown'), t2.get('max_drawdown'), fmt=".1f")
    _row("Calmar",       bt.get('calmar_ratio'), t2.get('calmar_ratio'), fmt=".2f", suffix="")
    _row("Win Rate",     bt.get('win_rate'), t2.get('win_rate'), fmt=".1f")
    _row("Profit Factor", bt.get('profit_factor'), t2.get('profit_factor'), fmt=".2f", suffix="")
    _row("Trades",       bt.get('n_trades'), t2.get('n_trades'), fmt="d", suffix="")
    _row("Avg Win",      bt.get('avg_win'), t2.get('avg_win'))
    _row("Avg Loss",     bt.get('avg_loss'), t2.get('avg_loss'))

    # Liquidations
    liq1 = bt.get('liquidations', 0)
    liq2 = t2.get('liquidations', 0) if has_t2 else 0
    if liq1 > 0 or liq2 > 0:
        _row("Liquidations", liq1, liq2 if has_t2 else None, fmt="d", suffix="")

    # ── Degradation ──
    if has_t2:
        print(f"  ║{'─' * w}║")
        deg_sr = t2.get('degradation_sharpe', 0)
        deg_ret = t2.get('degradation_return', 0)
        di_sr = "✅" if abs(deg_sr) < 30 else "⚠️" if abs(deg_sr) < 60 else "❌"
        di_ret = "✅" if abs(deg_ret) < 30 else "⚠️" if abs(deg_ret) < 60 else "❌"
        icon = "🟢" if t2.get('total_return', 0) > 0 else "🔴"
        print(f"  ║  Degradation SR:  {deg_sr:+.1f}% {di_sr}   "
              f"Ret: {deg_ret:+.1f}% {di_ret}   "
              f"T2 result: {icon}{'':>{w-55}}║")

    # ── Validation ──
    print(f"  ║{'─' * w}║")
    if isinstance(val, dict) and 'layers_passed' in val:
        layers = val['layers_passed']
        checks = [
            ('WFA',         val.get('wfa_passed', False)),
            ('MonteCarlo',  val.get('mc_passed', False)),
            ('Sensitivity', val.get('sensitivity_passed', False)),
            ('Overfit',     val.get('overfit_passed', False)),
        ]
        parts = [f"{'✅' if p else '❌'} {n}" for n, p in checks]
        v_line = f"  ║  Validation: {layers}/4 — {' | '.join(parts)}"
        v_line += " " * max(0, w - len(v_line) + 4) + "║"
        print(v_line)
    else:
        skip = best.get('skipped_reason', 'gate failed')
        print(f"  ║  Validation: skipped ({skip})"
              f"{'':>{max(0, w - 30 - len(str(skip)))}}║")

    # Targets
    if isinstance(tgt, dict) and 'n_passed' in tgt:
        t_line = (f"  ║  Targets:    {tgt['n_passed']}/{tgt['n_targets']} "
                  f"({args.get('targets','conservative')})")
        t_line += " " * max(0, w - len(t_line) + 4) + "║"
        print(t_line)

    # ── Monthly T1 ──
    if isinstance(mo, dict) and 'months' in mo:
        print(f"  ║{'─' * w}║")
        print(f"  ║  📅 T1 Monthly:"
              f"{'':>{w - 18}}║")
        months = mo['months']
        for i in range(0, len(months), 4):
            chunk = months[i:i+4]
            parts = []
            for m in chunk:
                mi = "✅" if m['pnl_pct'] > 0 else "❌"
                parts.append(f"{m['month']}: {m['pnl_pct']:+.1f}% "
                             f"({m['n_trades']}t {m['win_rate']:.0f}%wr) {mi}")
            line = "  ║  " + "   ".join(parts)
            line += " " * max(0, w - len(line) + 4) + "║"
            print(line)
        avg = mo.get('avg_return', 0)
        pct = mo.get('pct_positive', 0)
        msr = mo.get('monthly_sharpe', 0)
        worst = mo.get('worst_month', 0)
        best_m = mo.get('best_month', 0)
        s_line = (f"  ║  ── Avg: {avg:+.1f}%/mo | "
                  f"Pos: {pct:.0f}% | mSR: {msr:.2f} | "
                  f"Worst: {worst:+.1f}% | Best: {best_m:+.1f}%")
        s_line += " " * max(0, w - len(s_line) + 4) + "║"
        print(s_line)

    # ── Monthly T2 ──
    t2_mo = t2.get('monthly', {}) if has_t2 else {}
    if isinstance(t2_mo, dict) and 'months' in t2_mo:
        print(f"  ║{'─' * w}║")
        print(f"  ║  📅 T2 Monthly:"
              f"{'':>{w - 18}}║")
        months = t2_mo['months']
        for i in range(0, len(months), 4):
            chunk = months[i:i+4]
            parts = []
            for m in chunk:
                mi = "✅" if m['pnl_pct'] > 0 else "❌"
                parts.append(f"{m['month']}: {m['pnl_pct']:+.1f}% "
                             f"({m['n_trades']}t {m['win_rate']:.0f}%wr) {mi}")
            line = "  ║  " + "   ".join(parts)
            line += " " * max(0, w - len(line) + 4) + "║"
            print(line)
        avg = t2_mo.get('avg_return', 0)
        pct = t2_mo.get('pct_positive', 0)
        msr = t2_mo.get('monthly_sharpe', 0)
        worst = t2_mo.get('worst_month', 0)
        best_m = t2_mo.get('best_month', 0)
        s_line = (f"  ║  ── Avg: {avg:+.1f}%/mo | "
                  f"Pos: {pct:.0f}% | mSR: {msr:.2f} | "
                  f"Worst: {worst:+.1f}% | Best: {best_m:+.1f}%")
        s_line += " " * max(0, w - len(s_line) + 4) + "║"
        print(s_line)

    # ── Key params ──
    params = best.get('params', {})
    if params:
        print(f"  ║{'─' * w}║")
        # Show most important params in compact form
        key_params = ['alpha_method', 'sltp_type', 'leverage', 'sl_atr_mult',
                      'tp1_rr', 'confidence_min', 'trail_activate_pct',
                      'trail_pullback_pct', 'be_pct']
        shown = {k: params[k] for k in key_params if k in params}
        rest = {k: v for k, v in params.items() if k not in shown}
        p_line = "  ║  Params: " + ", ".join(
            f"{k}={v}" for k, v in shown.items())
        if len(p_line) > w:
            p_line = p_line[:w+2]
        p_line += " " * max(0, w - len(p_line) + 4) + "║"
        print(p_line)
        if rest:
            r_line = "  ║          " + ", ".join(
                f"{k}={v}" for k, v in rest.items())
            if len(r_line) > w:
                r_line = r_line[:w+2]
            r_line += " " * max(0, w - len(r_line) + 4) + "║"
            print(r_line)

    print(f"  ╚{'═' * w}╝")


def _print_summary(all_results, total_elapsed, output_root, args):
    has_t2 = bool(args.get('bt2_start'))

    print(f"\n\n{'═' * 90}")
    print(f"  PIPELINE v2 SUMMARY — {len(all_results)} symbol(s) in "
          f"{total_elapsed:.0f}s")
    print(f"{'═' * 90}")

    if has_t2:
        print(f"  {'Symbol':<10} {'#':>3} {'T1 SR':>6} {'T1 Ret':>8} "
              f"{'WR%':>5} {'DD%':>5} {'PF':>5} {'T':>4} {'Val':>4} {'Tgt':>4} "
              f"│ {'T2 SR':>6} {'T2 Ret':>8} {'WR%':>5} {'DD%':>5} {'T':>4} "
              f"{'Deg':>5}")
        print(f"  {'─' * 86}")
    else:
        print(f"  {'Symbol':<10} {'#':>3} {'SR':>6} {'Ret':>8} {'WR%':>5} "
              f"{'DD%':>5} {'PF':>5} {'T':>4} {'Val':>4} {'Tgt':>4}")
        print(f"  {'─' * 58}")

    summary = {
        'pipeline_version': '2.0',
        'timestamp': datetime.now().isoformat(),
        'config': args,
        'total_elapsed': round(total_elapsed, 1),
        'symbols': {},
    }

    for symbol, sr in all_results.items():
        best = sr.get('best_trial')
        if (best and isinstance(best.get('backtest'), dict)
                and 'sharpe_ratio' in best['backtest']):
            bt = best['backtest']
            val = best.get('validate', {})
            tgt = best.get('targets', {})
            vs = (f"{val.get('layers_passed','?')}/4"
                  if isinstance(val, dict) and 'layers_passed' in val
                  else '—')
            ts = (f"{tgt.get('n_passed','?')}/{tgt.get('n_targets','?')}"
                  if isinstance(tgt, dict) and 'n_passed' in tgt else '—')

            line = (f"  {symbol:<12} "
                    f"#{best.get('trial_rank', '?'):>3} "
                    f"{bt['sharpe_ratio']:>6.2f} "
                    f"{bt['total_return']:>+8.1f}% "
                    f"{bt['win_rate']:>5.1f}% "
                    f"{bt['max_drawdown']:>5.1f}% "
                    f"{bt.get('profit_factor', 0):>5.2f} "
                    f"{bt.get('n_trades', 0):>4} "
                    f"{vs:>4} {ts:>4}")

            t2 = sr.get('t2_result', {})
            if has_t2 and isinstance(t2, dict) and 'sharpe_ratio' in t2:
                deg = t2.get('degradation_sharpe', 0)
                icon = "🟢" if t2.get('total_return', 0) > 0 else "🔴"
                if t2.get('skipped'):
                    line += f" │ {'—':>6} {'SKIP':>8} {'—':>5} {'—':>5} {'—':>4} {'—':>5}"
                else:
                    line += (f" │ "
                             f"{t2['sharpe_ratio']:>6.2f} "
                             f"{t2['total_return']:>+8.1f}% "
                             f"{t2.get('win_rate', 0):>5.1f}% "
                             f"{t2.get('max_drawdown', 0):>5.1f}% "
                             f"{t2.get('n_trades', 0):>4} "
                             f"{deg:>+5.0f}% {icon}")
            print(line)

            sym_summary = {
                'best_trial': best.get('trial_rank'),
                't1_sharpe': bt['sharpe_ratio'],
                't1_return': bt['total_return'],
                't1_win_rate': bt['win_rate'],
                't1_max_dd': bt['max_drawdown'],
                't1_pf': bt.get('profit_factor', 0),
                't1_trades': bt.get('n_trades', 0),
                'validated': (val.get('all_passed', False)
                              if isinstance(val, dict) else False),
                'val_layers': (val.get('layers_passed', 0)
                               if isinstance(val, dict) else 0),
            }
            if isinstance(t2, dict) and 'sharpe_ratio' in t2:
                sym_summary.update({
                    't2_sharpe': t2['sharpe_ratio'],
                    't2_return': t2['total_return'],
                    't2_win_rate': t2.get('win_rate', 0),
                    't2_max_dd': t2.get('max_drawdown', 0),
                    't2_pf': t2.get('profit_factor', 0),
                    't2_trades': t2.get('n_trades', 0),
                    't2_degradation_sr': t2.get('degradation_sharpe', 0),
                    't2_degradation_ret': t2.get('degradation_return', 0),
                })
            summary['symbols'][symbol] = sym_summary
        else:
            print(f"  {symbol:<10}  — no valid results —")
            summary['symbols'][symbol] = {'status': 'FAILED'}

    print(f"  {'─' * (86 if has_t2 else 58)}")
    print(f"  Output: {output_root}")
    print(f"{'═' * 90}")

    sf = output_root / "pipeline_summary.json"
    with open(sf, 'w') as f:
        json.dump(summary, f, indent=2, default=_jd)
    print(f"\n📁 Summary: {sf}")


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print(__doc__)
        sys.exit(0)
    args = parse_args()
    run_pipeline(args)