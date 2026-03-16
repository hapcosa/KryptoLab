#!/usr/bin/env python3
"""
CryptoLab — Rolling WFO Pipeline (Dual Test — Bot Simulation)
===============================================================

Per window: optimize(IS) → evaluate top-N on TEST-1 (select best)
            → backtest winner on TEST-2 (verify on unseen data)

TEST-2 simulates live trading after auto-reoptimization.
The T1→T2 degradation ratio tells you how reliable the selection is.

Usage:
    python wfo_pipeline.py \\
      --symbol ETHUSDT --strategy cybercycle --tf 1h \\
      --start 2023-01-01 --end 2025-06-01 \\
      --opt-days 90 --test1-days 60 --test2-days 30 --step-days 30 \\
      --method bayesian --objective monthly_robust \\
      --n-trials 150 --n-jobs -1 --top-n 10

    # Quick grid test:
    python wfo_pipeline.py \\
      --symbol BTCUSDT --strategy cybercycle --tf 4h \\
      --start 2023-06-01 --end 2025-12-01 \\
      --opt-days 120 --test1-days 60 --test2-days 30 --step-days 60 \\
      --method grid --top-n 5

    # Optimize only risk params:
    python wfo_pipeline.py \\
      --symbol SOLUSDT --strategy cybercycle --tf 1h \\
      --start 2024-01-01 --end 2026-01-01 \\
      --opt-days 90 --test1-days 60 --test2-days 30 --step-days 30 \\
      --optimize-params sl_atr_mult,tp1_rr,confidence_min,leverage

    # Require validation + win rate + retry if not found:
    python wfo_pipeline.py \\
      --symbol ETHUSDT --strategy cybercycle --tf 1h \\
      --start 2023-01-01 --end 2025-06-01 \\
      --opt-days 90 --test1-days 60 --test2-days 30 --step-days 30 \\
      --n-trials 400 --min-validate 2 --min-win-rate 40 \\
      --max-retries 2 --retry-trials-delta 100

Full options:
    --symbol          STR   Trading pair (required)
    --strategy        STR   cybercycle|gaussbands|smartmoney (default: cybercycle)
    --tf              STR   Timeframe (default: 1h)
    --start           DATE  Global start YYYY-MM-DD (required)
    --end             DATE  Global end YYYY-MM-DD (required)
    --opt-days        INT   IS optimization window days (required)
    --test1-days      INT   TEST-1 selection window days (required)
    --test2-days      INT   TEST-2 verification window days (required)
    --step-days       INT   Days between re-optimizations (required)
    --capital         FLOAT Starting capital (default: 1000)
    --leverage        FLOAT Leverage (default: 3.0)
    --method          STR   grid|bayesian|genetic (default: bayesian)
    --objective       STR   Objective function (default: monthly_robust)
    --n-trials        INT   Trials per window (default: 150)
    --n-jobs          INT   Parallel workers (default: -1)
    --top-n           INT   Top N trials to evaluate per window (default: 10)
    --mc-sims         INT   Monte Carlo sims for validate (default: 500)
    --targets         STR   conservative|aggressive|consistency
    --no-intrabar           Disable intra-bar signals
    --min-trades-per-week FLOAT  Min trades/week for gate (default: 1.5)
    --min-validate    INT   Min validation layers to accept trial (0-4, default: 0)
    --min-win-rate    FLOAT Min win rate % to accept trial (0-100, default: 0)
    --max-retries     INT   Max re-optimization retries per window (default: 0)
    --retry-trials-delta INT  Extra trials per retry, added to same study (default: 100)
    --optimize-params STR   Only optimize these params (comma-sep)
    --exclude-params  STR   Exclude these params
"""

import sys
from datetime import datetime


def parse_args():
    args = {
        'symbol': None, 'strategy': 'cybercycle', 'timeframe': '1h',
        'start': None, 'end': None,
        'opt_days': None, 'test1_days': None, 'test2_days': None,
        'step_days': None,
        'capital': 1000.0, 'leverage': 3.0,
        'method': 'bayesian', 'objective': 'monthly_robust',
        'n_trials': 150, 'n_jobs': -1, 'top_n': 10,
        'mc_sims': 500, 'targets': 'conservative',
        'no_intrabar': False, 'min_trades_per_week': 1.5,
        'min_validate': 0, 'min_win_rate': 0.0,
        'max_retries': 0, 'retry_trials_delta': 100,
        'optimize_params': None, 'exclude_params': None,
    }

    i = 1
    while i < len(sys.argv):
        a = sys.argv[i]
        if a == '--symbol':          args['symbol'] = sys.argv[i+1]; i += 2
        elif a == '--strategy':      args['strategy'] = sys.argv[i+1]; i += 2
        elif a == '--tf':            args['timeframe'] = sys.argv[i+1]; i += 2
        elif a == '--start':         args['start'] = sys.argv[i+1]; i += 2
        elif a == '--end':           args['end'] = sys.argv[i+1]; i += 2
        elif a == '--opt-days':      args['opt_days'] = int(sys.argv[i+1]); i += 2
        elif a == '--test1-days':    args['test1_days'] = int(sys.argv[i+1]); i += 2
        elif a == '--test2-days':    args['test2_days'] = int(sys.argv[i+1]); i += 2
        elif a == '--step-days':     args['step_days'] = int(sys.argv[i+1]); i += 2
        elif a == '--capital':       args['capital'] = float(sys.argv[i+1]); i += 2
        elif a == '--leverage':      args['leverage'] = float(sys.argv[i+1]); i += 2
        elif a == '--method':        args['method'] = sys.argv[i+1]; i += 2
        elif a == '--objective':     args['objective'] = sys.argv[i+1]; i += 2
        elif a == '--n-trials':      args['n_trials'] = int(sys.argv[i+1]); i += 2
        elif a == '--n-jobs':        args['n_jobs'] = int(sys.argv[i+1]); i += 2
        elif a == '--top-n':         args['top_n'] = int(sys.argv[i+1]); i += 2
        elif a == '--mc-sims':       args['mc_sims'] = int(sys.argv[i+1]); i += 2
        elif a == '--targets':       args['targets'] = sys.argv[i+1]; i += 2
        elif a == '--min-trades-per-week': args['min_trades_per_week'] = float(sys.argv[i+1]); i += 2
        elif a == '--min-validate':  args['min_validate'] = int(sys.argv[i+1]); i += 2
        elif a == '--min-win-rate':  args['min_win_rate'] = float(sys.argv[i+1]); i += 2
        elif a == '--max-retries':   args['max_retries'] = int(sys.argv[i+1]); i += 2
        elif a == '--retry-trials-delta': args['retry_trials_delta'] = int(sys.argv[i+1]); i += 2
        elif a == '--no-intrabar':   args['no_intrabar'] = True; i += 1
        elif a == '--optimize-params': args['optimize_params'] = sys.argv[i+1]; i += 2
        elif a == '--exclude-params':  args['exclude_params'] = sys.argv[i+1]; i += 2
        elif a in ('--help', '-h'):  print(__doc__); sys.exit(0)
        else:
            print(f"⚠️  Unknown: {a}"); i += 1

    # Validate
    errs = []
    if not args['symbol']:      errs.append("--symbol")
    if not args['start']:       errs.append("--start")
    if not args['end']:         errs.append("--end")
    if args['opt_days'] is None:   errs.append("--opt-days")
    if args['test1_days'] is None: errs.append("--test1-days")
    if args['test2_days'] is None: errs.append("--test2-days")
    if args['step_days'] is None:  errs.append("--step-days")

    if errs:
        print(f"\n❌ Missing: {', '.join(errs)}")
        print(f"\nUsage: python wfo_pipeline.py --symbol ETHUSDT "
              f"--start 2023-01-01 --end 2025-06-01 "
              f"--opt-days 90 --test1-days 60 --test2-days 30 "
              f"--step-days 30")
        sys.exit(1)

    for label, val in [('--start', args['start']), ('--end', args['end'])]:
        try:
            datetime.strptime(val, "%Y-%m-%d")
        except ValueError:
            print(f"❌ Bad date {label}: '{val}'"); sys.exit(1)

    s = datetime.strptime(args['start'], "%Y-%m-%d")
    e = datetime.strptime(args['end'], "%Y-%m-%d")
    total = (e - s).days
    need = args['opt_days'] + args['test1_days'] + args['test2_days']
    if total < need:
        print(f"❌ {total}d available, need ≥{need}d "
              f"(opt={args['opt_days']} + t1={args['test1_days']} "
              f"+ t2={args['test2_days']})")
        sys.exit(1)

    if args['step_days'] < 1:
        print("❌ --step-days must be ≥ 1"); sys.exit(1)

    return args


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ('--help', '-h'):
        print(__doc__); sys.exit(0)

    args = parse_args()

    from optimize.rolling_wfo import (
        RollingWFO, generate_wfo_windows, export_wfo_results)

    windows = generate_wfo_windows(
        args['start'], args['end'],
        args['opt_days'], args['test1_days'],
        args['test2_days'], args['step_days'])

    print(f"\n📋 {len(windows)} windows will be generated")
    print(f"   Per window: optimize({args['opt_days']}d) → "
          f"top-{args['top_n']} on T1({args['test1_days']}d) "
          f"with validate+regime+targets → "
          f"winner on T2({args['test2_days']}d)")
    if args['min_validate'] > 0 or args['min_win_rate'] > 0:
        extra = abs(args['retry_trials_delta'])
        gates = []
        if args['min_validate'] > 0:
            gates.append(f"≥{args['min_validate']}/4 validation layers")
        if args['min_win_rate'] > 0:
            gates.append(f"WR≥{args['min_win_rate']:.0f}%")
        print(f"   Quality gate: {' + '.join(gates)}")
        print(f"   Retry: up to {args['max_retries']}× adding "
              f"+{extra} trials/retry (same Optuna study, incremental)")

    ps = None
    ep = None
    if args['optimize_params']:
        ps = [p.strip() for p in args['optimize_params'].split(',')]
        print(f"   Optimizing only: {', '.join(ps)}")
    if args['exclude_params']:
        ep = [p.strip() for p in args['exclude_params'].split(',')]
        print(f"   Excluding: {', '.join(ep)}")

    # Time estimate
    et = {'grid': 2, 'bayesian': 1, 'genetic': 1}
    opt_s = args['n_trials'] * et.get(args['method'], 1)
    eval_s = args['top_n'] * 15
    t2_s = 5
    per_w = opt_s + eval_s + t2_s
    total_m = len(windows) * per_w / 60
    print(f"   Estimated: ~{total_m:.0f} min ({len(windows)} × ~{per_w}s)")

    wfo = RollingWFO(
        opt_days=args['opt_days'],
        test1_days=args['test1_days'],
        test2_days=args['test2_days'],
        step_days=args['step_days'],
        method=args['method'],
        objective=args['objective'],
        n_trials=args['n_trials'],
        n_jobs=args['n_jobs'],
        top_n=args['top_n'],
        mc_sims=args['mc_sims'],
        target_preset=args['targets'],
        no_intrabar=args['no_intrabar'],
        min_trades_per_week=args['min_trades_per_week'],
        min_validate_layers=args['min_validate'],
        min_win_rate=args['min_win_rate'],
        max_retries=args['max_retries'],
        retry_trials_delta=args['retry_trials_delta'],
        param_subset=ps,
        exclude_params=ep,
        verbose=True,
    )

    try:
        result = wfo.run(
            strategy_name=args['strategy'],
            symbol=args['symbol'],
            timeframe=args['timeframe'],
            start_date=args['start'],
            end_date=args['end'],
            capital=args['capital'],
            leverage=args['leverage'],
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted"); sys.exit(1)
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        import traceback; traceback.print_exc(); sys.exit(1)

    jp = export_wfo_results(result, "output")
    print(f"\n  📄 Results: {jp}")

    # ── VERDICT ──
    print("\n" + "═" * 70)
    nt = result.n_total_windows
    pv = result.n_validated_windows / nt * 100 if nt > 0 else 0
    deg = abs(result.avg_degradation_sharpe)
    ws = result.window_sharpe
    pp = result.pct_profitable_t2
    ar = result.avg_return_per_window

    if ws >= 1.0 and pp >= 60 and deg < 30 and pv >= 50:
        print("  ✅ BOT-READY: Strong window Sharpe + low degradation + validated")
    elif ws >= 0.7 and pp >= 50 and deg < 50:
        print("  ✅ PROMISING: Decent per-window performance")
    elif ar > 0 and pp >= 40:
        print("  ⚠️  MIXED: Positive avg but inconsistent or high degradation")
    elif ar > 0:
        print("  ⚠️  MARGINAL: Barely positive average")
    else:
        print("  ❌ WEAK: Strategy fails on truly unseen data (T2)")

    print(f"     Window Sharpe: {ws:.2f} | "
          f"Avg Return/Win: {ar:+.2f}% | "
          f"Annualized: {result.annual_return:+,.0f}%")
    print(f"     Degradation: {result.avg_degradation_sharpe:+.1f}% SR | "
          f"Profitable T2: {pp:.0f}% | "
          f"Validated: {result.n_validated_windows}/{nt}")

    if deg < 20:
        print("     📊 Selection process is reliable (low degradation)")
    elif deg < 40:
        print("     📊 Selection process is OK but monitor drift")
    else:
        print("     📊 Selection process may be overfit — "
              "try longer T1 or fewer trials")

    print("═" * 70)


if __name__ == '__main__':
    main()