"""
CryptoLab — Effective Parameter Counting

Computes the true number of active parameters for a CyberCycle strategy
configuration, accounting for conditional dependencies.

When alpha_method='kalman', homodyne/mama/autocorrelation params are NOT
explored → they should NOT count as degrees of freedom for DSR.

When sltp_type='sltp_fixed', ATR/RR params are NOT explored and vice versa.

This module is the SINGLE SOURCE OF TRUTH for parameter dependencies.
Used by:
  - optimize/bayesian.py    → conditional search space
  - optimize/anti_overfit.py → effective n_trials for DSR
  - optimize/deflated_sharpe.py → (receives corrected n_trials)

Reference table (CyberCycle v7.1):
┌──────────────────┬──────────────────────────────────────────────┬───────────┐
│ alpha_method      │ Active method-specific params                │ Effective │
├──────────────────┼──────────────────────────────────────────────┼───────────┤
│ manual            │ manual_alpha                                 │ ~26       │
│ homodyne          │ hd_min_period, hd_max_period, alpha_floor    │ ~28       │
│ mama              │ mama_fast, mama_slow, alpha_floor             │ ~28       │
│ autocorrelation   │ ac_min_period, ac_max_period, ac_avg_length, │ ~29       │
│                   │ alpha_floor                                   │           │
│ kalman            │ kal_process_noise, kal_meas_noise,           │ ~31       │
│                   │ kal_alpha_fast, kal_alpha_slow,              │           │
│                   │ kal_sensitivity, alpha_floor                  │           │
├──────────────────┼──────────────────────────────────────────────┼───────────┤
│ sltp_type         │ Active risk params                           │ delta     │
├──────────────────┼──────────────────────────────────────────────┼───────────┤
│ slatr_tprr        │ sl_atr_mult, tp1_rr, tp1_size, tp2_rr       │ +4        │
│ sltp_fixed        │ sl_fixed_pct, tp1_fixed_pct, tp1_fixed_size, │ +4        │
│                   │ tp2_fixed_pct                                │           │
└──────────────────┴──────────────────────────────────────────────┴───────────┘
"""
from typing import Dict, Set, List, Any, Optional


# ═══════════════════════════════════════════════════════════════════
#  ALPHA METHOD DEPENDENCY MAP
#  Single source of truth — update here when adding new alpha methods
# ═══════════════════════════════════════════════════════════════════

ALPHA_METHOD_PARAMS: Dict[str, Set[str]] = {
    'manual':          {'manual_alpha'},
    'homodyne':        {'hd_min_period', 'hd_max_period', 'alpha_floor'},
    'mama':            {'mama_fast', 'mama_slow', 'alpha_floor'},
    'autocorrelation': {'ac_min_period', 'ac_max_period', 'ac_avg_length', 'alpha_floor'},
    'kalman':          {'kal_process_noise', 'kal_meas_noise',
                        'kal_alpha_fast', 'kal_alpha_slow',
                        'kal_sensitivity', 'alpha_floor'},
}

# Union of ALL method-specific params (across all methods)
ALL_METHOD_PARAMS: Set[str] = set()
for _v in ALPHA_METHOD_PARAMS.values():
    ALL_METHOD_PARAMS |= _v


# ═══════════════════════════════════════════════════════════════════
#  SL/TP MODE DEPENDENCIES (v7.1)
#
#  sltp_type = 'slatr_tprr' → ATR-based SL + R:R TPs
#  sltp_type = 'sltp_fixed' → fixed % SL + fixed % TPs
#
#  Params compartidos (siempre activos):
#    leverage, be_pct, use_trailing, trail_activate_pct, trail_pullback_pct
# ═══════════════════════════════════════════════════════════════════

SLTP_MODE_PARAMS: Dict[str, Set[str]] = {
    'slatr_tprr': {'sl_atr_mult', 'tp1_rr', 'tp1_size', 'tp2_rr'},
    'sltp_fixed': {'sl_fixed_pct', 'tp1_fixed_pct', 'tp1_fixed_size', 'tp2_fixed_pct'},
}

# Union of ALL sltp-specific params
ALL_SLTP_PARAMS: Set[str] = set()
for _v in SLTP_MODE_PARAMS.values():
    ALL_SLTP_PARAMS |= _v


# ═══════════════════════════════════════════════════════════════════
#  CORE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def get_inactive_params(params: Dict[str, Any]) -> Set[str]:
    """
    Given a set of strategy parameters, return the set of param NAMES
    that are inactive (not used) due to conditional dependencies.

    Args:
        params: Dict with at least 'alpha_method' and optionally 'sltp_type'

    Returns:
        Set of param names that should NOT count as degrees of freedom
    """
    inactive = set()

    # ── Alpha method dependencies ────────────────────────────────
    alpha_method = params.get('alpha_method', 'mama')
    active_alpha_params = ALPHA_METHOD_PARAMS.get(alpha_method, set())
    inactive |= (ALL_METHOD_PARAMS - active_alpha_params)

    # ── SL/TP mode dependencies ──────────────────────────────────
    sltp_type = params.get('sltp_type', 'slatr_tprr')
    active_sltp_params = SLTP_MODE_PARAMS.get(sltp_type, set())
    inactive |= (ALL_SLTP_PARAMS - active_sltp_params)

    return inactive


def get_active_params(params: Dict[str, Any]) -> Set[str]:
    """
    Inverse of get_inactive_params — return params that ARE active.
    Useful for logging/display.
    """
    inactive = get_inactive_params(params)
    active_alpha = ALPHA_METHOD_PARAMS.get(params.get('alpha_method', 'mama'), set())
    active_sltp = SLTP_MODE_PARAMS.get(params.get('sltp_type', 'slatr_tprr'), set())
    return active_alpha | active_sltp


def count_effective_params(param_defs: list,
                           params: Optional[Dict[str, Any]] = None,
                           alpha_method: Optional[str] = None,
                           sltp_type: Optional[str] = None) -> int:
    """
    Count the effective number of parameters (active dimensions).

    Args:
        param_defs: List of ParamDef from strategy.parameter_defs()
        params:     Full params dict (used to determine modes)
        alpha_method: Override alpha_method (if params not available)
        sltp_type:    Override sltp_type (if params not available)

    Returns:
        Number of parameters that are actually active
    """
    if params:
        am = params.get('alpha_method', alpha_method or 'mama')
        st = params.get('sltp_type', sltp_type or 'slatr_tprr')
    else:
        am = alpha_method or 'mama'
        st = sltp_type or 'slatr_tprr'

    inactive = get_inactive_params({'alpha_method': am, 'sltp_type': st})

    count = 0
    for pdef in param_defs:
        if pdef.name not in inactive:
            count += 1

    return count


def compute_effective_n_trials(param_grid: Dict[str, List[Any]],
                                params: Optional[Dict[str, Any]] = None,
                                alpha_method: Optional[str] = None,
                                sltp_type: Optional[str] = None) -> int:
    """
    Compute effective number of grid trials, excluding inactive param combos.

    For a grid like {'alpha_method': ['mama','kalman'], 'confidence_min': [70,80,90]},
    if the best params use kalman, then mama-specific combos are irrelevant.

    For DSR: n_trials should reflect actual exploration, not theoretical max.

    Args:
        param_grid:   Grid dict {param_name: [values]}
        params:       Best params dict (to determine active method)
        alpha_method: Override
        sltp_type:    Override

    Returns:
        Effective number of independent trials
    """
    if not param_grid:
        return 1

    # Determine which params are inactive
    am = alpha_method
    st = sltp_type
    if not am and params:
        am = params.get('alpha_method')
    if not st and params:
        st = params.get('sltp_type')

    # If grid explores multiple methods/modes → all combos are valid
    if am is None and 'alpha_method' in param_grid:
        am = None  # exploring multiple methods
    if st is None and 'sltp_type' in param_grid:
        st = None  # exploring multiple modes

    if am or st:
        inactive = get_inactive_params({
            'alpha_method': am or 'mama',
            'sltp_type': st or 'slatr_tprr',
        })
    else:
        inactive = set()

    # Compute product of active grid dimensions only
    n_trials = 1
    for key, values in param_grid.items():
        if key not in inactive:
            n_trials *= len(values)

    return max(1, n_trials)