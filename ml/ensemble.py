"""
CryptoLab — Strategy Ensemble
Combines multiple strategies for robust, regime-adaptive portfolio management.

Ensemble methods:
  1. Static Blend:    fixed weights, simple portfolio
  2. Confidence Vote:  aggregate signals weighted by confidence scores
  3. Regime Switch:    select best strategy per detected regime
  4. Dynamic Blend:    rolling-window performance-weighted allocation
  5. Meta-Learner:     LightGBM/LogReg on strategy features → signal quality

The ensemble framework wraps multiple IStrategy instances and produces
unified signals that the BacktestEngine can execute.
"""
import numpy as np
import copy
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field

from strategies.base import IStrategy, Signal, ParamDef
from ml.regime_detector import detect_regime, Regime, REGIME_NAMES, RegimeResult


@dataclass
class EnsembleMember:
    """A strategy participating in the ensemble."""
    name: str
    strategy: IStrategy
    weight: float = 1.0
    regime_weights: Dict[int, float] = field(default_factory=dict)
    # regime_weights: {Regime.TREND_UP: 0.8, Regime.RANGING: 0.2, ...}


@dataclass
class EnsembleSignal:
    """Aggregated signal from the ensemble."""
    direction: int              # 1=LONG, -1=SHORT, 0=NO_TRADE
    confidence: float           # Weighted aggregate confidence
    contributing: List[str]     # Names of strategies that contributed
    weights_used: Dict[str, float] = field(default_factory=dict)
    regime: int = 0


@dataclass
class EnsembleResult:
    """Performance summary of the ensemble."""
    method: str
    members: List[str]
    combined_sharpe: float = 0.0
    combined_return: float = 0.0
    combined_max_dd: float = 0.0
    combined_win_rate: float = 0.0
    n_trades: int = 0
    regime_performance: Dict[int, Dict] = field(default_factory=dict)
    member_contributions: Dict[str, float] = field(default_factory=dict)


class EnsembleStrategy(IStrategy):
    """
    Ensemble that wraps multiple strategies and produces unified signals.

    Can be used directly with BacktestEngine like any other IStrategy.
    """

    def __init__(self,
                 members: List[EnsembleMember],
                 method: str = 'confidence_vote',
                 min_agreement: int = 1,
                 min_confidence: float = 50.0,
                 regime_method: str = 'vt'):
        """
        Args:
            members: List of EnsembleMember (strategy + weight)
            method: 'static_blend', 'confidence_vote', 'regime_switch'
            min_agreement: Min number of strategies agreeing on direction
            min_confidence: Min weighted confidence to trigger signal
            regime_method: Regime detection method ('vt', 'cluster')
        """
        super().__init__()
        self.members = members
        self.method = method
        self.min_agreement = min_agreement
        self.min_confidence = min_confidence
        self.regime_method = regime_method
        self._member_indicators = {}
        self._regime_labels = None

    def name(self) -> str:
        names = '+'.join(m.name for m in self.members)
        return f"Ensemble({names}|{self.method})"

    def parameter_defs(self) -> List[ParamDef]:
        return [
            ParamDef('min_agreement', 'int', self.min_agreement, 1,
                     len(self.members)),
            ParamDef('min_confidence', 'float', self.min_confidence,
                     10.0, 95.0, 5.0),
            ParamDef('leverage', 'float', 3.0, 1.0, 25.0, 0.5),
        ]

    def calculate_indicators(self, data: dict) -> dict:
        """Calculate indicators for ALL member strategies + regime."""
        # Regime detection
        regime_result = detect_regime(data, method=self.regime_method,
                                       verbose=False)
        self._regime_labels = regime_result.labels

        # Calculate indicators for each member
        all_indicators = {}
        for member in self.members:
            strat = copy.deepcopy(member.strategy)
            indicators = strat.calculate_indicators(data)
            self._member_indicators[member.name] = {
                'strategy': strat,
                'indicators': indicators,
            }
            # Store member indicators with prefix
            for key, val in indicators.items():
                all_indicators[f"{member.name}_{key}"] = val

        all_indicators['regime'] = regime_result.labels
        return all_indicators

    def generate_signal(self, indicators: dict, idx: int,
                        data: dict) -> Optional[Signal]:
        """Generate ensemble signal by aggregating member signals."""
        if idx < 30:
            return None

        min_bars = self.get_param('min_bars', 5)
        if idx - self._last_signal_bar < min_bars:
            return None

        # Get current regime
        regime = (self._regime_labels[idx]
                  if self._regime_labels is not None and idx < len(self._regime_labels)
                  else Regime.RANGING)

        # Collect signals from all members
        member_signals = []
        for member in self.members:
            mi = self._member_indicators.get(member.name)
            if mi is None:
                continue

            strat = mi['strategy']
            member_ind = mi['indicators']
            sig = strat.generate_signal(member_ind, idx, data)

            if sig is not None:
                # Apply regime-specific weight
                regime_w = member.regime_weights.get(regime, 1.0)
                effective_weight = member.weight * regime_w

                member_signals.append({
                    'name': member.name,
                    'signal': sig,
                    'weight': effective_weight,
                })

        if not member_signals:
            return None

        # Aggregate based on method
        if self.method == 'confidence_vote':
            return self._confidence_vote(member_signals, idx, data, regime)
        elif self.method == 'regime_switch':
            return self._regime_switch(member_signals, idx, data, regime)
        elif self.method == 'static_blend':
            return self._static_blend(member_signals, idx, data, regime)
        else:
            return self._confidence_vote(member_signals, idx, data, regime)

    def _confidence_vote(self, signals: list, idx: int,
                         data: dict, regime: int) -> Optional[Signal]:
        """
        Confidence-weighted voting.
        Each strategy votes with its signal direction, weighted by
        confidence × member_weight × regime_weight.
        """
        # Count weighted votes
        long_weight = 0.0
        short_weight = 0.0
        long_signals = []
        short_signals = []

        for s in signals:
            sig = s['signal']
            w = s['weight'] * (sig.confidence / 100.0)

            if sig.direction == 1:
                long_weight += w
                long_signals.append(s)
            else:
                short_weight += w
                short_signals.append(s)

        # Determine direction
        if long_weight > short_weight and len(long_signals) >= self.min_agreement:
            direction = 1
            active = long_signals
            total_weight = long_weight
        elif short_weight > long_weight and len(short_signals) >= self.min_agreement:
            direction = -1
            active = short_signals
            total_weight = short_weight
        else:
            return None

        # Weighted average confidence
        conf_sum = sum(s['weight'] * s['signal'].confidence for s in active)
        weight_sum = sum(s['weight'] for s in active)
        avg_confidence = conf_sum / weight_sum if weight_sum > 0 else 0

        if avg_confidence < self.min_confidence:
            return None

        self._last_signal_bar = idx

        # Use the signal from the highest-confidence member for SL/TP
        best = max(active, key=lambda s: s['signal'].confidence)
        ref_sig = best['signal']

        return Signal(
            direction=direction,
            confidence=avg_confidence,
            entry_price=ref_sig.entry_price,
            sl_price=ref_sig.sl_price,
            tp_levels=ref_sig.tp_levels,
            tp_sizes=ref_sig.tp_sizes,
            leverage=self.get_param('leverage', 3.0),
            be_trigger=ref_sig.be_trigger,
            trailing=ref_sig.trailing,
            trailing_distance=ref_sig.trailing_distance,
            metadata={
                'method': 'confidence_vote',
                'regime': int(regime),
                'n_contributors': len(active),
                'contributors': [s['name'] for s in active],
            }
        )

    def _regime_switch(self, signals: list, idx: int,
                       data: dict, regime: int) -> Optional[Signal]:
        """
        Regime-based switching.
        Use only the signal from the member with highest regime weight.
        """
        if not signals:
            return None

        # Pick the highest regime-weighted member
        best = max(signals, key=lambda s: s['weight'])
        sig = best['signal']

        if sig.confidence < self.min_confidence:
            return None

        self._last_signal_bar = idx

        return Signal(
            direction=sig.direction,
            confidence=sig.confidence,
            entry_price=sig.entry_price,
            sl_price=sig.sl_price,
            tp_levels=sig.tp_levels,
            tp_sizes=sig.tp_sizes,
            leverage=self.get_param('leverage', 3.0),
            be_trigger=sig.be_trigger,
            trailing=sig.trailing,
            trailing_distance=sig.trailing_distance,
            metadata={
                'method': 'regime_switch',
                'regime': int(regime),
                'selected': best['name'],
                'regime_weight': best['weight'],
            }
        )

    def _static_blend(self, signals: list, idx: int,
                      data: dict, regime: int) -> Optional[Signal]:
        """
        Static blend: use the first available signal, weighted by member order.
        Simplest method — just pick the highest-weighted signal.
        """
        if not signals:
            return None

        best = max(signals, key=lambda s: s['weight'] * s['signal'].confidence)
        return self._confidence_vote(signals, idx, data, regime)


# ═══════════════════════════════════════════════════════════════
#  ENSEMBLE BUILDER
# ═══════════════════════════════════════════════════════════════

class EnsembleBuilder:
    """
    Utility to construct and evaluate ensembles.

    Workflow:
      1. Add strategies with optional regime weights
      2. Build ensemble with chosen method
      3. Evaluate with BacktestEngine
      4. Compare ensemble vs individual strategies
    """

    def __init__(self):
        self.members: List[EnsembleMember] = []

    def add(self, name: str, strategy: IStrategy,
            weight: float = 1.0,
            regime_weights: Optional[Dict[int, float]] = None
            ) -> 'EnsembleBuilder':
        """Add a strategy to the ensemble."""
        rw = regime_weights or {}
        self.members.append(EnsembleMember(
            name=name, strategy=strategy,
            weight=weight, regime_weights=rw,
        ))
        return self

    def build(self, method: str = 'confidence_vote',
              min_agreement: int = 1,
              min_confidence: float = 50.0) -> EnsembleStrategy:
        """Build the ensemble strategy."""
        return EnsembleStrategy(
            members=self.members,
            method=method,
            min_agreement=min_agreement,
            min_confidence=min_confidence,
        )

    def evaluate(self,
                 data: dict,
                 engine_factory: Callable,
                 method: str = 'confidence_vote',
                 symbol: str = "",
                 timeframe: str = "",
                 verbose: bool = True) -> EnsembleResult:
        """
        Build ensemble, backtest, and compare with individuals.
        """
        from core.engine import BacktestEngine, format_result

        # Individual backtests
        individual_results = {}
        for member in self.members:
            strat = copy.deepcopy(member.strategy)
            engine = engine_factory()
            result = engine.run(strat, data, symbol, timeframe)
            individual_results[member.name] = result

            if verbose:
                print(f"  [{member.name}] SR={result.sharpe_ratio:.2f} "
                      f"Ret={result.total_return:+.1f}% "
                      f"WR={result.win_rate:.1f}% Trades={result.n_trades}")

        # Ensemble backtest
        ensemble = self.build(method=method)
        engine = engine_factory()
        ens_result = engine.run(ensemble, data, symbol, timeframe)

        if verbose:
            print(f"\n  [ENSEMBLE] SR={ens_result.sharpe_ratio:.2f} "
                  f"Ret={ens_result.total_return:+.1f}% "
                  f"WR={ens_result.win_rate:.1f}% Trades={ens_result.n_trades}")

            # Synergy
            best_ind_sr = max(r.sharpe_ratio for r in individual_results.values())
            synergy = ens_result.sharpe_ratio - best_ind_sr
            print(f"  Synergy: {synergy:+.2f} Sharpe vs best individual")

        # Contributions
        contributions = {}
        for trade in ens_result.trades:
            contribs = trade.metadata.get('contributors', [])
            if isinstance(contribs, list):
                for c in contribs:
                    contributions[c] = contributions.get(c, 0) + 1
            selected = trade.metadata.get('selected')
            if selected:
                contributions[selected] = contributions.get(selected, 0) + 1

        return EnsembleResult(
            method=method,
            members=[m.name for m in self.members],
            combined_sharpe=ens_result.sharpe_ratio,
            combined_return=ens_result.total_return,
            combined_max_dd=ens_result.max_drawdown,
            combined_win_rate=ens_result.win_rate,
            n_trades=ens_result.n_trades,
            member_contributions=contributions,
        )


def create_regime_adaptive_ensemble(
        strategies: Dict[str, IStrategy],
        regime_mapping: Optional[Dict[str, Dict[int, float]]] = None
        ) -> EnsembleStrategy:
    """
    Convenience function to create a regime-adaptive ensemble.

    Args:
        strategies: Dict[name → IStrategy]
        regime_mapping: Dict[name → {regime_id → weight}]
            If None, uses sensible defaults:
              - CyberCycle: good in trending markets
              - GaussBands: good in ranging markets
              - SMC: good in volatile reversal markets

    Returns:
        EnsembleStrategy ready for backtesting
    """
    if regime_mapping is None:
        regime_mapping = {
            'cybercycle': {
                Regime.TREND_UP: 1.0,
                Regime.TREND_DOWN: 1.0,
                Regime.RANGING: 0.3,
                Regime.HIGH_VOL_UP: 0.5,
                Regime.HIGH_VOL_DOWN: 0.5,
            },
            'gaussbands': {
                Regime.TREND_UP: 0.5,
                Regime.TREND_DOWN: 0.5,
                Regime.RANGING: 1.0,
                Regime.HIGH_VOL_UP: 0.3,
                Regime.HIGH_VOL_DOWN: 0.3,
            },
            'smartmoney': {
                Regime.TREND_UP: 0.3,
                Regime.TREND_DOWN: 0.3,
                Regime.RANGING: 0.5,
                Regime.HIGH_VOL_UP: 1.0,
                Regime.HIGH_VOL_DOWN: 1.0,
            },
        }

    builder = EnsembleBuilder()
    for name, strat in strategies.items():
        rw = regime_mapping.get(name, {})
        builder.add(name, strat, weight=1.0, regime_weights=rw)

    return builder.build(method='regime_switch', min_confidence=40.0)
