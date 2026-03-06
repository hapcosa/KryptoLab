"""
CryptoLab — iTrend Strategy v1.0

Ehlers Instantaneous Trendline as standalone trading strategy.

Señal principal:
  - LONG  cuando iTrend cambia a alcista: it[i] > it[i-2] AND it[i-1] <= it[i-3]
  - SHORT cuando iTrend cambia a bajista: it[i] < it[i-2] AND it[i-1] >= it[i-3]

Filtros:
  - Slope mínimo: evita cruces planos/dudosos sin lag adicional
  - Volumen: confirma participación en el movimiento
  - Extensión: no entrar si precio está demasiado alejado del iTrend

Dual SL/TP: slatr_tprr | sltp_fixed
"""
import numpy as np
from typing import Optional, List

from strategies.base import IStrategy, Signal, ParamDef
from indicators.ehlers import itrend
from indicators.common import atr, volume_ratio


# ═══════════════════════════════════════════════════════════════
#  CONFIDENCE — iTrend Strategy
#
#  Componentes (max 100):
#    Dirección del cruce:   30  — señal base (cambio de dirección iTrend)
#    Slope quality:         25  — pendiente suficientemente pronunciada
#    Volume:                25  — volumen por encima del promedio
#    Extension OK:          20  — precio no muy alejado del iTrend
#  ─────────────────────────────────────────────────────────────
#  Lógica: el slope y la extensión reemplazan la confirmación de
#  barras adicionales — filtran calidad sin añadir lag.
# ═══════════════════════════════════════════════════════════════

def compute_confidence_itrend(
    is_buy: bool,
    slope_ok: bool,
    volume_ok: bool,
    extension_ok: bool,
    use_vol: bool,
) -> float:
    """
    Confidence scoring iTrend v1.0.

        Dirección cruce:  30  (siempre presente si hay señal)
        Slope quality:    25  (pendiente >= slope_min_pct)
        Volume:           25  (vol_ratio >= volume_mult)
        Extension OK:     20  (precio dentro de max_extension_pct del iTrend)
        ──────────────────
        Total max:       100
    """
    conf = 30.0  # cruce confirmado — base fija
    conf += 25.0 if slope_ok else 0.0
    conf += 25.0 if (volume_ok if use_vol else True) else 0.0
    conf += 20.0 if extension_ok else 0.0
    return min(conf, 100.0)


class ITrendStrategy(IStrategy):

    def name(self) -> str:
        return "iTrend v1.0"

    def parameter_defs(self) -> List[ParamDef]:
        return [
            # ── iTrend core ─────────────────────────────────────────
            ParamDef('itrend_alpha', 'float', 0.07, 0.01, 0.30, 0.01),

            # ── Signal params ───────────────────────────────────────
            # min_bars: barras mínimas entre señales (evita reentradas rápidas)
            ParamDef('min_bars', 'int', 10, 3, 48, 1),

            # Umbral mínimo de confianza para abrir posición
            ParamDef('confidence_min', 'float', 55.0, 30.0, 95.0, 5.0),

            # ── Slope filter ────────────────────────────────────────
            # Pendiente mínima del iTrend expresada como % del precio
            # it[i] - it[i-2] >= entry_price * slope_min_pct / 100
            # 0.0 = desactivado (cualquier cruce cuenta)
            ParamDef('slope_min_pct', 'float', 0.05, 0.0, 0.5, 0.01),

            # ── Extension filter ────────────────────────────────────
            # No entrar si |close - itrend| > max_extension_pct% del precio
            # Evita perseguir el precio cuando ya se alejó demasiado
            # 0.0 = desactivado
            ParamDef('max_extension_pct', 'float', 2.0, 0.0, 8.0, 0.25),

            # ── Volume filter ───────────────────────────────────────
            ParamDef('use_volume', 'bool', True),
            ParamDef('volume_mult', 'float', 1.2, 0.5, 4.0, 0.1),

            # ═══════════════════════════════════════════════════════
            #  SL/TP MODE SELECTOR
            # ═══════════════════════════════════════════════════════
            #  slatr_tprr → SL basado en ATR, TP basado en R:R
            #  sltp_fixed → SL y TP como % fijo del entry price
            # ───────────────────────────────────────────────────────
            ParamDef('sltp_type', 'categorical', 'slatr_tprr',
                     options=['slatr_tprr', 'sltp_fixed']),

            # ── ATR mode (slatr_tprr) ────────────────────────────
            ParamDef('leverage',    'float', 10.0,  3.0, 30.0, 1.0),
            ParamDef('sl_atr_mult', 'float',  2.0,  0.5,  5.0, 0.25),
            ParamDef('tp1_rr',      'float',  2.0,  0.5,  6.0, 0.25),
            ParamDef('tp1_size',    'float',  0.6,  0.1,  0.9, 0.05),
            ParamDef('tp2_rr',      'float',  4.0,  1.0, 10.0, 0.25),

            # ── Fixed mode (sltp_fixed) ──────────────────────────
            ParamDef('sl_fixed_pct',   'float', 2.0, 0.3,  6.0, 0.1),
            ParamDef('tp1_fixed_pct',  'float', 3.0, 0.5,  8.0, 0.25),
            ParamDef('tp1_fixed_size', 'float', 0.6, 0.1,  0.9, 0.05),
            ParamDef('tp2_fixed_pct',  'float', 5.0, 1.0, 15.0, 0.5),

            # ── Break-even ──────────────────────────────────────────
            # be_pct=0 → desactivado
            ParamDef('be_pct', 'float', 1.0, 0.0, 3.0, 0.1),

            # ── Trailing stop ───────────────────────────────────────
            ParamDef('use_trailing',      'bool',  True),
            ParamDef('trail_activate_pct','float', 1.5, 0.0, 5.0, 0.25),
            ParamDef('trail_pullback_pct','float', 0.8, 0.1, 3.0, 0.10),
        ]

    # ─────────────────────────────────────────────────────────────
    #  INDICATORS
    # ─────────────────────────────────────────────────────────────

    def calculate_indicators(self, data: dict) -> dict:
        """Calculate iTrend indicators."""
        close = data['close']
        vol   = data['volume']
        n     = len(close)

        # ── Instantaneous Trendline ─────────────────────────────
        it = itrend(close, self.get_param('itrend_alpha', 0.07))

        # Dirección: bull cuando it[i] > it[i-2]
        bull_trend = np.zeros(n, dtype=bool)
        bear_trend = np.zeros(n, dtype=bool)
        for i in range(2, n):
            bull_trend[i] = it[i] > it[i - 2]
            bear_trend[i] = it[i] < it[i - 2]

        # ── Cruce de dirección ──────────────────────────────────
        # bull_cross: transición bear→bull (primera barra que it[i]>it[i-2]
        #             mientras it[i-1]<=it[i-3])
        bull_cross = np.zeros(n, dtype=bool)
        bear_cross = np.zeros(n, dtype=bool)
        for i in range(4, n):
            bull_cross[i] = bull_trend[i] and not bull_trend[i - 1]
            bear_cross[i] = bear_trend[i] and not bear_trend[i - 1]

        # ── Slope (pendiente 2 barras, normalizada por precio) ──
        # slope[i] = (it[i] - it[i-2]) / it[i-2] * 100  (en %)
        slope = np.zeros(n)
        for i in range(2, n):
            if it[i - 2] != 0:
                slope[i] = (it[i] - it[i - 2]) / abs(it[i - 2]) * 100.0

        # ── Extension (distancia close-itrend en %) ─────────────
        extension = np.zeros(n)
        for i in range(n):
            if it[i] != 0:
                extension[i] = abs(close[i] - it[i]) / abs(it[i]) * 100.0

        # ── Volume ratio ────────────────────────────────────────
        vol_ratio = volume_ratio(vol, 20)

        # ── ATR ─────────────────────────────────────────────────
        atr_vals = atr(data['high'], data['low'], close, 14)

        return {
            'itrend':     it,
            'bull_trend': bull_trend,
            'bear_trend': bear_trend,
            'bull_cross': bull_cross,
            'bear_cross': bear_cross,
            'slope':      slope,
            'extension':  extension,
            'vol_ratio':  vol_ratio,
            'atr':        atr_vals,
        }

    # ─────────────────────────────────────────────────────────────
    #  SIGNAL GENERATION
    # ─────────────────────────────────────────────────────────────

    def generate_signal(self, ind: dict, idx: int,
                        data: dict) -> Optional[Signal]:
        """Generate iTrend signal at bar idx."""
        if idx < 10:
            return None

        bull_cross = ind['bull_cross'][idx]
        bear_cross = ind['bear_cross'][idx]

        if not (bull_cross or bear_cross):
            return None

        # ── Min bars between signals ────────────────────────────
        if idx - self._last_signal_bar < self.get_param('min_bars', 10):
            return None

        is_buy      = bull_cross
        direction   = 1 if is_buy else -1
        entry       = data['close'][idx]
        use_vol     = self.get_param('use_volume', True)

        # ── Slope filter ────────────────────────────────────────
        slope_min = self.get_param('slope_min_pct', 0.05)
        raw_slope = ind['slope'][idx]
        # Para longs queremos pendiente positiva, para shorts negativa
        directional_slope = raw_slope if is_buy else -raw_slope
        slope_ok = (directional_slope >= slope_min) if slope_min > 0.0 else True

        # ── Volume filter ────────────────────────────────────────
        vol_mult  = self.get_param('volume_mult', 1.2)
        volume_ok = ind['vol_ratio'][idx] >= vol_mult

        # ── Extension filter ─────────────────────────────────────
        max_ext = self.get_param('max_extension_pct', 2.0)
        extension_ok = (ind['extension'][idx] <= max_ext) if max_ext > 0.0 else True

        # ── Confidence ───────────────────────────────────────────
        conf = compute_confidence_itrend(
            is_buy=is_buy,
            slope_ok=slope_ok,
            volume_ok=volume_ok,
            extension_ok=extension_ok,
            use_vol=use_vol,
        )

        if conf < self.get_param('confidence_min', 55.0):
            return None

        self._last_signal_bar = idx

        # ── SL/TP — dual mode ────────────────────────────────────
        atr_val   = ind['atr'][idx]
        sltp_type = self.get_param('sltp_type', 'slatr_tprr')

        if sltp_type == 'sltp_fixed':
            sl_pct   = self.get_param('sl_fixed_pct',   2.0) / 100.0
            tp1_pct  = self.get_param('tp1_fixed_pct',  3.0) / 100.0
            tp2_pct  = self.get_param('tp2_fixed_pct',  5.0) / 100.0
            tp1_size = self.get_param('tp1_fixed_size', 0.6)
            sl       = entry * (1.0 - direction * sl_pct)
            tp1      = entry * (1.0 + direction * tp1_pct)
            tp2      = entry * (1.0 + direction * tp2_pct)
            sl_dist  = abs(entry - sl)
        else:  # slatr_tprr
            sl_dist  = atr_val * self.get_param('sl_atr_mult', 2.0)
            sl       = entry - direction * sl_dist
            tp1_size = self.get_param('tp1_size', 0.6)
            tp1      = entry + direction * sl_dist * self.get_param('tp1_rr', 2.0)
            tp2      = entry + direction * sl_dist * self.get_param('tp2_rr', 4.0)

        tp2_size  = round(1.0 - tp1_size, 8)
        tp_levels = [tp1, tp2]
        tp_sizes  = [tp1_size, tp2_size]

        # ── Break-even ───────────────────────────────────────────
        be_pct = self.get_param('be_pct', 1.0)
        if be_pct > 0.0:
            be_trigger = entry + direction * entry * (be_pct / 100.0)
        else:
            be_trigger = 0.0

        # ── Trailing stop ────────────────────────────────────────
        use_trailing = self.get_param('use_trailing', True)
        if use_trailing:
            trail_activate_pct = self.get_param('trail_activate_pct', 1.5)
            trail_pullback_pct = self.get_param('trail_pullback_pct', 0.8)
            trailing_distance  = entry * (trail_pullback_pct / 100.0)

            trail_activation_price = entry + direction * entry * (trail_activate_pct / 100.0)
            if be_pct > 0.0:
                dist_be    = abs(be_trigger - entry)
                dist_trail = abs(trail_activation_price - entry)
                be_trigger = be_trigger if dist_be <= dist_trail else trail_activation_price
            else:
                be_trigger = trail_activation_price
        else:
            trailing_distance = 0.0

        return Signal(
            direction=direction,
            confidence=conf,
            entry_price=entry,
            sl_price=sl,
            tp_levels=tp_levels,
            tp_sizes=tp_sizes,
            leverage=self.get_param('leverage', 10.0),
            be_trigger=be_trigger,
            trailing=use_trailing,
            trailing_distance=trailing_distance,
            metadata={
                'sltp_mode':     sltp_type,
                'itrend':        float(ind['itrend'][idx]),
                'slope_pct':     float(ind['slope'][idx]),
                'extension_pct': float(ind['extension'][idx]),
                'vol_ratio':     float(ind['vol_ratio'][idx]),
                'slope_ok':      slope_ok,
                'volume_ok':     volume_ok,
                'extension_ok':  extension_ok,
                'sl_dist_atr':   sl_dist / atr_val if atr_val > 0 else 0,
                'tp1':           tp1,
                'tp2':           tp2,
                'be_pct':        be_pct,
                'trail_pct':     self.get_param('trail_pullback_pct', 0.8) if use_trailing else 0,
            }
        )