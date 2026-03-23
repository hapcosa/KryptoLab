#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════
  PATCH: Agregar TP3 a CyberCycleStrategyv3
═══════════════════════════════════════════════════════════════════════

  Archivos a modificar:
    1. strategies/cybercyclev3.py
    2. indicators/incremental_ehlers.py  (IncrementalCyberCycleV3)
    3. optimize/param_dependencies.py

  Lógica:
    - ATR mode:  tp1_size + tp2_size + tp3_size = 1.0 (tp3_size = residuo)
    - Fixed mode: tp1_fixed_size + tp2_fixed_size + tp3_fixed_size = 1.0

  Ejecutar:  python tp3_patch_cybercyclev3.py
  (Aplica los cambios in-place con backups .bak)
═══════════════════════════════════════════════════════════════════════
"""

import re
import shutil
import sys
from pathlib import Path


def patch_file(filepath: str, replacements: list):
    """
    Apply a list of (old, new) replacements to a file.
    Creates a .bak backup before modifying.
    """
    p = Path(filepath)
    if not p.exists():
        print(f"  ⚠  SKIP: {filepath} no encontrado")
        return False

    content = p.read_text()
    shutil.copy2(p, p.with_suffix(p.suffix + '.bak'))

    for i, (old, new) in enumerate(replacements):
        if old not in content:
            print(f"  ⚠  Replacement #{i+1} NOT FOUND in {filepath}")
            print(f"      First 80 chars of old: {old[:80]}...")
            continue
        count = content.count(old)
        if count > 1:
            print(f"  ⚠  Replacement #{i+1} found {count} times — replacing first only")
            content = content.replace(old, new, 1)
        else:
            content = content.replace(old, new)
        print(f"  ✓  Replacement #{i+1} applied")

    p.write_text(content)
    print(f"  ✓  {filepath} patched (backup: {p.suffix}.bak)")
    return True


# ═══════════════════════════════════════════════════════════════════
#  PATCH 1: strategies/cybercyclev3.py
# ═══════════════════════════════════════════════════════════════════

CYBERCYCLEV3_REPLACEMENTS = [

    # ─────────────────────────────────────────────────────────────
    #  1A. ParamDefs — ATR mode: agregar tp2_size y tp3_rr
    # ─────────────────────────────────────────────────────────────
    (
        # OLD — ATR params (2 TP)
        """            ParamDef('tp1_rr', 'float', 2.0, 0.5, 5.0, 0.25),
            ParamDef('tp1_size', 'float', 0.6, 0.1, 0.9, 0.05),
            ParamDef('tp2_rr', 'float', 3.0, 1.0, 10.0, 0.25),""",
        # NEW — ATR params (3 TP)
        """            ParamDef('tp1_rr', 'float', 2.0, 0.5, 5.0, 0.25),
            ParamDef('tp1_size', 'float', 0.4, 0.1, 0.7, 0.05),
            ParamDef('tp2_rr', 'float', 3.0, 1.0, 10.0, 0.25),
            ParamDef('tp2_size', 'float', 0.3, 0.1, 0.6, 0.05),
            ParamDef('tp3_rr', 'float', 5.0, 2.0, 15.0, 0.5),"""
    ),

    # ─────────────────────────────────────────────────────────────
    #  1B. ParamDefs — Fixed mode: agregar tp2_fixed_size y tp3_fixed_pct
    # ─────────────────────────────────────────────────────────────
    (
        # OLD — Fixed params (2 TP)
        """            ParamDef('tp1_fixed_pct', 'float', 1.0, 0.5, 5.0, 0.25),
            ParamDef('tp1_fixed_size', 'float', 0.35, 0.1, 0.9, 0.05),
            ParamDef('tp2_fixed_pct', 'float', 2.0, 1.0, 10.0, 0.5),""",
        # NEW — Fixed params (3 TP)
        """            ParamDef('tp1_fixed_pct', 'float', 1.0, 0.5, 5.0, 0.25),
            ParamDef('tp1_fixed_size', 'float', 0.4, 0.1, 0.7, 0.05),
            ParamDef('tp2_fixed_pct', 'float', 2.0, 1.0, 10.0, 0.5),
            ParamDef('tp2_fixed_size', 'float', 0.3, 0.1, 0.6, 0.05),
            ParamDef('tp3_fixed_pct', 'float', 4.0, 2.0, 20.0, 0.5),"""
    ),

    # ─────────────────────────────────────────────────────────────
    #  1C. _compute_sltp_atr_rr — 3 TPs con sizes
    # ─────────────────────────────────────────────────────────────
    (
        # OLD
        """    def _compute_sltp_atr_rr(self, entry: float, direction: int,
                              atr_val: float) -> dict:
        \"\"\"SL = entry ∓ ATR × mult, TP = entry ± risk × R:R.\"\"\"
        sl_dist = atr_val * self.get_param('sl_atr_mult', 1.5)
        sl = entry - direction * sl_dist
        risk = sl_dist

        tp1_rr = self.get_param('tp1_rr', 2.0)
        tp2_rr = self.get_param('tp2_rr', 3.0)
        tp1_size = self.get_param('tp1_size', 0.6)
        tp2_size = round(1.0 - tp1_size, 8)

        tp1 = entry + direction * risk * tp1_rr
        tp2 = entry + direction * risk * tp2_rr

        return {
            'sl': sl,
            'tp_levels': [tp1, tp2],
            'tp_sizes': [tp1_size, tp2_size],
            'sl_dist': sl_dist,
            'risk': risk,
            'mode': 'slatr_tprr',
        }""",
        # NEW
        """    def _compute_sltp_atr_rr(self, entry: float, direction: int,
                              atr_val: float) -> dict:
        \"\"\"SL = entry ∓ ATR × mult, TP = entry ± risk × R:R. (3 TPs)\"\"\"
        sl_dist = atr_val * self.get_param('sl_atr_mult', 1.5)
        sl = entry - direction * sl_dist
        risk = sl_dist

        tp1_rr = self.get_param('tp1_rr', 2.0)
        tp2_rr = self.get_param('tp2_rr', 3.0)
        tp3_rr = self.get_param('tp3_rr', 5.0)
        tp1_size = self.get_param('tp1_size', 0.4)
        tp2_size = self.get_param('tp2_size', 0.3)
        tp3_size = round(1.0 - tp1_size - tp2_size, 8)
        tp3_size = max(tp3_size, 0.01)  # safety floor

        tp1 = entry + direction * risk * tp1_rr
        tp2 = entry + direction * risk * tp2_rr
        tp3 = entry + direction * risk * tp3_rr

        return {
            'sl': sl,
            'tp_levels': [tp1, tp2, tp3],
            'tp_sizes': [tp1_size, tp2_size, tp3_size],
            'sl_dist': sl_dist,
            'risk': risk,
            'mode': 'slatr_tprr',
        }"""
    ),

    # ─────────────────────────────────────────────────────────────
    #  1D. _compute_sltp_fixed — 3 TPs con sizes
    # ─────────────────────────────────────────────────────────────
    (
        # OLD
        """    def _compute_sltp_fixed(self, entry: float, direction: int) -> dict:
        \"\"\"SL = entry × (1 ∓ sl_pct/100), TP = entry × (1 ± tp_pct/100).\"\"\"
        sl_pct = self.get_param('sl_fixed_pct', 2.0) / 100.0
        tp1_pct = self.get_param('tp1_fixed_pct', 1.0) / 100.0
        tp2_pct = self.get_param('tp2_fixed_pct', 2.0) / 100.0
        tp1_size = self.get_param('tp1_fixed_size', 0.35)
        tp2_size = round(1.0 - tp1_size, 8)

        sl = entry * (1.0 - direction * sl_pct)
        tp1 = entry * (1.0 + direction * tp1_pct)
        tp2 = entry * (1.0 + direction * tp2_pct)

        sl_dist = abs(entry - sl)
        risk = sl_dist

        return {
            'sl': sl,
            'tp_levels': [tp1, tp2],
            'tp_sizes': [tp1_size, tp2_size],
            'sl_dist': sl_dist,
            'risk': risk,
            'mode': 'sltp_fixed',
        }""",
        # NEW
        """    def _compute_sltp_fixed(self, entry: float, direction: int) -> dict:
        \"\"\"SL = entry × (1 ∓ sl_pct/100), TP = entry × (1 ± tp_pct/100). (3 TPs)\"\"\"
        sl_pct = self.get_param('sl_fixed_pct', 2.0) / 100.0
        tp1_pct = self.get_param('tp1_fixed_pct', 1.0) / 100.0
        tp2_pct = self.get_param('tp2_fixed_pct', 2.0) / 100.0
        tp3_pct = self.get_param('tp3_fixed_pct', 4.0) / 100.0
        tp1_size = self.get_param('tp1_fixed_size', 0.4)
        tp2_size = self.get_param('tp2_fixed_size', 0.3)
        tp3_size = round(1.0 - tp1_size - tp2_size, 8)
        tp3_size = max(tp3_size, 0.01)  # safety floor

        sl = entry * (1.0 - direction * sl_pct)
        tp1 = entry * (1.0 + direction * tp1_pct)
        tp2 = entry * (1.0 + direction * tp2_pct)
        tp3 = entry * (1.0 + direction * tp3_pct)

        sl_dist = abs(entry - sl)
        risk = sl_dist

        return {
            'sl': sl,
            'tp_levels': [tp1, tp2, tp3],
            'tp_sizes': [tp1_size, tp2_size, tp3_size],
            'sl_dist': sl_dist,
            'risk': risk,
            'mode': 'sltp_fixed',
        }"""
    ),

    # ─────────────────────────────────────────────────────────────
    #  1E. Metadata en generate_signal — agregar tp3
    # ─────────────────────────────────────────────────────────────
    (
        # OLD
        """                'tp1': tp_levels[0] if tp_levels else 0,
                'tp2': tp_levels[1] if len(tp_levels) > 1 else 0,""",
        # NEW
        """                'tp1': tp_levels[0] if tp_levels else 0,
                'tp2': tp_levels[1] if len(tp_levels) > 1 else 0,
                'tp3': tp_levels[2] if len(tp_levels) > 2 else 0,"""
    ),
]


# ═══════════════════════════════════════════════════════════════════
#  PATCH 2: indicators/incremental_ehlers.py
#  (IncrementalCyberCycleV3._compute  — Build Signal section)
# ═══════════════════════════════════════════════════════════════════

INCREMENTAL_V3_REPLACEMENTS = [
    # ─────────────────────────────────────────────────────────────
    #  2A. IncrementalCyberCycleV3 — Fixed mode block
    # ─────────────────────────────────────────────────────────────
    (
        # OLD — V3 fixed block
        """        if self.p.get('sltp_type', 'slatr_tprr') == 'sltp_fixed':
            sl_d = entry * (1 - direction * self.p.get('sl_fixed_pct', 2.0) / 100)
            tp1 = entry * (1 + direction * self.p.get('tp1_fixed_pct', 1.0) / 100)
            tp2 = entry * (1 + direction * self.p.get('tp2_fixed_pct', 2.0) / 100)
            tp1s = self.p.get('tp1_fixed_size', 0.35)
            sld = abs(entry - sl_d)
        else:
            sld = atr * self.p.get('sl_atr_mult', 1.5)
            sl_d = entry - direction * sld
            tp1s = self.p.get('tp1_size', 0.6)
            tp1 = entry + direction * sld * self.p.get('tp1_rr', 2.0)
            tp2 = entry + direction * sld * self.p.get('tp2_rr', 3.0)

        tp2s = round(1.0 - tp1s, 8)""",
        # NEW — V3 con 3 TPs
        """        if self.p.get('sltp_type', 'slatr_tprr') == 'sltp_fixed':
            sl_d = entry * (1 - direction * self.p.get('sl_fixed_pct', 2.0) / 100)
            tp1 = entry * (1 + direction * self.p.get('tp1_fixed_pct', 1.0) / 100)
            tp2 = entry * (1 + direction * self.p.get('tp2_fixed_pct', 2.0) / 100)
            tp3 = entry * (1 + direction * self.p.get('tp3_fixed_pct', 4.0) / 100)
            tp1s = self.p.get('tp1_fixed_size', 0.4)
            tp2s = self.p.get('tp2_fixed_size', 0.3)
            sld = abs(entry - sl_d)
        else:
            sld = atr * self.p.get('sl_atr_mult', 1.5)
            sl_d = entry - direction * sld
            tp1s = self.p.get('tp1_size', 0.4)
            tp2s = self.p.get('tp2_size', 0.3)
            tp1 = entry + direction * sld * self.p.get('tp1_rr', 2.0)
            tp2 = entry + direction * sld * self.p.get('tp2_rr', 3.0)
            tp3 = entry + direction * sld * self.p.get('tp3_rr', 5.0)

        tp3s = round(max(1.0 - tp1s - tp2s, 0.01), 8)"""
    ),

    # ─────────────────────────────────────────────────────────────
    #  2B. IncrementalCyberCycleV3 — Signal construction
    # ─────────────────────────────────────────────────────────────
    (
        # OLD — Signal with [tp1, tp2]
        """            tp_levels=[tp1, tp2], tp_sizes=[tp1s, tp2s],
            leverage=self.p.get('leverage', 20.0),""",
        # NEW — Signal with [tp1, tp2, tp3]
        """            tp_levels=[tp1, tp2, tp3], tp_sizes=[tp1s, tp2s, tp3s],
            leverage=self.p.get('leverage', 20.0),"""
    ),

    # ─────────────────────────────────────────────────────────────
    #  2C. IncrementalCyberCycleV3 — metadata tp3
    # ─────────────────────────────────────────────────────────────
    (
        # OLD
        """                'tp1': tp1, 'tp2': tp2,
                'be_pct': be_pct, 'trail_pct': tpull if utl else 0,
                'partial_bar': not commit,
                'entry_source': 'hl2',""",
        # NEW
        """                'tp1': tp1, 'tp2': tp2, 'tp3': tp3,
                'be_pct': be_pct, 'trail_pct': tpull if utl else 0,
                'partial_bar': not commit,
                'entry_source': 'hl2',"""
    ),
]


# ═══════════════════════════════════════════════════════════════════
#  PATCH 3: optimize/param_dependencies.py
# ═══════════════════════════════════════════════════════════════════

PARAM_DEPS_REPLACEMENTS = [
    (
        # OLD
        """SLTP_MODE_PARAMS: Dict[str, Set[str]] = {
    'slatr_tprr': {'sl_atr_mult', 'tp1_rr', 'tp1_size', 'tp2_rr'},
    'sltp_fixed': {'sl_fixed_pct', 'tp1_fixed_pct', 'tp1_fixed_size', 'tp2_fixed_pct'},
}""",
        # NEW
        """SLTP_MODE_PARAMS: Dict[str, Set[str]] = {
    'slatr_tprr': {'sl_atr_mult', 'tp1_rr', 'tp1_size', 'tp2_rr', 'tp2_size', 'tp3_rr'},
    'sltp_fixed': {'sl_fixed_pct', 'tp1_fixed_pct', 'tp1_fixed_size', 'tp2_fixed_pct', 'tp2_fixed_size', 'tp3_fixed_pct'},
}"""
    ),
]


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  PATCH: Agregar TP3 a CyberCycleStrategyv3")
    print("=" * 65)

    print("\n[1/3] Patching strategies/cybercyclev3.py ...")
    patch_file("strategies/cybercyclev3.py", CYBERCYCLEV3_REPLACEMENTS)

    print("\n[2/3] Patching indicators/incremental_ehlers.py ...")
    patch_file("indicators/incremental_ehlers.py", INCREMENTAL_V3_REPLACEMENTS)

    print("\n[3/3] Patching optimize/param_dependencies.py ...")
    patch_file("optimize/param_dependencies.py", PARAM_DEPS_REPLACEMENTS)

    print("\n" + "=" * 65)
    print("  DONE — Resumen de nuevos parámetros:")
    print("=" * 65)
    print("""
  ATR mode (slatr_tprr):
    tp1_size  → fracción para TP1  (default 0.4)
    tp2_size  → fracción para TP2  (default 0.3)  ← NUEVO
    tp3_rr    → R:R para TP3       (default 5.0)  ← NUEVO
    tp3_size  = 1.0 - tp1_size - tp2_size          (calculado)

  Fixed mode (sltp_fixed):
    tp1_fixed_size → fracción para TP1  (default 0.4)
    tp2_fixed_size → fracción para TP2  (default 0.3)  ← NUEVO
    tp3_fixed_pct  → % para TP3         (default 4.0)  ← NUEVO
    tp3_fixed_size = 1.0 - tp1_fixed_size - tp2_fixed_size  (calculado)

  Constraint: tp1_size + tp2_size < 1.0 (floor: tp3 ≥ 0.01)
  El optimizador debe respetar: tp1_size + tp2_size ≤ 0.99
""")

    print("  Nuevos comandos de optimización fase 2A (ATR):")
    print("    --exclude-params ahora incluir: tp2_fixed_size,tp3_fixed_pct")
    print("  Nuevos comandos de optimización fase 2B (Fixed):")
    print("    --exclude-params ahora incluir: tp2_size,tp3_rr")
    print()


if __name__ == "__main__":
    main()