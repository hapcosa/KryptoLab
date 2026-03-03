"""
Alpha Floor Diagnostic — ¿MAMA aporta algo o el floor hace todo?

Uso:
  python -m tools.alpha_diagnostic --symbol SOLUSDT --tf 15m \
    --start 2022-06-01 --end 2023-06-01 \
    --mama-fast 0.75 --mama-slow 0.08 --alpha-floor 0.22
"""
import numpy as np
import sys
from data.bitget_client import DataCache
from indicators.ehlers import mama_alpha

def main():
    # Parse args (simplificado)
    args = {}
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '--symbol':
            args['symbol'] = sys.argv[i+1]; i += 2
        elif sys.argv[i] == '--tf':
            args['tf'] = sys.argv[i+1]; i += 2
        elif sys.argv[i] == '--start':
            args['start'] = sys.argv[i+1]; i += 2
        elif sys.argv[i] == '--end':
            args['end'] = sys.argv[i+1]; i += 2
        elif sys.argv[i] == '--mama-fast':
            args['mama_fast'] = float(sys.argv[i+1]); i += 2
        elif sys.argv[i] == '--mama-slow':
            args['mama_slow'] = float(sys.argv[i+1]); i += 2
        elif sys.argv[i] == '--alpha-floor':
            args['floor'] = float(sys.argv[i+1]); i += 2
        else:
            i += 1

    symbol = args.get('symbol', 'SOLUSDT')
    tf = args.get('tf', '15m')
    fast = args.get('mama_fast', 0.75)
    slow = args.get('mama_slow', 0.08)
    floor = args.get('floor', 0.22)

    # Load data
    from data.bitget_client import DataCache
    cache = DataCache()
    df = cache.load(symbol, tf, args.get('start'), args.get('end'))

    if len(df) == 0:
        print(f"❌ No cached data for {symbol} {tf}. Run download first.")
        return

    print(f"📊 Loaded {len(df):,} bars from cache")

    # Convert to numpy
    src = ((df['high'].values + df['low'].values) / 2.0).astype(np.float64)
    n = len(src)

    # Compute raw MAMA alpha (sin floor)
    raw_alpha, period = mama_alpha(src, fast, slow)

    # Apply floor
    floored_alpha = np.maximum(raw_alpha, floor)

    # ── Estadísticas ──
    warmup = 100  # ignorar primeras barras inestables
    raw = raw_alpha[warmup:]
    flr = floored_alpha[warmup:]

    clipped = raw < floor  # barras donde el floor domina
    clipped_pct = np.mean(clipped[warmup:]) * 100

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║           MAMA Alpha Floor Diagnostic                        ║
╠══════════════════════════════════════════════════════════════╣
║  {symbol} | {tf} | {n:,} barras                             
║  MAMA: fast={fast}, slow={slow}                              
║  Alpha floor: {floor}                                        
╠══════════════════════════════════════════════════════════════╣
║                                                              
║  RAW MAMA Alpha (sin floor):                                 
║    Mean:   {np.mean(raw):.4f}                                
║    Median: {np.median(raw):.4f}                              
║    Std:    {np.std(raw):.4f}                                 
║    Min:    {np.min(raw):.4f}                                 
║    Max:    {np.max(raw):.4f}                                 
║    P10:    {np.percentile(raw, 10):.4f}                      
║    P25:    {np.percentile(raw, 25):.4f}                      
║    P50:    {np.percentile(raw, 50):.4f}                      
║    P75:    {np.percentile(raw, 75):.4f}                      
║    P90:    {np.percentile(raw, 90):.4f}                      
║                                                              
║  FLOOR IMPACT:                                               
║    Barras clippeadas por floor: {clipped_pct:.1f}%           
║    (MAMA < {floor} → forzado a {floor})                      
║                                                              
║  AFTER FLOOR:                                                
║    Mean:   {np.mean(flr):.4f}                                
║    Median: {np.median(flr):.4f}                              
║    Std:    {np.std(flr):.4f}                                 
║                                                              
╠══════════════════════════════════════════════════════════════╣
║  VEREDICTO:                                                  """)

    if clipped_pct > 90:
        print(f"║  🔴 MAMA es IRRELEVANTE ({clipped_pct:.0f}% clippeado)")
        print(f"║     El floor={floor} actúa como manual_alpha={floor}")
        print(f"║     → MAMA no aporta adaptividad, es un alpha fijo disfrazado")
    elif clipped_pct > 60:
        print(f"║  🟡 MAMA PARCIALMENTE DOMINADA ({clipped_pct:.0f}% clippeado)")
        print(f"║     El floor domina la mayoría del tiempo, pero MAMA")
        print(f"║     aporta algo en {100-clipped_pct:.0f}% de las barras (momentos rápidos)")
    else:
        print(f"║  🟢 MAMA ES ACTIVA ({100-clipped_pct:.0f}% de barras usan MAMA real)")
        print(f"║     El floor solo interviene en {clipped_pct:.0f}% de las barras")

    print(f"""║                                                              
╠══════════════════════════════════════════════════════════════╣
║  TESTS RECOMENDADOS:                                         
║                                                              
║  A) Backtest con trial original (MAMA + floor={floor})       
║  B) Backtest MAMA sin floor: --alpha-floor 0.0               
║     → Si B << A, el performance viene del floor, no de MAMA  
║  C) Backtest manual con alpha={floor}:                       
║     --alpha-method manual --manual-alpha {floor}             
║     → Si C ≈ A, MAMA no aporta nada sobre un alpha fijo     
║                                                              
║  Si B << A y C ≈ A → MAMA es inútil, usar manual            
║  Si B ≈ A → MAMA funciona sola, el floor es innecesario     
║  Si B < A pero C < A → MAMA aporta marginalmente            
╚══════════════════════════════════════════════════════════════╝
""")

    # Histogram text-based
    print("  Distribución de MAMA raw alpha:")
    bins = np.linspace(slow, fast, 20)
    counts, edges = np.histogram(raw, bins=bins)
    max_count = max(counts) if max(counts) > 0 else 1
    for j in range(len(counts)):
        bar_len = int(40 * counts[j] / max_count)
        marker = " ◄── floor" if edges[j] <= floor < edges[j+1] else ""
        print(f"  {edges[j]:.2f}-{edges[j+1]:.2f} | {'█' * bar_len} {counts[j]}{marker}")

if __name__ == '__main__':
    main()