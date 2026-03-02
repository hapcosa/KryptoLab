Optimización por Fases para CyberCycle v7.0
Este documento describe una metodología de optimización por fases para la estrategia CyberCycle v7.0 en CryptoLab. El objetivo es evitar que el optimizador ajuste simultáneamente demasiados parámetros y permitir un análisis aislado de cada componente, reduciendo el riesgo de sobreoptimización y mejorando la interpretabilidad de los resultados.

Cada fase produce un archivo JSON con los mejores parámetros, que se utiliza como punto de partida (--params-file) en la fase siguiente. De esta forma se construye incrementalmente una configuración robusta.

📋 Convenciones en los comandos
--exclude-params → excluye una lista de parámetros; el optimizador no los modificará (se mantienen con su valor actual, ya sea por defecto o desde el archivo cargado).

--optimize-params → especifica una lista de parámetros a optimizar; el resto se mantienen fijos.

--params-file → carga un archivo JSON con parámetros previos (por ejemplo, de una fase anterior).

--objective monthly_robust → se utiliza por su equilibrio entre rendimiento y consistencia; puede cambiarse por cualquier otro objetivo soportado (sharpe, composite, weekly_robust, etc.).

--n-trials → número de pruebas; ajustar según el tiempo disponible (200‑300 suele ser suficiente para espacios reducidos).

⚠️ Importante: Asegúrate de tener los datos descargados para el símbolo y timeframe correspondiente antes de ejecutar los comandos.

Fase 1 – Señal y filtros básicos
En esta fase se mantienen fijos todos los parámetros de risk management (SL/TP, trailing, apalancamiento, etc.) y se optimiza únicamente la lógica de generación de señal: método alpha, suavizado, trigger, filtros de tendencia/volumen, etc.

1A. Señal + filtros completos (risk management fijo)
bash
python cli.py optimize --strategy cybercycle --symbol SOLUSDT --tf 1h \
  --method bayesian --objective monthly_robust --n-trials 300 \
  --exclude-params sl_atr_mult,tp1_rr,tp1_size,tp2_rr,sl_fixed_pct,tp1_fixed_pct,tp1_fixed_size,tp2_fixed_pct,sltp_type,leverage,be_pct,use_trailing,trail_activate_pct,trail_pullback_pct
Propósito: Encontrar la mejor configuración de la señal sin que el optimizador pueda compensar una señal débil con stops muy ajustados o TPs muy generosos.

1B. Solo método alpha y ciclo (sin filtros de confianza)
Si se quiere aislar aún más el núcleo de la señal:

bash
python cli.py optimize --strategy cybercycle --symbol SOLUSDT --tf 1h \
  --method bayesian --objective monthly_robust --n-trials 200 \
  --optimize-params alpha_method,manual_alpha,alpha_floor,mama_fast,mama_slow,kal_process_noise,kal_meas_noise,kal_alpha_fast,kal_alpha_slow,kal_sensitivity,trigger_ema
Nota: Este comando asume que el resto de parámetros (filtros, risk) ya están fijados (por defecto o desde un JSON previo). Se recomienda ejecutarlo después de la fase 1A, usando --params-file con el resultado de 1A.

Fase 2 – Risk management
Una vez que tenemos una buena configuración de señal (guardada, por ejemplo, como params_cybercycle_SOLUSDT_1h_fase1.json), cargamos esos parámetros y optimizamos exclusivamente los relacionados con la gestión de riesgo.

2A. Solo ATR + R:R (excluir fixed)
bash
python cli.py optimize --strategy cybercycle --symbol SOLUSDT --tf 1h \
  --method bayesian --objective monthly_robust --n-trials 200 \
  --params-file output/params_cybercycle_SOLUSDT_1h_fase1.json \
  --exclude-params sl_fixed_pct,tp1_fixed_pct,tp1_fixed_size,tp2_fixed_pct,alpha_method,manual_alpha,alpha_floor,mama_fast,mama_slow,kal_process_noise,kal_meas_noise,kal_alpha_fast,kal_alpha_slow,kal_sensitivity
Propósito: Optimizar SL/TP basado en ATR y relaciones R:R, manteniendo fijos los parámetros de señal y los stops fijos.

2B. Solo fixed % (excluir ATR/RR)
bash
python cli.py optimize --strategy cybercycle --symbol SOLUSDT --tf 1h \
  --method bayesian --objective monthly_robust --n-trials 200 \
  --params-file output/params_cybercycle_SOLUSDT_1h_fase1.json \
  --exclude-params sl_atr_mult,tp1_rr,tp1_size,tp2_rr,alpha_method,manual_alpha,alpha_floor,mama_fast,mama_slow,kal_process_noise,kal_meas_noise,kal_alpha_fast,kal_alpha_slow,kal_sensitivity
Propósito: Optimizar SL/TP fijos (porcentajes), manteniendo fijos los parámetros de señal y los ATR.

2C. Todo risk management junto (ATR + fixed + trailing + BE + leverage)
bash
python cli.py optimize --strategy cybercycle --symbol SOLUSDT --tf 1h \
  --method bayesian --objective monthly_robust --n-trials 300 \
  --params-file output/params_cybercycle_SOLUSDT_1h_fase1.json \
  --optimize-params sltp_type,sl_atr_mult,tp1_rr,tp1_size,tp2_rr,sl_fixed_pct,tp1_fixed_pct,tp1_fixed_size,tp2_fixed_pct,leverage,be_pct,use_trailing,trail_activate_pct,trail_pullback_pct
Propósito: Una vez definida la señal, se busca la mejor combinación de gestión de riesgo, incluyendo ambos modos (ATR y fijo) para que el optimizador elija.

Fase 3 – Filtros de calidad
Con la señal y el risk ya definidos (archivo params_fase2.json), ahora se afinan los filtros que determinan qué trades se ejecutan realmente.

bash
python cli.py optimize --strategy cybercycle --symbol SOLUSDT --tf 1h \
  --method bayesian --objective monthly_robust --n-trials 200 \
  --params-file output/params_cybercycle_SOLUSDT_1h_fase2.json \
  --optimize-params confidence_min,cycle_str_pctile,cycle_str_lookback,use_trend,use_volume,volume_mult,ob_level,os_level,min_bars
Propósito: Determinar qué tan estrictos deben ser los filtros (confianza mínima, amplitud del ciclo, uso de tendencia/volumen, niveles OB/OS, separación entre señales) para maximizar la calidad de las operaciones.

Fase 4 – Exploración completa
Una vez que cada grupo ha sido optimizado por separado, se puede realizar una exploración final con todos los parámetros abiertos (excepto quizás algunos booleanos triviales) para refinar la combinación global.

4A. Full exploration (excluyendo solo bools obvios)
bash
python cli.py optimize --strategy cybercycle --symbol SOLUSDT --tf 1h \
  --method bayesian --objective monthly_robust --n-trials 500 \
  --params-file output/params_cybercycle_SOLUSDT_1h_fase3.json \
  --exclude-params use_trend,use_volume
Nota: Se excluyen use_trend y use_volume porque normalmente se decide manualmente si se usan o no; si se quiere que el optimizador decida, simplemente no se excluyen.

4B. Full sin trailing (para comparar con versión con trailing)
bash
python cli.py optimize --strategy cybercycle --symbol SOLUSDT --tf 1h \
  --method bayesian --objective monthly_robust --n-trials 300 \
  --exclude-params use_trailing,trail_activate_pct,trail_pullback_pct
Esto permite evaluar si el trailing realmente aporta valor o si la estrategia funciona igual de bien sin él.

Fase 5 – Robustez cross‑timeframe y cross‑symbol
Una vez obtenida una configuración prometedora (ej. params_fase4.json), se prueba en otros marcos temporales y pares para verificar que no está sobreoptimizada al timeframe o símbolo original.

5A. Misma configuración en 4h
bash
python cli.py optimize --strategy cybercycle --symbol SOLUSDT --tf 4h \
  --method bayesian --objective monthly_robust --n-trials 200 \
  --params-file output/params_cybercycle_SOLUSDT_1h_fase4.json \
  --optimize-params confidence_min,sl_atr_mult,tp1_rr,tp2_rr,leverage,min_bars
Se reoptimizan solo unos pocos parámetros (los más sensibles al timeframe) para adaptarse al nuevo TF, manteniendo la estructura general.

5B. Otro par – mismos params, distinto mercado
bash
python cli.py optimize --strategy cybercycle --symbol BTCUSDT --tf 1h \
  --method bayesian --objective monthly_robust --n-trials 200 \
  --params-file output/params_cybercycle_SOLUSDT_1h_fase4.json \
  --optimize-params confidence_min,sl_atr_mult,tp1_rr,tp2_rr,leverage
Fase 6 – Validación anti‑overfitting
Finalmente, se toma el mejor resultado de las fases anteriores y se somete al pipeline completo de 4 capas.

bash
python cli.py validate --strategy cybercycle --symbol SOLUSDT --tf 1h \
  --params-file output/params_cybercycle_SOLUSDT_1h_fase4.json
Opcionalmente, se puede probar también en 4h y BTCUSDT para confirmar robustez.

bash
python cli.py validate --strategy cybercycle --symbol SOLUSDT --tf 4h \
  --params-file output/params_cybercycle_SOLUSDT_1h_fase4.json

python cli.py validate --strategy cybercycle --symbol BTCUSDT --tf 1h \
  --params-file output/params_cybercycle_SOLUSDT_1h_fase4.json
📌 Notas adicionales
Nombres de archivos: Se recomienda usar nombres descriptivos para cada fase, por ejemplo:

params_cybercycle_SOLUSDT_1h_fase1.json

params_cybercycle_SOLUSDT_1h_fase2.json

etc.

Número de trials: Ajustar según el tiempo disponible; 200‑300 suele ser suficiente para bayesiano en un espacio reducido. Para exploración completa (fase 4), 500 trials pueden dar mejores resultados.

Objetivo: Se ha usado monthly_robust por su equilibrio, pero puede sustituirse por cualquier otro objetivo soportado (sharpe, return, calmar, composite, weekly_robust, etc.).

Dependencia de archivos: Asegurarse de que el archivo --params-file de cada fase existe antes de ejecutar el comando.

Exploración manual: Después de cada fase, se puede inspeccionar el JSON generado y decidir si se toman los mejores parámetros tal cual o se ajustan manualmente antes de la siguiente fase.

Parámetros excluidos vs. optimizados: Recuerda que --exclude-params bloquea la modificación de esos parámetros, mientras que --optimize-params especifica explícitamente los que se pueden variar. No se deben usar ambos a la vez.

🧩 Resumen de parámetros por fase
Fase	Parámetros optimizados	Parámetros fijos
1A	Señal + filtros (sin risk)	Todos los de risk management
1B	Solo método alpha y ciclo	Filtros y risk management
2A	ATR + R:R	Señal + fixed stops
2B	Fixed stops	Señal + ATR/RR
2C	Todo risk management	Señal
3	Filtros de calidad	Señal + risk
4A	Todo excepto bools triviales	(ninguno, o los excluidos)
4B	Todo excepto trailing	Trailing
5A/B	Ajuste fino en nuevo TF/par	Mayoría de parámetros
