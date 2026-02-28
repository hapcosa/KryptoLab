"""
Custom HyperOpt Loss Function - Quality Over Quantity
Ubicación: user_data/hyperopts/QualityOverQuantityLoss.py

Esta función de pérdida está diseñada para:
1. Penalizar estrategias con demasiados trades
2. Recompensar profit factor alto
3. Castigar drawdown excesivo
4. Favorecer consistencia sobre ganancias brutas
"""

from datetime import datetime
from pandas import DataFrame
import numpy as np
from freqtrade.optimize.hyperopt import IHyperOptLoss
from freqtrade.data.metrics import calculate_max_drawdown


class QualityOverQuantityLoss(IHyperOptLoss):
    """
    Loss function que prioriza calidad de trades sobre cantidad.
    
    Penalizaciones:
    - Muchos trades (>100 en el período = penalización exponencial)
    - Drawdown alto (>10% = penalización severa)
    - Profit factor bajo (<1.5 = penalización)
    - Win rate extremo (>70% = posible overfitting)
    
    Recompensas:
    - Profit factor alto
    - Consistencia en ganancias
    - Balance riesgo/retorno
    """
    
    # Configuración de targets
    TARGET_TRADES_PER_MONTH = 10  # ~2.5 por semana
    MAX_ACCEPTABLE_DRAWDOWN = 0.15  # 15%
    MIN_PROFIT_FACTOR = 1.5
    MAX_WIN_RATE = 0.70  # Penalizar si >70% (posible overfit)
    
    @staticmethod
    def hyperopt_loss_function(
        results: DataFrame,
        trade_count: int,
        min_date: datetime,
        max_date: datetime,
        config: dict,
        processed: dict,
        backtest_stats: dict,
        *args,
        **kwargs
    ) -> float:
        """
        Calcula el loss para hyperopt.
        Menor valor = mejor resultado.
        """
        
        # Si no hay suficientes trades, penalizar fuertemente
        if trade_count < 10:
            return 1000.0
        
        # --- CÁLCULOS BASE ---
        total_profit = results['profit_abs'].sum()
        total_profit_pct = results['profit_ratio'].sum() * 100
        
        # Duración del backtest en meses
        duration_days = (max_date - min_date).days
        duration_months = max(duration_days / 30, 1)
        
        # Trades por mes
        trades_per_month = trade_count / duration_months
        
        # --- 1. PENALIZACIÓN POR EXCESO DE TRADES ---
        target_total_trades = QualityOverQuantityLoss.TARGET_TRADES_PER_MONTH * duration_months
        
        if trade_count > target_total_trades * 1.5:
            # Penalización exponencial por exceso
            excess_ratio = trade_count / target_total_trades
            trade_penalty = (excess_ratio - 1.5) ** 2 * 50
        elif trade_count < target_total_trades * 0.3:
            # También penalizar muy pocos trades
            trade_penalty = 20
        else:
            trade_penalty = 0
        
        # --- 2. PENALIZACIÓN POR DRAWDOWN ---
        try:
            # Calcular drawdown
            cumulative = results['profit_abs'].cumsum()
            max_drawdown_abs = cumulative.min()
            
            if max_drawdown_abs < 0:
                # Normalizar drawdown como porcentaje del capital inicial (estimado)
                starting_balance = backtest_stats.get('starting_balance', 1000)
                max_drawdown_pct = abs(max_drawdown_abs) / starting_balance
                
                if max_drawdown_pct > QualityOverQuantityLoss.MAX_ACCEPTABLE_DRAWDOWN:
                    # Penalización severa por drawdown excesivo
                    dd_excess = max_drawdown_pct - QualityOverQuantityLoss.MAX_ACCEPTABLE_DRAWDOWN
                    drawdown_penalty = dd_excess * 200
                else:
                    drawdown_penalty = max_drawdown_pct * 10  # Penalización leve
            else:
                drawdown_penalty = 0
        except:
            drawdown_penalty = 0
        
        # --- 3. PROFIT FACTOR ---
        winning_trades = results.loc[results['profit_abs'] > 0, 'profit_abs']
        losing_trades = results.loc[results['profit_abs'] < 0, 'profit_abs']
        
        total_wins = winning_trades.sum() if len(winning_trades) > 0 else 0
        total_losses = abs(losing_trades.sum()) if len(losing_trades) > 0 else 0.001
        
        profit_factor = total_wins / total_losses if total_losses > 0 else 10
        
        if profit_factor < QualityOverQuantityLoss.MIN_PROFIT_FACTOR:
            pf_penalty = (QualityOverQuantityLoss.MIN_PROFIT_FACTOR - profit_factor) * 30
        else:
            pf_penalty = 0
        
        # Bonus por profit factor muy bueno
        pf_bonus = min(profit_factor - 1, 3) * 5 if profit_factor > 1 else 0
        
        # --- 4. WIN RATE ---
        win_rate = len(winning_trades) / trade_count if trade_count > 0 else 0
        
        # Penalizar win rate extremo (posible overfitting)
        if win_rate > QualityOverQuantityLoss.MAX_WIN_RATE:
            winrate_penalty = (win_rate - QualityOverQuantityLoss.MAX_WIN_RATE) * 100
        elif win_rate < 0.35:
            # También penalizar win rate muy bajo
            winrate_penalty = (0.35 - win_rate) * 50
        else:
            winrate_penalty = 0
        
        # --- 5. CONSISTENCIA ---
        # Calcular desviación estándar de profits
        profit_std = results['profit_ratio'].std()
        
        # Penalizar alta variabilidad
        if profit_std > 0.05:  # 5% de desviación
            consistency_penalty = (profit_std - 0.05) * 100
        else:
            consistency_penalty = 0
        
        # --- 6. CALIDAD PROMEDIO DE TRADES ---
        avg_profit_per_trade = total_profit_pct / trade_count if trade_count > 0 else 0
        
        # Bonus si el profit promedio por trade es bueno
        if avg_profit_per_trade > 1:  # >1% promedio por trade
            quality_bonus = min(avg_profit_per_trade, 5) * 2
        else:
            quality_bonus = 0
        
        # --- CÁLCULO FINAL DEL LOSS ---
        # Comenzamos con el negativo del profit (queremos maximizar profit)
        base_loss = -total_profit_pct
        
        # Aplicar penalizaciones
        total_penalty = (
            trade_penalty +
            drawdown_penalty +
            pf_penalty +
            winrate_penalty +
            consistency_penalty
        )
        
        # Aplicar bonuses
        total_bonus = pf_bonus + quality_bonus
        
        # Loss final
        loss = base_loss + total_penalty - total_bonus
        
        return loss


class ConservativeCalmarLoss(IHyperOptLoss):
    """
    Versión más conservadora de Calmar que:
    - Penaliza más el drawdown
    - Limita el número de trades aceptables
    - Favorece consistencia
    """
    
    @staticmethod
    def hyperopt_loss_function(
        results: DataFrame,
        trade_count: int,
        min_date: datetime,
        max_date: datetime,
        config: dict,
        processed: dict,
        backtest_stats: dict,
        *args,
        **kwargs
    ) -> float:
        
        if trade_count < 10:
            return 1000.0
        
        total_profit = results['profit_ratio'].sum()
        duration_days = (max_date - min_date).days
        duration_months = max(duration_days / 30, 1)
        
        # Trades por mes
        trades_per_month = trade_count / duration_months
        
        # Penalizar muchos trades
        if trades_per_month > 15:
            trade_multiplier = 0.5  # Reducir el valor de la estrategia
        elif trades_per_month > 10:
            trade_multiplier = 0.75
        else:
            trade_multiplier = 1.0
        
        # Calcular drawdown
        try:
            cumulative = results['profit_abs'].cumsum()
            running_max = cumulative.cummax()
            drawdown_series = cumulative - running_max
            max_drawdown = abs(drawdown_series.min())
            starting_balance = backtest_stats.get('starting_balance', 1000)
            max_drawdown_pct = max_drawdown / starting_balance
        except:
            max_drawdown_pct = 0.2
        
        # Calmar ratio modificado
        if max_drawdown_pct > 0:
            # Penalizar más el drawdown (elevar al cuadrado)
            calmar = total_profit / (max_drawdown_pct ** 1.5)
        else:
            calmar = total_profit * 10
        
        # Aplicar multiplicador de trades
        adjusted_calmar = calmar * trade_multiplier
        
        return -adjusted_calmar


class RegimeAwareLoss(IHyperOptLoss):
    """
    Loss function que considera el rendimiento en diferentes regímenes de mercado.
    Requiere que los trades tengan información del régimen.
    
    Nota: Esta función es más avanzada y requiere modificar la estrategia
    para incluir el régimen en los tags de entrada.
    """
    
    @staticmethod
    def hyperopt_loss_function(
        results: DataFrame,
        trade_count: int,
        min_date: datetime,
        max_date: datetime,
        config: dict,
        processed: dict,
        backtest_stats: dict,
        *args,
        **kwargs
    ) -> float:
        
        if trade_count < 10:
            return 1000.0
        
        total_profit = results['profit_ratio'].sum() * 100
        
        # Calcular métricas básicas
        winning_trades = results.loc[results['profit_ratio'] > 0]
        losing_trades = results.loc[results['profit_ratio'] < 0]
        
        win_rate = len(winning_trades) / trade_count
        
        # Profit factor
        total_wins = winning_trades['profit_ratio'].sum() if len(winning_trades) > 0 else 0
        total_losses = abs(losing_trades['profit_ratio'].sum()) if len(losing_trades) > 0 else 0.001
        profit_factor = total_wins / total_losses
        
        # Drawdown
        cumulative = results['profit_abs'].cumsum()
        max_drawdown = abs(cumulative.min()) if cumulative.min() < 0 else 0
        starting_balance = backtest_stats.get('starting_balance', 1000)
        max_drawdown_pct = max_drawdown / starting_balance
        
        # --- Score combinado ---
        # Queremos:
        # - Profit alto pero no exagerado
        # - Win rate entre 40-65%
        # - Profit factor > 1.5
        # - Drawdown < 15%
        
        score = total_profit
        
        # Penalizaciones
        if max_drawdown_pct > 0.15:
            score -= (max_drawdown_pct - 0.15) * 500
        
        if profit_factor < 1.5:
            score -= (1.5 - profit_factor) * 50
        
        if win_rate > 0.70:
            score -= (win_rate - 0.70) * 200
        elif win_rate < 0.40:
            score -= (0.40 - win_rate) * 100
        
        # Penalizar muchos trades
        duration_months = max((max_date - min_date).days / 30, 1)
        trades_per_month = trade_count / duration_months
        
        if trades_per_month > 12:
            score -= (trades_per_month - 12) * 5
        
        return -score
