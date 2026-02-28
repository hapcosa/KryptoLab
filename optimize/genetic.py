"""
CryptoLab — Genetic Algorithm Optimizer (DEAP)
Evolutionary strategy optimization via DEAP.

Features:
- Real-valued + categorical parameter encoding
- Tournament selection with configurable pressure
- BLX-α crossover (blend) for continuous params
- Gaussian mutation with adaptive σ
- Elitism (top N survive unchanged)
- Hall of Fame tracking
- Convergence detection (early stop if stagnant)

Reference: Goldberg "Genetic Algorithms in Search, Optimization and Machine Learning" (1989)
"""
import numpy as np
import copy
import time
import random
from typing import Dict, List, Any, Callable, Optional, Tuple
from dataclasses import dataclass, field

try:
    from deap import base, creator, tools, algorithms
    HAS_DEAP = True
except ImportError:
    HAS_DEAP = False


@dataclass
class GeneticTrial:
    """Result from a single individual evaluation."""
    generation: int
    params: Dict[str, Any]
    sharpe_ratio: float = 0.0
    total_return: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    n_trades: int = 0
    fitness: float = 0.0


@dataclass
class GeneticResult:
    """Complete genetic optimization results."""
    hall_of_fame: List[GeneticTrial]
    best_trial: Optional[GeneticTrial] = None
    best_params: Dict[str, Any] = field(default_factory=dict)
    n_generations: int = 0
    n_evaluations: int = 0
    convergence_gen: int = -1
    elapsed_seconds: float = 0.0
    fitness_history: List[float] = field(default_factory=list)


class GeneticOptimizer:
    """
    Genetic Algorithm optimizer for strategy parameters.

    Encodes each parameter as a gene in a chromosome. Continuous params
    use real-valued encoding; categorical params use integer indexing.

    Evolution loop:
      1. Initialize random population
      2. Evaluate fitness (backtest each individual)
      3. Select parents (tournament)
      4. Crossover (BLX-α for reals, uniform for categoricals)
      5. Mutate (Gaussian for reals, random flip for categoricals)
      6. Apply elitism (keep top-k unchanged)
      7. Repeat until max generations or convergence
    """

    def __init__(self,
                 population_size: int = 50,
                 n_generations: int = 30,
                 crossover_prob: float = 0.7,
                 mutation_prob: float = 0.2,
                 tournament_size: int = 3,
                 elite_count: int = 2,
                 objective: str = 'sharpe',
                 min_trades: int = 10,
                 convergence_patience: int = 8,
                 seed: int = 42,
                 verbose: bool = True):
        """
        Args:
            population_size: Number of individuals per generation
            n_generations: Maximum generations
            crossover_prob: Probability of crossover per pair
            mutation_prob: Probability of mutation per gene
            tournament_size: Selection tournament size
            elite_count: Top individuals that survive unchanged
            objective: Fitness function name
            min_trades: Min trades for valid individual
            convergence_patience: Stop if no improvement for N generations
            seed: Random seed
            verbose: Print progress
        """
        if not HAS_DEAP:
            raise ImportError("DEAP required: pip install deap")

        self.pop_size = population_size
        self.n_generations = n_generations
        self.cx_prob = crossover_prob
        self.mut_prob = mutation_prob
        self.tourn_size = tournament_size
        self.elite_count = elite_count
        self.objective_name = objective
        self.min_trades = min_trades
        self.patience = convergence_patience
        self.seed = seed
        self.verbose = verbose

    def _build_gene_spec(self, param_defs: list,
                         param_subset: Optional[List[str]] = None
                         ) -> List[dict]:
        """Build gene specification from ParamDefs."""
        genes = []
        for pdef in param_defs:
            if param_subset and pdef.name not in param_subset:
                continue

            if pdef.ptype == 'float':
                genes.append({
                    'name': pdef.name,
                    'type': 'float',
                    'min': pdef.min_val,
                    'max': pdef.max_val,
                    'step': pdef.step,
                    'default': pdef.default,
                })
            elif pdef.ptype == 'int':
                genes.append({
                    'name': pdef.name,
                    'type': 'int',
                    'min': pdef.min_val,
                    'max': pdef.max_val,
                    'default': pdef.default,
                })
            elif pdef.ptype == 'bool':
                genes.append({
                    'name': pdef.name,
                    'type': 'categorical',
                    'options': [True, False],
                    'default': pdef.default,
                })
            elif pdef.ptype == 'categorical':
                genes.append({
                    'name': pdef.name,
                    'type': 'categorical',
                    'options': pdef.options,
                    'default': pdef.default,
                })
        return genes

    def _chromosome_to_params(self, individual, gene_spec: list) -> Dict[str, Any]:
        """Decode chromosome to parameter dict."""
        params = {}
        for i, gene in enumerate(gene_spec):
            val = individual[i]
            if gene['type'] == 'float':
                val = float(np.clip(val, gene['min'], gene['max']))
                if gene.get('step'):
                    val = round(val / gene['step']) * gene['step']
            elif gene['type'] == 'int':
                val = int(np.clip(round(val), gene['min'], gene['max']))
            elif gene['type'] == 'categorical':
                idx = int(np.clip(round(val), 0, len(gene['options']) - 1))
                val = gene['options'][idx]
            params[gene['name']] = val
        return params

    def _compute_fitness(self, result) -> float:
        """Compute fitness from BacktestResult."""
        if result.n_trades < self.min_trades:
            return -999.0

        if self.objective_name == 'sharpe':
            return result.sharpe_ratio
        elif self.objective_name == 'return':
            return result.total_return
        elif self.objective_name == 'composite':
            sr = max(0, result.sharpe_ratio)
            pf = min(5, result.profit_factor) / 5.0 if result.profit_factor > 0 else 0
            cal = min(5, result.calmar_ratio) / 5.0 if result.calmar_ratio > 0 else 0
            wr = result.win_rate / 100.0
            return sr * 0.4 + pf * 0.2 + cal * 0.2 + wr * 0.2
        return result.sharpe_ratio

    def run(self,
            strategy,
            data: dict,
            engine_factory: Callable,
            symbol: str = "",
            timeframe: str = "",
            param_subset: Optional[List[str]] = None
            ) -> GeneticResult:
        """
        Execute genetic optimization.

        Args:
            strategy: IStrategy instance
            data: Full OHLCV data dict
            engine_factory: Callable returning BacktestEngine
            symbol: For reporting
            timeframe: For reporting
            param_subset: Only optimize these params (None = all)

        Returns:
            GeneticResult with hall of fame and convergence info
        """
        t0 = time.time()
        random.seed(self.seed)
        np.random.seed(self.seed)

        gene_spec = self._build_gene_spec(strategy.parameter_defs(), param_subset)
        n_genes = len(gene_spec)

        if self.verbose:
            print(f"\n  Genetic Optimizer: pop={self.pop_size} × "
                  f"gen={self.n_generations}, "
                  f"{n_genes} genes, obj={self.objective_name}")

        # ── DEAP setup ──
        # Clean previous creator classes if they exist
        if hasattr(creator, 'FitnessMax'):
            del creator.FitnessMax
        if hasattr(creator, 'Individual'):
            del creator.Individual

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()

        # Gene initialization
        for i, gene in enumerate(gene_spec):
            if gene['type'] == 'float':
                toolbox.register(f"gene_{i}", random.uniform,
                                gene['min'], gene['max'])
            elif gene['type'] == 'int':
                toolbox.register(f"gene_{i}", random.randint,
                                gene['min'], gene['max'])
            elif gene['type'] == 'categorical':
                toolbox.register(f"gene_{i}", random.randint,
                                0, len(gene['options']) - 1)

        def init_individual():
            ind = []
            for i in range(n_genes):
                ind.append(toolbox.__getattribute__(f"gene_{i}")())
            return creator.Individual(ind)

        toolbox.register("individual", init_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Evaluation cache
        eval_cache = {}
        all_trials = []
        n_evals = [0]

        def evaluate(individual):
            params = self._chromosome_to_params(individual, gene_spec)
            cache_key = tuple(sorted(params.items()))

            if cache_key in eval_cache:
                return eval_cache[cache_key]

            strat = copy.deepcopy(strategy)
            strat.set_params(params)
            engine = engine_factory()
            result = engine.run(strat, data, symbol, timeframe)

            fit = self._compute_fitness(result)
            n_evals[0] += 1

            trial = GeneticTrial(
                generation=-1,
                params=params,
                sharpe_ratio=result.sharpe_ratio,
                total_return=result.total_return,
                max_drawdown=result.max_drawdown,
                win_rate=result.win_rate,
                profit_factor=result.profit_factor,
                n_trades=result.n_trades,
                fitness=fit,
            )
            all_trials.append(trial)
            eval_cache[cache_key] = (fit,)
            return (fit,)

        toolbox.register("evaluate", evaluate)
        toolbox.register("select", tools.selTournament,
                        tournsize=self.tourn_size)

        # BLX-alpha crossover for real-valued
        def cx_blend(ind1, ind2, alpha=0.3):
            for i, gene in enumerate(gene_spec):
                if gene['type'] in ('float', 'int'):
                    lo = min(ind1[i], ind2[i])
                    hi = max(ind1[i], ind2[i])
                    rng = hi - lo
                    ind1[i] = random.uniform(lo - alpha * rng, hi + alpha * rng)
                    ind2[i] = random.uniform(lo - alpha * rng, hi + alpha * rng)
                else:
                    if random.random() < 0.5:
                        ind1[i], ind2[i] = ind2[i], ind1[i]
            return ind1, ind2

        toolbox.register("mate", cx_blend)

        # Gaussian mutation
        def mut_gaussian(individual, sigma_frac=0.1):
            for i, gene in enumerate(gene_spec):
                if random.random() < self.mut_prob:
                    if gene['type'] == 'float':
                        sigma = (gene['max'] - gene['min']) * sigma_frac
                        individual[i] += random.gauss(0, sigma)
                        individual[i] = np.clip(individual[i],
                                                gene['min'], gene['max'])
                    elif gene['type'] == 'int':
                        sigma = max(1, (gene['max'] - gene['min']) * sigma_frac)
                        individual[i] += round(random.gauss(0, sigma))
                        individual[i] = int(np.clip(individual[i],
                                                     gene['min'], gene['max']))
                    elif gene['type'] == 'categorical':
                        individual[i] = random.randint(
                            0, len(gene['options']) - 1)
            return (individual,)

        toolbox.register("mutate", mut_gaussian)

        # Hall of fame
        hof = tools.HallOfFame(10)

        # ── Evolution loop ──
        pop = toolbox.population(n=self.pop_size)
        fitness_history = []
        best_fitness = -np.inf
        stagnation = 0

        # Evaluate initial population
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        hof.update(pop)

        for gen in range(self.n_generations):
            # Select next generation
            offspring = toolbox.select(pop, len(pop) - self.elite_count)
            offspring = list(map(toolbox.clone, offspring))

            # Crossover
            for i in range(0, len(offspring) - 1, 2):
                if random.random() < self.cx_prob:
                    toolbox.mate(offspring[i], offspring[i + 1])
                    del offspring[i].fitness.values
                    del offspring[i + 1].fitness.values

            # Mutation (adaptive sigma: decreases over generations)
            sigma_frac = 0.15 * (1 - gen / self.n_generations) + 0.02
            for ind in offspring:
                toolbox.mutate(ind, sigma_frac=sigma_frac)
                if not ind.fitness.valid:
                    del ind.fitness.values

            # Evaluate invalid individuals
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(toolbox.evaluate, invalid))
            for ind, fit in zip(invalid, fitnesses):
                ind.fitness.values = fit

            # Elitism: keep top N from previous generation
            elite = tools.selBest(pop, self.elite_count)
            pop[:] = elite + offspring

            hof.update(pop)

            # Stats
            gen_fits = [ind.fitness.values[0] for ind in pop
                        if ind.fitness.values[0] > -900]
            gen_best = max(gen_fits) if gen_fits else -999
            gen_avg = np.mean(gen_fits) if gen_fits else 0
            fitness_history.append(gen_best)

            if gen_best > best_fitness + 0.001:
                best_fitness = gen_best
                stagnation = 0
            else:
                stagnation += 1

            if self.verbose and (gen + 1) % 5 == 0:
                print(f"    Gen {gen+1:3d}/{self.n_generations}: "
                      f"best={gen_best:.3f} avg={gen_avg:.3f} "
                      f"evals={n_evals[0]}")

            # Convergence check
            if stagnation >= self.patience:
                if self.verbose:
                    print(f"    Converged at gen {gen+1} "
                          f"(no improvement for {self.patience} gens)")
                break

        # Build result
        hof_trials = []
        for ind in hof:
            params = self._chromosome_to_params(ind, gene_spec)
            matching = [t for t in all_trials
                        if t.params == params]
            if matching:
                hof_trials.append(matching[-1])

        best = hof_trials[0] if hof_trials else (
            max(all_trials, key=lambda t: t.fitness) if all_trials else None)

        result = GeneticResult(
            hall_of_fame=hof_trials,
            best_trial=best,
            best_params=best.params if best else {},
            n_generations=gen + 1 if 'gen' in dir() else 0,
            n_evaluations=n_evals[0],
            convergence_gen=gen + 1 - stagnation if stagnation >= self.patience else -1,
            elapsed_seconds=time.time() - t0,
            fitness_history=fitness_history,
        )

        if self.verbose and best:
            print(f"\n  Genetic Optimization Complete ({result.elapsed_seconds:.1f}s)")
            print(f"  Generations: {result.n_generations}, "
                  f"Evaluations: {result.n_evaluations}")
            print(f"  Best: fitness={best.fitness:.3f} "
                  f"SR={best.sharpe_ratio:.2f} Ret={best.total_return:+.1f}% "
                  f"WR={best.win_rate:.1f}% DD={best.max_drawdown:.1f}%")
            print(f"  Params: {best.params}")

        return result
