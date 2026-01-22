"""
Genetic Algorithm Optimizer Module
Implements GA-based optimization for fuzzy rule bases and membership functions.
Includes multi-objective optimization for accuracy vs interpretability trade-off.
"""

import numpy as np
import random
from typing import List, Tuple, Dict, Callable, Optional
from copy import deepcopy
from deap import base, creator, tools, algorithms
import warnings

from .rule_generation import FuzzyRule


class GeneticOptimizer:
    """
    Genetic Algorithm optimizer for fuzzy rule-based classification systems.
    Supports optimization of:
    - Rule weights
    - Rule selection (which rules to include)
    - Membership function parameters
    """
    
    def __init__(self, 
                 population_size: int = 100,
                 n_generations: int = 50,
                 crossover_prob: float = 0.8,
                 mutation_prob: float = 0.2,
                 elite_size: int = 5,
                 tournament_size: int = 3,
                 random_state: int = 42):
        """
        Args:
            population_size: Number of individuals in population
            n_generations: Number of generations to evolve
            crossover_prob: Probability of crossover
            mutation_prob: Probability of mutation
            elite_size: Number of best individuals to preserve
            tournament_size: Tournament selection size
            random_state: Random seed for reproducibility
        """
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        self.random_state = random_state
        
        self.best_individual = None
        self.best_fitness = 0.0
        self.fitness_history = []
        
        random.seed(random_state)
        np.random.seed(random_state)
    
    def optimize_rule_weights(self, rules: List[FuzzyRule],
                              X: np.ndarray, y: np.ndarray,
                              mf_manager,
                              fitness_func: Callable = None) -> List[FuzzyRule]:
        """
        Optimize rule weights using genetic algorithm.
        
        Args:
            rules: List of fuzzy rules to optimize
            X: Training features
            y: Training labels
            mf_manager: MembershipFunctionManager instance
            fitness_func: Custom fitness function (optional)
            
        Returns:
            Optimized rules with updated weights
        """
        if len(rules) == 0:
            return rules
        
        n_rules = len(rules)
        
        # Setup DEAP
        if hasattr(creator, 'FitnessMax'):
            del creator.FitnessMax
        if hasattr(creator, 'Individual'):
            del creator.Individual
        
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        
        # Individual: array of rule weights [0, 1]
        toolbox.register("attr_weight", random.random)
        toolbox.register("individual", tools.initRepeat, creator.Individual,
                        toolbox.attr_weight, n=n_rules)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        # Fitness function
        def evaluate(individual):
            # Apply weights to rules
            temp_rules = deepcopy(rules)
            for i, rule in enumerate(temp_rules):
                rule.weight = individual[i]
            
            # Calculate accuracy
            accuracy = self._calculate_accuracy(temp_rules, X, y, mf_manager)
            return (accuracy,)
        
        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)
        
        # Run GA
        population = toolbox.population(n=self.population_size)
        
        # Evaluate initial population
        fitnesses = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        self.fitness_history = []
        
        for gen in range(self.n_generations):
            # Select elite
            elite = tools.selBest(population, self.elite_size)
            
            # Select for breeding
            offspring = toolbox.select(population, len(population) - self.elite_size)
            offspring = list(map(toolbox.clone, offspring))
            
            # Crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.crossover_prob:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            # Mutation
            for mutant in offspring:
                if random.random() < self.mutation_prob:
                    toolbox.mutate(mutant)
                    # Clip weights to [0, 1]
                    for i in range(len(mutant)):
                        mutant[i] = max(0, min(1, mutant[i]))
                    del mutant.fitness.values
            
            # Evaluate offspring
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Replace population
            population[:] = elite + offspring
            
            # Track best fitness
            best = tools.selBest(population, 1)[0]
            self.fitness_history.append(best.fitness.values[0])
        
        # Get best individual
        self.best_individual = tools.selBest(population, 1)[0]
        self.best_fitness = self.best_individual.fitness.values[0]
        
        # Apply best weights to rules
        for i, rule in enumerate(rules):
            rule.weight = self.best_individual[i]
        
        return rules
    
    def optimize_rule_selection(self, rules: List[FuzzyRule],
                                X: np.ndarray, y: np.ndarray,
                                mf_manager,
                                max_rules: int = None) -> List[FuzzyRule]:
        """
        Select optimal subset of rules using genetic algorithm.
        
        Args:
            rules: Candidate rules
            X: Training features
            y: Training labels
            mf_manager: MembershipFunctionManager instance
            max_rules: Maximum number of rules to select
            
        Returns:
            Selected subset of rules
        """
        if len(rules) == 0:
            return rules
        
        n_rules = len(rules)
        if max_rules is None:
            max_rules = n_rules
        
        # Setup DEAP
        if hasattr(creator, 'FitnessMulti'):
            del creator.FitnessMulti
        if hasattr(creator, 'IndividualBinary'):
            del creator.IndividualBinary
        
        # Multi-objective: maximize accuracy, minimize number of rules
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, -0.1))
        creator.create("IndividualBinary", list, fitness=creator.FitnessMulti)
        
        toolbox = base.Toolbox()
        
        # Individual: binary array indicating rule selection
        toolbox.register("attr_bool", random.randint, 0, 1)
        toolbox.register("individual", tools.initRepeat, creator.IndividualBinary,
                        toolbox.attr_bool, n=n_rules)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        def evaluate(individual):
            # Select rules
            selected_rules = [rules[i] for i in range(n_rules) if individual[i] == 1]
            
            if len(selected_rules) == 0:
                return (0.0, n_rules)
            
            if len(selected_rules) > max_rules:
                return (0.0, len(selected_rules))
            
            accuracy = self._calculate_accuracy(selected_rules, X, y, mf_manager)
            n_selected = len(selected_rules)
            
            return (accuracy, n_selected)
        
        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
        toolbox.register("select", tools.selNSGA2)
        
        # Run GA
        population = toolbox.population(n=self.population_size)
        
        # Evaluate initial population
        fitnesses = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        for gen in range(self.n_generations):
            offspring = algorithms.varAnd(population, toolbox, 
                                         self.crossover_prob, self.mutation_prob)
            
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            population = toolbox.select(population + offspring, self.population_size)
        
        # Select best individual (highest accuracy with reasonable rule count)
        best = max(population, key=lambda ind: ind.fitness.values[0] - 0.01 * ind.fitness.values[1])
        
        selected_rules = [rules[i] for i in range(n_rules) if best[i] == 1]
        return selected_rules
    
    def optimize_membership_functions(self, mf_manager,
                                      rules: List[FuzzyRule],
                                      X: np.ndarray, y: np.ndarray) -> None:
        """
        Optimize membership function parameters using genetic algorithm.
        
        Args:
            mf_manager: MembershipFunctionManager to optimize
            rules: Current rule base
            X: Training features
            y: Training labels
        """
        # Get current parameters
        original_params = mf_manager.get_params_flat()
        n_params = len(original_params)
        
        if n_params == 0:
            return
        
        # Setup DEAP
        if hasattr(creator, 'FitnessMF'):
            del creator.FitnessMF
        if hasattr(creator, 'IndividualMF'):
            del creator.IndividualMF
        
        creator.create("FitnessMF", base.Fitness, weights=(1.0,))
        creator.create("IndividualMF", list, fitness=creator.FitnessMF)
        
        toolbox = base.Toolbox()
        
        # Individual: perturbation factors for MF parameters
        def init_param():
            return random.gauss(1.0, 0.1)  # Small perturbation around 1.0
        
        toolbox.register("attr_param", init_param)
        toolbox.register("individual", tools.initRepeat, creator.IndividualMF,
                        toolbox.attr_param, n=n_params)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        def evaluate(individual):
            # Apply perturbation to parameters
            new_params = original_params * np.array(individual)
            
            # Create temporary MF manager
            temp_mf = mf_manager.copy()
            temp_mf.set_params_flat(new_params)
            
            # Calculate accuracy
            accuracy = self._calculate_accuracy(rules, X, y, temp_mf)
            return (accuracy,)
        
        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", tools.cxBlend, alpha=0.3)
        toolbox.register("mutate", tools.mutGaussian, mu=1.0, sigma=0.1, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)
        
        # Run GA with fewer generations for MF optimization
        population = toolbox.population(n=self.population_size // 2)
        
        fitnesses = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        for gen in range(self.n_generations // 2):
            elite = tools.selBest(population, self.elite_size)
            offspring = toolbox.select(population, len(population) - self.elite_size)
            offspring = list(map(toolbox.clone, offspring))
            
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.crossover_prob:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            for mutant in offspring:
                if random.random() < self.mutation_prob:
                    toolbox.mutate(mutant)
                    # Clip perturbation factors
                    for i in range(len(mutant)):
                        mutant[i] = max(0.5, min(1.5, mutant[i]))
                    del mutant.fitness.values
            
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            population[:] = elite + offspring
        
        # Apply best parameters
        best = tools.selBest(population, 1)[0]
        best_params = original_params * np.array(best)
        mf_manager.set_params_flat(best_params)
    
    def _calculate_accuracy(self, rules: List[FuzzyRule],
                           X: np.ndarray, y: np.ndarray,
                           mf_manager) -> float:
        """Calculate classification accuracy for given rules (optimized)."""
        if len(rules) == 0:
            return 0.0
        
        n_samples = len(X)
        n_rules = len(rules)
        n_features = X.shape[1]
        n_partitions = mf_manager.n_partitions
        
        # Pre-compute fuzzified values
        fuzzified = mf_manager.fuzzify_batch(X)
        
        # Pre-extract rule data for vectorized operations
        antecedents = np.array([r.antecedent for r in rules])
        consequents = np.array([r.consequent for r in rules])
        weights = np.array([r.weight for r in rules])
        
        correct = 0
        for i in range(n_samples):
            # Compute matching for all rules at once
            matching = np.ones(n_rules) * weights
            for feat_idx in range(n_features):
                fuzzy_sets = antecedents[:, feat_idx]
                valid_mask = fuzzy_sets >= 0
                matching[valid_mask] *= fuzzified[i, feat_idx, fuzzy_sets[valid_mask]]
            
            # Find best matching rule
            best_idx = np.argmax(matching)
            if matching[best_idx] > 0 and consequents[best_idx] == y[i]:
                correct += 1
        
        return correct / n_samples


class MultiObjectiveOptimizer(GeneticOptimizer):
    """
    Multi-objective optimizer balancing accuracy and interpretability.
    Uses NSGA-II for Pareto-optimal solutions.
    """
    
    def __init__(self, 
                 population_size: int = 100,
                 n_generations: int = 50,
                 accuracy_weight: float = 0.7,
                 interpretability_weight: float = 0.3,
                 **kwargs):
        super().__init__(population_size, n_generations, **kwargs)
        self.accuracy_weight = accuracy_weight
        self.interpretability_weight = interpretability_weight
        self.pareto_front = []
    
    def optimize(self, rules: List[FuzzyRule],
                 X: np.ndarray, y: np.ndarray,
                 mf_manager,
                 max_rules: int = None) -> Tuple[List[FuzzyRule], List[Tuple]]:
        """
        Multi-objective optimization of rule base.
        
        Returns:
            Tuple of (optimized rules, pareto front solutions)
        """
        if len(rules) == 0:
            return rules, []
        
        n_rules = len(rules)
        
        # Setup DEAP for NSGA-II
        if hasattr(creator, 'FitnessMO'):
            del creator.FitnessMO
        if hasattr(creator, 'IndividualMO'):
            del creator.IndividualMO
        
        # Objectives: maximize accuracy, minimize complexity (number of rules)
        creator.create("FitnessMO", base.Fitness, weights=(1.0, -1.0))
        creator.create("IndividualMO", list, fitness=creator.FitnessMO)
        
        toolbox = base.Toolbox()
        
        # Individual: [rule_selection (binary), rule_weights (float)]
        def create_individual():
            selection = [random.randint(0, 1) for _ in range(n_rules)]
            weights = [random.random() for _ in range(n_rules)]
            return creator.IndividualMO(selection + weights)
        
        toolbox.register("individual", create_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        def evaluate(individual):
            selection = individual[:n_rules]
            weights = individual[n_rules:]
            
            selected_rules = []
            for i in range(n_rules):
                if selection[i] == 1:
                    rule_copy = deepcopy(rules[i])
                    rule_copy.weight = weights[i]
                    selected_rules.append(rule_copy)
            
            if len(selected_rules) == 0:
                return (0.0, n_rules)
            
            accuracy = self._calculate_accuracy(selected_rules, X, y, mf_manager)
            complexity = len(selected_rules)
            
            return (accuracy, complexity)
        
        toolbox.register("evaluate", evaluate)
        
        def mate(ind1, ind2):
            # Two-point crossover for selection part
            tools.cxTwoPoint(ind1[:n_rules], ind2[:n_rules])
            # Blend crossover for weights part
            for i in range(n_rules, len(ind1)):
                alpha = 0.5
                ind1[i], ind2[i] = (
                    (1 - alpha) * ind1[i] + alpha * ind2[i],
                    alpha * ind1[i] + (1 - alpha) * ind2[i]
                )
            return ind1, ind2
        
        def mutate(individual):
            # Flip bits for selection
            for i in range(n_rules):
                if random.random() < 0.1:
                    individual[i] = 1 - individual[i]
            # Gaussian mutation for weights
            for i in range(n_rules, len(individual)):
                if random.random() < 0.2:
                    individual[i] += random.gauss(0, 0.1)
                    individual[i] = max(0, min(1, individual[i]))
            return individual,
        
        toolbox.register("mate", mate)
        toolbox.register("mutate", mutate)
        toolbox.register("select", tools.selNSGA2)
        
        # Run NSGA-II
        population = toolbox.population(n=self.population_size)
        
        fitnesses = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        for gen in range(self.n_generations):
            offspring = algorithms.varAnd(population, toolbox,
                                         self.crossover_prob, self.mutation_prob)
            
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            population = toolbox.select(population + offspring, self.population_size)
        
        # Extract Pareto front
        self.pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
        
        # Select best solution based on weighted objectives
        def weighted_score(ind):
            acc, comp = ind.fitness.values
            return self.accuracy_weight * acc - self.interpretability_weight * (comp / n_rules)
        
        best = max(self.pareto_front, key=weighted_score)
        
        # Extract optimized rules
        selection = best[:n_rules]
        weights = best[n_rules:]
        
        optimized_rules = []
        for i in range(n_rules):
            if selection[i] == 1:
                rule_copy = deepcopy(rules[i])
                rule_copy.weight = weights[i]
                optimized_rules.append(rule_copy)
        
        pareto_solutions = [(ind.fitness.values[0], ind.fitness.values[1]) 
                           for ind in self.pareto_front]
        
        return optimized_rules, pareto_solutions


class AdaptiveGeneticOptimizer(GeneticOptimizer):
    """
    Adaptive GA with self-adjusting parameters for difficult optimization problems.
    Particularly useful for low-accuracy datasets.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.stagnation_counter = 0
        self.diversity_threshold = 0.1
    
    def optimize_rule_weights(self, rules: List[FuzzyRule],
                              X: np.ndarray, y: np.ndarray,
                              mf_manager,
                              fitness_func: Callable = None) -> List[FuzzyRule]:
        """Optimize with adaptive parameter control."""
        if len(rules) == 0:
            return rules
        
        n_rules = len(rules)
        
        # Setup DEAP
        if hasattr(creator, 'FitnessAdaptive'):
            del creator.FitnessAdaptive
        if hasattr(creator, 'IndividualAdaptive'):
            del creator.IndividualAdaptive
        
        creator.create("FitnessAdaptive", base.Fitness, weights=(1.0,))
        creator.create("IndividualAdaptive", list, fitness=creator.FitnessAdaptive)
        
        toolbox = base.Toolbox()
        toolbox.register("attr_weight", random.random)
        toolbox.register("individual", tools.initRepeat, creator.IndividualAdaptive,
                        toolbox.attr_weight, n=n_rules)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        def evaluate(individual):
            temp_rules = deepcopy(rules)
            for i, rule in enumerate(temp_rules):
                rule.weight = individual[i]
            accuracy = self._calculate_accuracy(temp_rules, X, y, mf_manager)
            return (accuracy,)
        
        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)
        
        population = toolbox.population(n=self.population_size)
        
        fitnesses = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        self.fitness_history = []
        prev_best = 0
        self.stagnation_counter = 0
        
        # Adaptive parameters
        current_mutation_prob = self.mutation_prob
        current_crossover_prob = self.crossover_prob
        
        for gen in range(self.n_generations):
            # Adaptive parameter adjustment
            if self.stagnation_counter > 5:
                # Increase diversity
                current_mutation_prob = min(0.5, current_mutation_prob * 1.2)
                # Inject random individuals
                n_random = self.population_size // 10
                random_inds = toolbox.population(n=n_random)
                fitnesses = map(toolbox.evaluate, random_inds)
                for ind, fit in zip(random_inds, fitnesses):
                    ind.fitness.values = fit
                population[-n_random:] = random_inds
            elif self.stagnation_counter == 0:
                # Converging well, reduce mutation
                current_mutation_prob = max(0.05, current_mutation_prob * 0.95)
            
            elite = tools.selBest(population, self.elite_size)
            offspring = toolbox.select(population, len(population) - self.elite_size)
            offspring = list(map(toolbox.clone, offspring))
            
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < current_crossover_prob:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            for mutant in offspring:
                if random.random() < current_mutation_prob:
                    toolbox.mutate(mutant)
                    for i in range(len(mutant)):
                        mutant[i] = max(0, min(1, mutant[i]))
                    del mutant.fitness.values
            
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            population[:] = elite + offspring
            
            best = tools.selBest(population, 1)[0]
            current_best = best.fitness.values[0]
            self.fitness_history.append(current_best)
            
            # Check for stagnation
            if abs(current_best - prev_best) < 0.001:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0
            prev_best = current_best
        
        self.best_individual = tools.selBest(population, 1)[0]
        self.best_fitness = self.best_individual.fitness.values[0]
        
        for i, rule in enumerate(rules):
            rule.weight = self.best_individual[i]
        
        return rules
