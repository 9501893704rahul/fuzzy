"""
Fuzzy Rule-Based Classification System
Main classifier implementation with multiple inference methods and optimization strategies.
Optimized for handling datasets with low baseline accuracy.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score
import warnings

from .membership_functions import MembershipFunctionManager
from .rule_generation import RuleGenerator, AdaptiveRuleGenerator, FuzzyRule
from .genetic_optimizer import GeneticOptimizer, MultiObjectiveOptimizer, AdaptiveGeneticOptimizer


class FuzzyRuleClassifier(BaseEstimator, ClassifierMixin):
    """
    Fuzzy Rule-Based Classification System (FRBCS).
    
    A complete implementation supporting:
    - Multiple rule generation methods
    - Genetic algorithm optimization
    - Various inference mechanisms
    - Interpretable rule output
    
    Optimized for datasets with low accuracy through:
    - Adaptive membership function partitioning
    - Multi-objective rule optimization
    - Ensemble inference methods
    - Class imbalance handling
    """
    
    def __init__(self,
                 n_partitions: int = 5,
                 mf_type: str = 'triangular',
                 partition_method: str = 'adaptive',
                 rule_method: str = 'hybrid',
                 inference_method: str = 'weighted_vote',
                 optimize: bool = True,
                 optimization_method: str = 'adaptive',
                 n_generations: int = 50,
                 population_size: int = 100,
                 max_rules: int = None,
                 min_rule_weight: float = 0.1,
                 normalize: bool = True,
                 handle_imbalance: bool = True,
                 random_state: int = 42,
                 verbose: bool = False):
        """
        Args:
            n_partitions: Number of fuzzy partitions per feature
            mf_type: Membership function type ('triangular', 'gaussian', 'trapezoidal')
            partition_method: MF partitioning method ('uniform', 'quantile', 'kmeans', 'adaptive', 'class_aware')
            rule_method: Rule generation method ('wang_mendel', 'clustering', 'decision_tree', 'hybrid', 'exhaustive')
            inference_method: Inference method ('winner_takes_all', 'weighted_vote', 'additive')
            optimize: Whether to use GA optimization
            optimization_method: GA method ('standard', 'multi_objective', 'adaptive')
            n_generations: Number of GA generations
            population_size: GA population size
            max_rules: Maximum number of rules (None for unlimited)
            min_rule_weight: Minimum rule weight threshold
            normalize: Whether to normalize input features
            handle_imbalance: Whether to handle class imbalance
            random_state: Random seed
            verbose: Print progress information
        """
        self.n_partitions = n_partitions
        self.mf_type = mf_type
        self.partition_method = partition_method
        self.rule_method = rule_method
        self.inference_method = inference_method
        self.optimize = optimize
        self.optimization_method = optimization_method
        self.n_generations = n_generations
        self.population_size = population_size
        self.max_rules = max_rules
        self.min_rule_weight = min_rule_weight
        self.normalize = normalize
        self.handle_imbalance = handle_imbalance
        self.random_state = random_state
        self.verbose = verbose
        
        # Internal components
        self.mf_manager = None
        self.rule_generator = None
        self.optimizer = None
        self.scaler = None
        self.rules = []
        self.classes_ = None
        self.n_features_ = None
        self.feature_names_ = None
        self.class_weights_ = None
        self.is_fitted_ = False
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            feature_names: List[str] = None) -> 'FuzzyRuleClassifier':
        """
        Fit the fuzzy classifier to training data.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels
            feature_names: Optional feature names for interpretability
            
        Returns:
            self
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        self.n_features_ = X.shape[1]
        self.classes_ = np.unique(y)
        self.feature_names_ = feature_names or [f'X{i}' for i in range(self.n_features_)]
        
        if self.verbose:
            print(f"Training Fuzzy Classifier on {X.shape[0]} samples, {self.n_features_} features")
            print(f"Classes: {self.classes_}")
        
        # Normalize data
        if self.normalize:
            self.scaler = MinMaxScaler()
            X = self.scaler.fit_transform(X)
        
        # Handle class imbalance
        if self.handle_imbalance:
            self._compute_class_weights(y)
        
        # Initialize membership function manager
        self.mf_manager = MembershipFunctionManager(
            n_partitions=self.n_partitions,
            mf_type=self.mf_type
        )
        self.mf_manager.fit(X, method=self.partition_method, y=y)
        
        if self.verbose:
            print(f"Created {self.n_partitions} fuzzy partitions per feature using {self.partition_method} method")
        
        # Initialize rule generator
        if self.rule_method == 'adaptive':
            self.rule_generator = AdaptiveRuleGenerator(self.mf_manager, method='adaptive')
        else:
            self.rule_generator = RuleGenerator(self.mf_manager, method=self.rule_method)
        
        # Generate rules
        self.rules = self.rule_generator.generate_rules(
            X, y,
            feature_names=self.feature_names_,
            max_rules=self.max_rules,
            min_confidence=self.min_rule_weight
        )
        
        if self.verbose:
            print(f"Generated {len(self.rules)} fuzzy rules")
        
        # Optimize rules
        if self.optimize and len(self.rules) > 0:
            self._optimize_rules(X, y)
        
        # Filter low-weight rules
        self.rules = [r for r in self.rules if r.weight >= self.min_rule_weight]
        
        if self.verbose:
            print(f"Final rule base: {len(self.rules)} rules")
            accuracy = self._evaluate_accuracy(X, y)
            print(f"Training accuracy: {accuracy:.4f}")
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Predicted class labels
        """
        if not self.is_fitted_:
            raise RuntimeError("Classifier not fitted. Call fit() first.")
        
        X = np.asarray(X)
        
        if self.normalize and self.scaler is not None:
            X = self.scaler.transform(X)
        
        predictions = []
        
        for i in range(len(X)):
            pred = self._predict_single(X[i])
            predictions.append(pred)
        
        return np.array(predictions)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Class probabilities (n_samples, n_classes)
        """
        if not self.is_fitted_:
            raise RuntimeError("Classifier not fitted. Call fit() first.")
        
        X = np.asarray(X)
        
        if self.normalize and self.scaler is not None:
            X = self.scaler.transform(X)
        
        probabilities = []
        
        for i in range(len(X)):
            proba = self._predict_proba_single(X[i])
            probabilities.append(proba)
        
        return np.array(probabilities)
    
    def _predict_single(self, x: np.ndarray) -> int:
        """Predict class for a single sample."""
        if len(self.rules) == 0:
            return self.classes_[0]  # Default to first class
        
        # Fuzzify input
        fuzzified = np.zeros((self.n_features_, self.n_partitions))
        for j in range(self.n_features_):
            fuzzified[j, :] = self.mf_manager.fuzzify(x[j], j)
        
        if self.inference_method == 'winner_takes_all':
            return self._winner_takes_all(fuzzified)
        elif self.inference_method == 'weighted_vote':
            return self._weighted_vote(fuzzified)
        elif self.inference_method == 'additive':
            return self._additive_inference(fuzzified)
        else:
            return self._weighted_vote(fuzzified)
    
    def _predict_proba_single(self, x: np.ndarray) -> np.ndarray:
        """Predict class probabilities for a single sample."""
        proba = np.zeros(len(self.classes_))
        
        if len(self.rules) == 0:
            proba[0] = 1.0
            return proba
        
        # Fuzzify input
        fuzzified = np.zeros((self.n_features_, self.n_partitions))
        for j in range(self.n_features_):
            fuzzified[j, :] = self.mf_manager.fuzzify(x[j], j)
        
        # Accumulate class scores
        class_scores = {c: 0.0 for c in self.classes_}
        
        for rule in self.rules:
            matching = rule.matches(fuzzified)
            if matching > 0:
                class_scores[rule.consequent] += matching
        
        # Normalize to probabilities
        total = sum(class_scores.values())
        if total > 0:
            for i, c in enumerate(self.classes_):
                proba[i] = class_scores[c] / total
        else:
            proba[0] = 1.0
        
        return proba
    
    def _winner_takes_all(self, fuzzified: np.ndarray) -> int:
        """Winner-takes-all inference: select class of rule with highest matching."""
        best_matching = 0
        best_class = self.classes_[0]
        
        for rule in self.rules:
            matching = rule.matches(fuzzified)
            if matching > best_matching:
                best_matching = matching
                best_class = rule.consequent
        
        return best_class
    
    def _weighted_vote(self, fuzzified: np.ndarray) -> int:
        """Weighted voting inference: accumulate weighted votes for each class."""
        class_votes = {c: 0.0 for c in self.classes_}
        
        for rule in self.rules:
            matching = rule.matches(fuzzified)
            if matching > 0:
                vote = matching
                if self.handle_imbalance and self.class_weights_ is not None:
                    vote *= self.class_weights_.get(rule.consequent, 1.0)
                class_votes[rule.consequent] += vote
        
        if max(class_votes.values()) == 0:
            return self.classes_[0]
        
        return max(class_votes.keys(), key=lambda c: class_votes[c])
    
    def _additive_inference(self, fuzzified: np.ndarray) -> int:
        """Additive inference: sum matching degrees for each class."""
        class_scores = {c: 0.0 for c in self.classes_}
        
        for rule in self.rules:
            matching = rule.matches(fuzzified)
            class_scores[rule.consequent] += matching
        
        if max(class_scores.values()) == 0:
            return self.classes_[0]
        
        return max(class_scores.keys(), key=lambda c: class_scores[c])
    
    def _optimize_rules(self, X: np.ndarray, y: np.ndarray):
        """Apply genetic algorithm optimization to rules."""
        if self.verbose:
            print(f"Optimizing rules using {self.optimization_method} GA...")
        
        if self.optimization_method == 'standard':
            self.optimizer = GeneticOptimizer(
                population_size=self.population_size,
                n_generations=self.n_generations,
                random_state=self.random_state
            )
            self.rules = self.optimizer.optimize_rule_weights(
                self.rules, X, y, self.mf_manager
            )
        
        elif self.optimization_method == 'multi_objective':
            self.optimizer = MultiObjectiveOptimizer(
                population_size=self.population_size,
                n_generations=self.n_generations,
                random_state=self.random_state
            )
            self.rules, pareto_front = self.optimizer.optimize(
                self.rules, X, y, self.mf_manager, self.max_rules
            )
            if self.verbose:
                print(f"Pareto front: {len(pareto_front)} solutions")
        
        elif self.optimization_method == 'adaptive':
            self.optimizer = AdaptiveGeneticOptimizer(
                population_size=self.population_size,
                n_generations=self.n_generations,
                random_state=self.random_state
            )
            self.rules = self.optimizer.optimize_rule_weights(
                self.rules, X, y, self.mf_manager
            )
            
            # Also optimize MF parameters for difficult datasets
            if self.verbose:
                print("Optimizing membership function parameters...")
            self.optimizer.optimize_membership_functions(
                self.mf_manager, self.rules, X, y
            )
        
        if self.verbose and self.optimizer:
            print(f"Optimization complete. Best fitness: {self.optimizer.best_fitness:.4f}")
    
    def _compute_class_weights(self, y: np.ndarray):
        """Compute class weights for handling imbalanced data."""
        class_counts = np.bincount(y.astype(int))
        total = len(y)
        n_classes = len(class_counts)
        
        self.class_weights_ = {}
        for i, count in enumerate(class_counts):
            if count > 0:
                self.class_weights_[i] = total / (n_classes * count)
    
    def _evaluate_accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate classification accuracy."""
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return accuracy score."""
        return self._evaluate_accuracy(X, y)
    
    def print_rules(self, n: int = None, sort_by: str = 'weight'):
        """
        Print rules in human-readable format.
        
        Args:
            n: Number of rules to print (None for all)
            sort_by: Sort criterion ('weight', 'support', 'combined')
        """
        if not self.is_fitted_:
            print("Classifier not fitted.")
            return
        
        rules = self.rules.copy()
        
        if sort_by == 'weight':
            rules = sorted(rules, key=lambda r: r.weight, reverse=True)
        elif sort_by == 'support':
            rules = sorted(rules, key=lambda r: r.support, reverse=True)
        elif sort_by == 'combined':
            rules = sorted(rules, key=lambda r: r.weight * r.support, reverse=True)
        
        if n:
            rules = rules[:n]
        
        class_names = [f'Class_{c}' for c in self.classes_]
        linguistic_labels = self.mf_manager.linguistic_labels
        
        print(f"\n{'='*70}")
        print(f"FUZZY RULE BASE ({len(rules)} rules)")
        print(f"{'='*70}")
        
        for i, rule in enumerate(rules):
            print(f"R{i+1}: {rule.to_string(self.feature_names_, linguistic_labels, class_names)}")
        
        print(f"{'='*70}\n")
    
    def get_rule_importance(self) -> Dict[int, float]:
        """Get importance scores for each rule."""
        importance = {}
        for i, rule in enumerate(self.rules):
            importance[i] = rule.weight * rule.support
        return importance
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Estimate feature importance based on rule usage."""
        importance = {name: 0.0 for name in self.feature_names_}
        
        for rule in self.rules:
            rule_importance = rule.weight * rule.support
            for feat_idx, fuzzy_set in enumerate(rule.antecedent):
                if fuzzy_set >= 0:  # Not "don't care"
                    importance[self.feature_names_[feat_idx]] += rule_importance
        
        # Normalize
        total = sum(importance.values())
        if total > 0:
            importance = {k: v/total for k, v in importance.items()}
        
        return importance
    
    def export_rules(self, format: str = 'text') -> Union[str, List[Dict]]:
        """
        Export rules in specified format.
        
        Args:
            format: Output format ('text', 'json', 'fcl')
            
        Returns:
            Rules in specified format
        """
        if format == 'text':
            lines = []
            for i, rule in enumerate(self.rules):
                lines.append(rule.to_string(
                    self.feature_names_,
                    self.mf_manager.linguistic_labels,
                    [f'Class_{c}' for c in self.classes_]
                ))
            return '\n'.join(lines)
        
        elif format == 'json':
            rules_json = []
            for rule in self.rules:
                rules_json.append({
                    'antecedent': {
                        self.feature_names_[i]: self.mf_manager.linguistic_labels[v]
                        for i, v in enumerate(rule.antecedent) if v >= 0
                    },
                    'consequent': int(rule.consequent),
                    'weight': float(rule.weight),
                    'support': int(rule.support)
                })
            return rules_json
        
        elif format == 'fcl':
            # Fuzzy Control Language format
            lines = ["FUNCTION_BLOCK fuzzy_classifier", ""]
            
            # Variables
            lines.append("VAR_INPUT")
            for name in self.feature_names_:
                lines.append(f"    {name}: REAL;")
            lines.append("END_VAR")
            lines.append("")
            
            lines.append("VAR_OUTPUT")
            lines.append("    class: REAL;")
            lines.append("END_VAR")
            lines.append("")
            
            # Rules
            lines.append("RULEBLOCK rules")
            lines.append("    AND: MIN;")
            lines.append("    ACT: MIN;")
            lines.append("    ACCU: MAX;")
            lines.append("")
            
            for i, rule in enumerate(self.rules):
                conditions = []
                for feat_idx, fuzzy_set in enumerate(rule.antecedent):
                    if fuzzy_set >= 0:
                        label = self.mf_manager.linguistic_labels[fuzzy_set]
                        conditions.append(f"{self.feature_names_[feat_idx]} IS {label}")
                
                if conditions:
                    lines.append(f"    RULE {i+1}: IF {' AND '.join(conditions)} THEN class IS Class_{rule.consequent};")
            
            lines.append("END_RULEBLOCK")
            lines.append("")
            lines.append("END_FUNCTION_BLOCK")
            
            return '\n'.join(lines)
        
        return ""


class EnsembleFuzzyClassifier(BaseEstimator, ClassifierMixin):
    """
    Ensemble of fuzzy classifiers for improved accuracy on difficult datasets.
    Combines multiple fuzzy classifiers with different configurations.
    """
    
    def __init__(self,
                 n_estimators: int = 5,
                 partition_range: Tuple[int, int] = (3, 7),
                 methods: List[str] = None,
                 voting: str = 'soft',
                 random_state: int = 42,
                 verbose: bool = False):
        """
        Args:
            n_estimators: Number of base classifiers
            partition_range: Range of partition numbers to use
            methods: List of rule generation methods to use
            voting: Voting method ('hard', 'soft')
            random_state: Random seed
            verbose: Print progress
        """
        self.n_estimators = n_estimators
        self.partition_range = partition_range
        self.methods = methods or ['wang_mendel', 'clustering', 'hybrid']
        self.voting = voting
        self.random_state = random_state
        self.verbose = verbose
        
        self.estimators_ = []
        self.classes_ = None
        self.is_fitted_ = False
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: List[str] = None) -> 'EnsembleFuzzyClassifier':
        """Fit ensemble of fuzzy classifiers."""
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        
        np.random.seed(self.random_state)
        
        self.estimators_ = []
        
        for i in range(self.n_estimators):
            # Vary configuration
            n_partitions = np.random.randint(self.partition_range[0], self.partition_range[1] + 1)
            method = self.methods[i % len(self.methods)]
            mf_type = np.random.choice(['triangular', 'gaussian'])
            
            if self.verbose:
                print(f"Training estimator {i+1}/{self.n_estimators}: "
                      f"partitions={n_partitions}, method={method}, mf={mf_type}")
            
            clf = FuzzyRuleClassifier(
                n_partitions=n_partitions,
                mf_type=mf_type,
                rule_method=method,
                optimize=True,
                optimization_method='adaptive',
                n_generations=30,
                population_size=50,
                random_state=self.random_state + i,
                verbose=False
            )
            
            clf.fit(X, y, feature_names)
            self.estimators_.append(clf)
        
        self.is_fitted_ = True
        
        if self.verbose:
            accuracy = self.score(X, y)
            print(f"Ensemble training accuracy: {accuracy:.4f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using ensemble voting."""
        if not self.is_fitted_:
            raise RuntimeError("Ensemble not fitted.")
        
        if self.voting == 'hard':
            predictions = np.array([clf.predict(X) for clf in self.estimators_])
            # Majority voting
            final_predictions = []
            for i in range(len(X)):
                votes = predictions[:, i]
                unique, counts = np.unique(votes, return_counts=True)
                final_predictions.append(unique[np.argmax(counts)])
            return np.array(final_predictions)
        
        else:  # soft voting
            probas = np.array([clf.predict_proba(X) for clf in self.estimators_])
            avg_proba = np.mean(probas, axis=0)
            return self.classes_[np.argmax(avg_proba, axis=1)]
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted_:
            raise RuntimeError("Ensemble not fitted.")
        
        probas = np.array([clf.predict_proba(X) for clf in self.estimators_])
        return np.mean(probas, axis=0)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return accuracy score."""
        predictions = self.predict(X)
        return np.mean(predictions == y)
