"""
Rule Generation Module
Implements multiple methods for automatic fuzzy rule generation:
- Wang-Mendel method
- Clustering-based method
- Decision Tree-based method
- Hybrid methods for improved accuracy on difficult datasets
"""

import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Set
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
import warnings


class FuzzyRule:
    """Represents a single fuzzy IF-THEN rule."""
    
    def __init__(self, antecedent: Tuple[int, ...], consequent: int, 
                 weight: float = 1.0, support: int = 0):
        """
        Args:
            antecedent: Tuple of fuzzy set indices for each feature
            consequent: Class label
            weight: Rule weight/confidence
            support: Number of training samples supporting this rule
        """
        self.antecedent = antecedent
        self.consequent = consequent
        self.weight = weight
        self.support = support
        self.fitness = 0.0  # For GA optimization
    
    def __repr__(self):
        return f"Rule(IF {self.antecedent} THEN {self.consequent}, w={self.weight:.3f})"
    
    def __eq__(self, other):
        if isinstance(other, FuzzyRule):
            return self.antecedent == other.antecedent
        return False
    
    def __hash__(self):
        return hash(self.antecedent)
    
    def matches(self, fuzzified_sample: np.ndarray) -> float:
        """
        Compute matching degree of a fuzzified sample with this rule.
        
        Args:
            fuzzified_sample: Array of shape (n_features, n_partitions)
            
        Returns:
            Matching degree (product of membership degrees)
        """
        matching = 1.0
        for feat_idx, fuzzy_set_idx in enumerate(self.antecedent):
            if fuzzy_set_idx >= 0:  # -1 means "don't care"
                matching *= fuzzified_sample[feat_idx, fuzzy_set_idx]
        return matching * self.weight
    
    def to_string(self, feature_names: List[str] = None, 
                  linguistic_labels: List[str] = None,
                  class_names: List[str] = None) -> str:
        """Convert rule to human-readable string."""
        if feature_names is None:
            feature_names = [f'X{i}' for i in range(len(self.antecedent))]
        if linguistic_labels is None:
            linguistic_labels = [f'L{i}' for i in range(max(self.antecedent) + 1)]
        
        conditions = []
        for feat_idx, fuzzy_set_idx in enumerate(self.antecedent):
            if fuzzy_set_idx >= 0:
                label = linguistic_labels[fuzzy_set_idx] if fuzzy_set_idx < len(linguistic_labels) else f'L{fuzzy_set_idx}'
                conditions.append(f"{feature_names[feat_idx]} is {label}")
        
        antecedent_str = " AND ".join(conditions) if conditions else "TRUE"
        
        if class_names and self.consequent < len(class_names):
            consequent_str = class_names[self.consequent]
        else:
            consequent_str = f"Class_{self.consequent}"
        
        return f"IF {antecedent_str} THEN {consequent_str} (w={self.weight:.3f}, support={self.support})"


class RuleGenerator:
    """
    Generates fuzzy rules from data using various methods.
    Optimized for handling datasets with low baseline accuracy.
    """
    
    def __init__(self, mf_manager, method: str = 'wang_mendel'):
        """
        Args:
            mf_manager: MembershipFunctionManager instance
            method: Rule generation method ('wang_mendel', 'clustering', 'decision_tree', 'hybrid')
        """
        self.mf_manager = mf_manager
        self.method = method
        self.rules: List[FuzzyRule] = []
        self.n_features = 0
        self.n_classes = 0
        self.class_labels = None
    
    def generate_rules(self, X: np.ndarray, y: np.ndarray, 
                       feature_names: List[str] = None,
                       max_rules: int = None,
                       min_support: int = 1,
                       min_confidence: float = 0.0) -> List[FuzzyRule]:
        """
        Generate fuzzy rules from training data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels
            feature_names: Optional feature names
            max_rules: Maximum number of rules to generate
            min_support: Minimum support for a rule
            min_confidence: Minimum confidence/weight for a rule
            
        Returns:
            List of generated FuzzyRule objects
        """
        self.n_features = X.shape[1]
        self.class_labels = np.unique(y)
        self.n_classes = len(self.class_labels)
        
        if self.method == 'wang_mendel':
            rules = self._wang_mendel(X, y)
        elif self.method == 'clustering':
            rules = self._clustering_based(X, y)
        elif self.method == 'decision_tree':
            rules = self._decision_tree_based(X, y)
        elif self.method == 'hybrid':
            rules = self._hybrid_method(X, y)
        elif self.method == 'exhaustive':
            rules = self._exhaustive_generation(X, y)
        else:
            rules = self._wang_mendel(X, y)
        
        # Filter rules by support and confidence
        rules = [r for r in rules if r.support >= min_support and r.weight >= min_confidence]
        
        # Limit number of rules if specified
        if max_rules and len(rules) > max_rules:
            rules = sorted(rules, key=lambda r: (r.weight * r.support), reverse=True)[:max_rules]
        
        self.rules = rules
        return rules
    
    def _wang_mendel(self, X: np.ndarray, y: np.ndarray) -> List[FuzzyRule]:
        """
        Wang-Mendel method for fuzzy rule generation.
        Creates one rule per training sample, then resolves conflicts.
        """
        n_samples = X.shape[0]
        rule_dict = defaultdict(lambda: defaultdict(float))  # antecedent -> class -> weight
        rule_support = defaultdict(lambda: defaultdict(int))  # antecedent -> class -> support
        
        for i in range(n_samples):
            # Fuzzify sample
            antecedent = []
            matching_degree = 1.0
            
            for feat_idx in range(self.n_features):
                memberships = self.mf_manager.fuzzify(X[i, feat_idx], feat_idx)
                dominant_set = np.argmax(memberships)
                antecedent.append(dominant_set)
                matching_degree *= memberships[dominant_set]
            
            antecedent = tuple(antecedent)
            class_label = y[i]
            
            # Accumulate weights
            rule_dict[antecedent][class_label] += matching_degree
            rule_support[antecedent][class_label] += 1
        
        # Create rules with conflict resolution
        rules = []
        for antecedent, class_weights in rule_dict.items():
            # Select class with highest accumulated weight
            best_class = max(class_weights.keys(), key=lambda c: class_weights[c])
            total_weight = sum(class_weights.values())
            confidence = class_weights[best_class] / total_weight if total_weight > 0 else 0
            support = rule_support[antecedent][best_class]
            
            rule = FuzzyRule(antecedent, best_class, confidence, support)
            rules.append(rule)
        
        return rules
    
    def _clustering_based(self, X: np.ndarray, y: np.ndarray) -> List[FuzzyRule]:
        """
        Clustering-based rule generation.
        Creates rules based on cluster centers for each class.
        """
        rules = []
        
        for class_label in self.class_labels:
            class_mask = y == class_label
            class_data = X[class_mask]
            
            if len(class_data) < 2:
                continue
            
            # Determine number of clusters based on class size
            n_clusters = min(max(2, len(class_data) // 10), 5)
            
            try:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                kmeans.fit(class_data)
                
                for center in kmeans.cluster_centers_:
                    antecedent = []
                    for feat_idx in range(self.n_features):
                        memberships = self.mf_manager.fuzzify(center[feat_idx], feat_idx)
                        dominant_set = np.argmax(memberships)
                        antecedent.append(dominant_set)
                    
                    antecedent = tuple(antecedent)
                    
                    # Calculate support and weight
                    support = np.sum(class_mask)
                    weight = 1.0 / n_clusters
                    
                    rule = FuzzyRule(antecedent, class_label, weight, support)
                    rules.append(rule)
            except:
                continue
        
        # Remove duplicate rules and resolve conflicts
        rules = self._resolve_conflicts(rules)
        return rules
    
    def _decision_tree_based(self, X: np.ndarray, y: np.ndarray) -> List[FuzzyRule]:
        """
        Decision tree-based rule generation.
        Extracts rules from decision tree paths and converts to fuzzy rules.
        """
        # Train decision tree
        dt = DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=42)
        dt.fit(X, y)
        
        # Extract rules from tree
        rules = []
        tree = dt.tree_
        
        def extract_rules(node_id, path_conditions):
            if tree.children_left[node_id] == tree.children_right[node_id]:  # Leaf
                # Get class prediction
                class_counts = tree.value[node_id][0]
                predicted_class = np.argmax(class_counts)
                support = int(class_counts[predicted_class])
                confidence = class_counts[predicted_class] / np.sum(class_counts)
                
                # Convert path conditions to fuzzy antecedent
                antecedent = [-1] * self.n_features  # -1 means "don't care"
                
                for feat_idx, threshold, direction in path_conditions:
                    # Find fuzzy set that best matches the threshold condition
                    memberships = self.mf_manager.fuzzify(threshold, feat_idx)
                    
                    if direction == 'left':  # <= threshold
                        # Use lower fuzzy sets
                        best_set = np.argmax(memberships[:len(memberships)//2 + 1])
                    else:  # > threshold
                        # Use higher fuzzy sets
                        best_set = len(memberships)//2 + np.argmax(memberships[len(memberships)//2:])
                    
                    antecedent[feat_idx] = best_set
                
                rule = FuzzyRule(tuple(antecedent), self.class_labels[predicted_class], 
                               confidence, support)
                rules.append(rule)
            else:
                feature = tree.feature[node_id]
                threshold = tree.threshold[node_id]
                
                # Left child (<=)
                extract_rules(tree.children_left[node_id], 
                            path_conditions + [(feature, threshold, 'left')])
                # Right child (>)
                extract_rules(tree.children_right[node_id], 
                            path_conditions + [(feature, threshold, 'right')])
        
        extract_rules(0, [])
        return rules
    
    def _hybrid_method(self, X: np.ndarray, y: np.ndarray) -> List[FuzzyRule]:
        """
        Hybrid method combining multiple rule generation approaches.
        Best for datasets with low accuracy - combines strengths of different methods.
        """
        # Generate rules using all methods
        wm_rules = self._wang_mendel(X, y)
        cluster_rules = self._clustering_based(X, y)
        dt_rules = self._decision_tree_based(X, y)
        
        # Combine all rules
        all_rules = wm_rules + cluster_rules + dt_rules
        
        # Remove duplicates and resolve conflicts
        rules = self._resolve_conflicts(all_rules)
        
        # Recalculate weights based on training data performance
        rules = self._recalculate_weights(rules, X, y)
        
        return rules
    
    def _exhaustive_generation(self, X: np.ndarray, y: np.ndarray) -> List[FuzzyRule]:
        """
        Generate rules by examining all possible antecedent combinations.
        More thorough but computationally expensive.
        """
        from itertools import product
        
        n_partitions = self.mf_manager.n_partitions
        
        # Limit to avoid combinatorial explosion
        if self.n_features > 6:
            warnings.warn("Too many features for exhaustive generation, using hybrid method")
            return self._hybrid_method(X, y)
        
        # Fuzzify all data
        fuzzified = self.mf_manager.fuzzify_batch(X)
        
        rules = []
        
        # Generate all possible antecedents
        for antecedent in product(range(n_partitions), repeat=self.n_features):
            # Calculate matching degree for all samples
            matching_degrees = np.ones(len(X))
            for feat_idx, fuzzy_set_idx in enumerate(antecedent):
                matching_degrees *= fuzzified[:, feat_idx, fuzzy_set_idx]
            
            # Skip if no samples match
            if np.sum(matching_degrees) < 0.01:
                continue
            
            # Calculate class weights
            class_weights = {}
            class_support = {}
            for class_label in self.class_labels:
                class_mask = y == class_label
                class_weights[class_label] = np.sum(matching_degrees[class_mask])
                class_support[class_label] = np.sum(class_mask & (matching_degrees > 0.1))
            
            # Select best class
            total_weight = sum(class_weights.values())
            if total_weight > 0:
                best_class = max(class_weights.keys(), key=lambda c: class_weights[c])
                confidence = class_weights[best_class] / total_weight
                support = class_support[best_class]
                
                if confidence > 0.5 and support > 0:
                    rule = FuzzyRule(antecedent, best_class, confidence, support)
                    rules.append(rule)
        
        return rules
    
    def _resolve_conflicts(self, rules: List[FuzzyRule]) -> List[FuzzyRule]:
        """Resolve conflicting rules (same antecedent, different consequent)."""
        rule_dict = {}
        
        for rule in rules:
            key = rule.antecedent
            if key not in rule_dict:
                rule_dict[key] = rule
            else:
                # Keep rule with higher weight * support
                existing = rule_dict[key]
                if rule.weight * rule.support > existing.weight * existing.support:
                    rule_dict[key] = rule
        
        return list(rule_dict.values())
    
    def _recalculate_weights(self, rules: List[FuzzyRule], 
                             X: np.ndarray, y: np.ndarray) -> List[FuzzyRule]:
        """Recalculate rule weights based on training data performance."""
        fuzzified = self.mf_manager.fuzzify_batch(X)
        
        for rule in rules:
            correct = 0
            total = 0
            
            for i in range(len(X)):
                matching = rule.matches(fuzzified[i])
                if matching > 0.1:
                    total += matching
                    if y[i] == rule.consequent:
                        correct += matching
            
            if total > 0:
                rule.weight = correct / total
                rule.support = int(total)
        
        return rules
    
    def add_rule(self, rule: FuzzyRule):
        """Add a rule to the rule base."""
        self.rules.append(rule)
    
    def remove_rule(self, rule: FuzzyRule):
        """Remove a rule from the rule base."""
        if rule in self.rules:
            self.rules.remove(rule)
    
    def get_rules(self) -> List[FuzzyRule]:
        """Get all rules."""
        return self.rules
    
    def set_rules(self, rules: List[FuzzyRule]):
        """Set the rule base."""
        self.rules = rules
    
    def print_rules(self, n: int = None, feature_names: List[str] = None,
                    class_names: List[str] = None):
        """Print rules in human-readable format."""
        rules_to_print = self.rules[:n] if n else self.rules
        linguistic_labels = self.mf_manager.linguistic_labels
        
        print(f"\n{'='*60}")
        print(f"FUZZY RULE BASE ({len(rules_to_print)} rules)")
        print(f"{'='*60}")
        
        for i, rule in enumerate(rules_to_print):
            print(f"R{i+1}: {rule.to_string(feature_names, linguistic_labels, class_names)}")
        
        print(f"{'='*60}\n")


class AdaptiveRuleGenerator(RuleGenerator):
    """
    Advanced rule generator with adaptive techniques for low-accuracy datasets.
    """
    
    def __init__(self, mf_manager, method: str = 'adaptive'):
        super().__init__(mf_manager, method)
        self.feature_importance = None
    
    def generate_rules(self, X: np.ndarray, y: np.ndarray,
                       feature_names: List[str] = None,
                       max_rules: int = None,
                       min_support: int = 1,
                       min_confidence: float = 0.0) -> List[FuzzyRule]:
        """Generate rules with adaptive feature selection."""
        self.n_features = X.shape[1]
        self.class_labels = np.unique(y)
        self.n_classes = len(self.class_labels)
        
        # Calculate feature importance
        self._calculate_feature_importance(X, y)
        
        # Generate rules using selected features
        if self.method == 'adaptive':
            rules = self._adaptive_generation(X, y)
        else:
            rules = super().generate_rules(X, y, feature_names, max_rules, 
                                          min_support, min_confidence)
        
        # Post-process rules
        rules = self._prune_rules(rules, X, y)
        rules = self._boost_minority_class_rules(rules, y)
        
        self.rules = rules
        return rules
    
    def _calculate_feature_importance(self, X: np.ndarray, y: np.ndarray):
        """Calculate feature importance using mutual information."""
        from sklearn.feature_selection import mutual_info_classif
        
        self.feature_importance = mutual_info_classif(X, y, random_state=42)
        self.feature_importance = self.feature_importance / np.sum(self.feature_importance)
    
    def _adaptive_generation(self, X: np.ndarray, y: np.ndarray) -> List[FuzzyRule]:
        """Generate rules focusing on important features."""
        # Select top features
        n_selected = max(2, int(self.n_features * 0.7))
        selected_features = np.argsort(self.feature_importance)[-n_selected:]
        
        # Generate rules using hybrid method on selected features
        rules = self._hybrid_method(X, y)
        
        # Also generate rules with "don't care" for less important features
        additional_rules = self._generate_partial_rules(X, y, selected_features)
        
        all_rules = rules + additional_rules
        return self._resolve_conflicts(all_rules)
    
    def _generate_partial_rules(self, X: np.ndarray, y: np.ndarray,
                                 important_features: np.ndarray) -> List[FuzzyRule]:
        """Generate rules using only important features."""
        from itertools import product
        
        n_partitions = self.mf_manager.n_partitions
        fuzzified = self.mf_manager.fuzzify_batch(X)
        
        rules = []
        
        # Generate rules for combinations of important features only
        for antecedent_partial in product(range(n_partitions), repeat=len(important_features)):
            # Create full antecedent with -1 for unimportant features
            antecedent = [-1] * self.n_features
            for i, feat_idx in enumerate(important_features):
                antecedent[feat_idx] = antecedent_partial[i]
            
            # Calculate matching degree
            matching_degrees = np.ones(len(X))
            for feat_idx in important_features:
                fuzzy_set_idx = antecedent[feat_idx]
                matching_degrees *= fuzzified[:, feat_idx, fuzzy_set_idx]
            
            if np.sum(matching_degrees) < 0.01:
                continue
            
            # Calculate class weights
            class_weights = {}
            for class_label in self.class_labels:
                class_mask = y == class_label
                class_weights[class_label] = np.sum(matching_degrees[class_mask])
            
            total_weight = sum(class_weights.values())
            if total_weight > 0:
                best_class = max(class_weights.keys(), key=lambda c: class_weights[c])
                confidence = class_weights[best_class] / total_weight
                support = int(np.sum(matching_degrees > 0.1))
                
                if confidence > 0.5 and support > 0:
                    rule = FuzzyRule(tuple(antecedent), best_class, confidence, support)
                    rules.append(rule)
        
        return rules
    
    def _prune_rules(self, rules: List[FuzzyRule], 
                     X: np.ndarray, y: np.ndarray) -> List[FuzzyRule]:
        """Remove redundant or low-quality rules."""
        if len(rules) <= 10:
            return rules
        
        # Sort by quality (weight * support)
        rules = sorted(rules, key=lambda r: r.weight * r.support, reverse=True)
        
        # Keep rules that contribute to accuracy
        fuzzified = self.mf_manager.fuzzify_batch(X)
        kept_rules = []
        covered_samples = set()
        
        for rule in rules:
            # Check if rule covers new samples correctly
            new_correct = 0
            for i in range(len(X)):
                if i in covered_samples:
                    continue
                matching = rule.matches(fuzzified[i])
                if matching > 0.1 and y[i] == rule.consequent:
                    new_correct += 1
                    covered_samples.add(i)
            
            if new_correct > 0 or len(kept_rules) < 10:
                kept_rules.append(rule)
        
        return kept_rules
    
    def _boost_minority_class_rules(self, rules: List[FuzzyRule], 
                                     y: np.ndarray) -> List[FuzzyRule]:
        """Boost weights of rules for minority classes."""
        class_counts = np.bincount(y.astype(int))
        max_count = np.max(class_counts)
        
        for rule in rules:
            class_count = class_counts[int(rule.consequent)]
            boost_factor = np.sqrt(max_count / class_count)
            rule.weight *= boost_factor
        
        return rules
