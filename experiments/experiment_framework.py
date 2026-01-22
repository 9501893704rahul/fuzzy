"""
Experiment Framework
Provides utilities for running systematic experiments on fuzzy classifiers.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Callable
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report,
                            roc_auc_score)
from sklearn.preprocessing import LabelEncoder
import time
import warnings
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fuzzy_classifier import FuzzyRuleClassifier, EnsembleFuzzyClassifier


class ExperimentFramework:
    """
    Framework for running experiments on fuzzy classifiers.
    Supports multiple datasets, configurations, and comparison with baselines.
    """
    
    def __init__(self, random_state: int = 42, n_folds: int = 5):
        """
        Args:
            random_state: Random seed for reproducibility
            n_folds: Number of cross-validation folds
        """
        self.random_state = random_state
        self.n_folds = n_folds
        self.results = {}
        self.datasets = {}
    
    def load_dataset(self, name: str, X: np.ndarray, y: np.ndarray,
                     feature_names: List[str] = None) -> None:
        """
        Load a dataset for experiments.
        
        Args:
            name: Dataset name
            X: Feature matrix
            y: Target labels
            feature_names: Optional feature names
        """
        # Encode labels if necessary
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        self.datasets[name] = {
            'X': np.asarray(X),
            'y': y_encoded,
            'feature_names': feature_names or [f'X{i}' for i in range(X.shape[1])],
            'class_names': list(le.classes_),
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'n_classes': len(np.unique(y_encoded))
        }
        
        print(f"Loaded dataset '{name}': {X.shape[0]} samples, {X.shape[1]} features, "
              f"{len(np.unique(y_encoded))} classes")
    
    def run_experiment(self, dataset_name: str, 
                       classifier_config: Dict = None,
                       experiment_name: str = None) -> Dict:
        """
        Run a single experiment on a dataset.
        
        Args:
            dataset_name: Name of loaded dataset
            classifier_config: Configuration for FuzzyRuleClassifier
            experiment_name: Name for this experiment
            
        Returns:
            Dictionary of results
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not loaded")
        
        data = self.datasets[dataset_name]
        X, y = data['X'], data['y']
        feature_names = data['feature_names']
        
        config = classifier_config or {}
        exp_name = experiment_name or f"{dataset_name}_default"
        
        print(f"\n{'='*60}")
        print(f"Running experiment: {exp_name}")
        print(f"{'='*60}")
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, 
                            random_state=self.random_state)
        
        fold_results = []
        all_predictions = []
        all_true = []
        
        start_time = time.time()
        
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Create and train classifier
            clf = FuzzyRuleClassifier(random_state=self.random_state, **config)
            clf.fit(X_train, y_train, feature_names)
            
            # Predict
            y_pred = clf.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            fold_results.append({
                'fold': fold + 1,
                'accuracy': accuracy,
                'n_rules': len(clf.rules)
            })
            
            all_predictions.extend(y_pred)
            all_true.extend(y_test)
            
            print(f"  Fold {fold+1}: Accuracy = {accuracy:.4f}, Rules = {len(clf.rules)}")
        
        total_time = time.time() - start_time
        
        # Aggregate results
        all_predictions = np.array(all_predictions)
        all_true = np.array(all_true)
        
        results = {
            'experiment_name': exp_name,
            'dataset': dataset_name,
            'config': config,
            'cv_accuracy_mean': np.mean([r['accuracy'] for r in fold_results]),
            'cv_accuracy_std': np.std([r['accuracy'] for r in fold_results]),
            'cv_rules_mean': np.mean([r['n_rules'] for r in fold_results]),
            'precision': precision_score(all_true, all_predictions, average='weighted', zero_division=0),
            'recall': recall_score(all_true, all_predictions, average='weighted', zero_division=0),
            'f1': f1_score(all_true, all_predictions, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(all_true, all_predictions),
            'fold_results': fold_results,
            'total_time': total_time
        }
        
        print(f"\nResults:")
        print(f"  CV Accuracy: {results['cv_accuracy_mean']:.4f} (+/- {results['cv_accuracy_std']:.4f})")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall: {results['recall']:.4f}")
        print(f"  F1 Score: {results['f1']:.4f}")
        print(f"  Avg Rules: {results['cv_rules_mean']:.1f}")
        print(f"  Time: {total_time:.2f}s")
        
        self.results[exp_name] = results
        return results
    
    def run_comparison_experiment(self, dataset_name: str,
                                  configurations: Dict[str, Dict]) -> pd.DataFrame:
        """
        Run multiple configurations and compare results.
        
        Args:
            dataset_name: Name of loaded dataset
            configurations: Dict of {config_name: config_dict}
            
        Returns:
            DataFrame with comparison results
        """
        comparison_results = []
        
        for config_name, config in configurations.items():
            exp_name = f"{dataset_name}_{config_name}"
            results = self.run_experiment(dataset_name, config, exp_name)
            
            comparison_results.append({
                'Configuration': config_name,
                'Accuracy': f"{results['cv_accuracy_mean']:.4f} ± {results['cv_accuracy_std']:.4f}",
                'Precision': f"{results['precision']:.4f}",
                'Recall': f"{results['recall']:.4f}",
                'F1': f"{results['f1']:.4f}",
                'Rules': f"{results['cv_rules_mean']:.1f}",
                'Time (s)': f"{results['total_time']:.2f}"
            })
        
        df = pd.DataFrame(comparison_results)
        print(f"\n{'='*80}")
        print(f"COMPARISON RESULTS FOR {dataset_name}")
        print(f"{'='*80}")
        print(df.to_string(index=False))
        
        return df
    
    def run_baseline_comparison(self, dataset_name: str,
                                fuzzy_config: Dict = None) -> pd.DataFrame:
        """
        Compare fuzzy classifier with traditional ML baselines.
        
        Args:
            dataset_name: Name of loaded dataset
            fuzzy_config: Configuration for fuzzy classifier
            
        Returns:
            DataFrame with comparison results
        """
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.svm import SVC
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.neural_network import MLPClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.linear_model import LogisticRegression
        
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not loaded")
        
        data = self.datasets[dataset_name]
        X, y = data['X'], data['y']
        
        # Define baselines
        baselines = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            'SVM (RBF)': SVC(kernel='rbf', random_state=self.random_state),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'Decision Tree': DecisionTreeClassifier(random_state=self.random_state),
            'Logistic Regression': LogisticRegression(random_state=self.random_state, max_iter=1000),
            'MLP': MLPClassifier(hidden_layer_sizes=(100,), random_state=self.random_state, max_iter=1000),
            'Gradient Boosting': GradientBoostingClassifier(random_state=self.random_state)
        }
        
        cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, 
                            random_state=self.random_state)
        
        results = []
        
        # Run fuzzy classifier
        print(f"\n{'='*60}")
        print(f"BASELINE COMPARISON FOR {dataset_name}")
        print(f"{'='*60}")
        
        config = fuzzy_config or {
            'n_partitions': 5,
            'rule_method': 'hybrid',
            'optimize': True,
            'optimization_method': 'adaptive'
        }
        
        print("\nFuzzy Classifier...")
        fuzzy_scores = []
        start_time = time.time()
        
        for train_idx, test_idx in cv.split(X, y):
            clf = FuzzyRuleClassifier(random_state=self.random_state, **config)
            clf.fit(X[train_idx], y[train_idx], data['feature_names'])
            score = clf.score(X[test_idx], y[test_idx])
            fuzzy_scores.append(score)
        
        fuzzy_time = time.time() - start_time
        
        results.append({
            'Classifier': 'Fuzzy RBCS',
            'Accuracy': f"{np.mean(fuzzy_scores):.4f} ± {np.std(fuzzy_scores):.4f}",
            'Interpretable': 'Yes',
            'Time (s)': f"{fuzzy_time:.2f}"
        })
        
        # Run ensemble fuzzy
        print("Ensemble Fuzzy Classifier...")
        ensemble_scores = []
        start_time = time.time()
        
        for train_idx, test_idx in cv.split(X, y):
            clf = EnsembleFuzzyClassifier(n_estimators=3, random_state=self.random_state)
            clf.fit(X[train_idx], y[train_idx], data['feature_names'])
            score = clf.score(X[test_idx], y[test_idx])
            ensemble_scores.append(score)
        
        ensemble_time = time.time() - start_time
        
        results.append({
            'Classifier': 'Ensemble Fuzzy',
            'Accuracy': f"{np.mean(ensemble_scores):.4f} ± {np.std(ensemble_scores):.4f}",
            'Interpretable': 'Partial',
            'Time (s)': f"{ensemble_time:.2f}"
        })
        
        # Run baselines
        for name, clf in baselines.items():
            print(f"{name}...")
            start_time = time.time()
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
            
            elapsed = time.time() - start_time
            
            results.append({
                'Classifier': name,
                'Accuracy': f"{np.mean(scores):.4f} ± {np.std(scores):.4f}",
                'Interpretable': 'Yes' if name == 'Decision Tree' else 'No',
                'Time (s)': f"{elapsed:.2f}"
            })
        
        df = pd.DataFrame(results)
        print(f"\n{'='*80}")
        print(df.to_string(index=False))
        
        return df
    
    def run_sensitivity_analysis(self, dataset_name: str,
                                 parameter: str,
                                 values: List) -> pd.DataFrame:
        """
        Analyze sensitivity to a specific parameter.
        
        Args:
            dataset_name: Name of loaded dataset
            parameter: Parameter name to vary
            values: List of values to test
            
        Returns:
            DataFrame with sensitivity results
        """
        results = []
        
        print(f"\n{'='*60}")
        print(f"SENSITIVITY ANALYSIS: {parameter}")
        print(f"{'='*60}")
        
        for value in values:
            config = {parameter: value}
            exp_name = f"{dataset_name}_{parameter}_{value}"
            
            result = self.run_experiment(dataset_name, config, exp_name)
            
            results.append({
                parameter: value,
                'Accuracy': result['cv_accuracy_mean'],
                'Std': result['cv_accuracy_std'],
                'Rules': result['cv_rules_mean'],
                'Time': result['total_time']
            })
        
        df = pd.DataFrame(results)
        print(f"\n{df.to_string(index=False)}")
        
        return df
    
    def run_robustness_test(self, dataset_name: str,
                            noise_levels: List[float] = None,
                            missing_rates: List[float] = None) -> Dict:
        """
        Test classifier robustness to noise and missing data.
        
        Args:
            dataset_name: Name of loaded dataset
            noise_levels: List of noise standard deviations to test
            missing_rates: List of missing data rates to test
            
        Returns:
            Dictionary with robustness results
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not loaded")
        
        data = self.datasets[dataset_name]
        X, y = data['X'].copy(), data['y'].copy()
        
        noise_levels = noise_levels or [0.0, 0.1, 0.2, 0.3, 0.5]
        missing_rates = missing_rates or [0.0, 0.05, 0.1, 0.2, 0.3]
        
        results = {'noise': [], 'missing': []}
        
        print(f"\n{'='*60}")
        print(f"ROBUSTNESS TEST FOR {dataset_name}")
        print(f"{'='*60}")
        
        # Noise test
        print("\nNoise Robustness:")
        for noise in noise_levels:
            X_noisy = X + np.random.normal(0, noise, X.shape)
            
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
            scores = []
            
            for train_idx, test_idx in cv.split(X_noisy, y):
                clf = FuzzyRuleClassifier(random_state=self.random_state, optimize=True)
                clf.fit(X_noisy[train_idx], y[train_idx])
                scores.append(clf.score(X_noisy[test_idx], y[test_idx]))
            
            results['noise'].append({
                'noise_level': noise,
                'accuracy': np.mean(scores),
                'std': np.std(scores)
            })
            print(f"  Noise σ={noise:.2f}: Accuracy = {np.mean(scores):.4f} ± {np.std(scores):.4f}")
        
        # Missing data test
        print("\nMissing Data Robustness:")
        for rate in missing_rates:
            X_missing = X.copy()
            mask = np.random.random(X.shape) < rate
            X_missing[mask] = np.nan
            
            # Simple imputation with mean
            col_means = np.nanmean(X_missing, axis=0)
            for j in range(X_missing.shape[1]):
                X_missing[np.isnan(X_missing[:, j]), j] = col_means[j]
            
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
            scores = []
            
            for train_idx, test_idx in cv.split(X_missing, y):
                clf = FuzzyRuleClassifier(random_state=self.random_state, optimize=True)
                clf.fit(X_missing[train_idx], y[train_idx])
                scores.append(clf.score(X_missing[test_idx], y[test_idx]))
            
            results['missing'].append({
                'missing_rate': rate,
                'accuracy': np.mean(scores),
                'std': np.std(scores)
            })
            print(f"  Missing {rate*100:.0f}%: Accuracy = {np.mean(scores):.4f} ± {np.std(scores):.4f}")
        
        return results
    
    def get_summary(self) -> pd.DataFrame:
        """Get summary of all experiments."""
        summary = []
        
        for name, result in self.results.items():
            summary.append({
                'Experiment': name,
                'Dataset': result['dataset'],
                'Accuracy': f"{result['cv_accuracy_mean']:.4f} ± {result['cv_accuracy_std']:.4f}",
                'F1': f"{result['f1']:.4f}",
                'Rules': f"{result['cv_rules_mean']:.1f}",
                'Time': f"{result['total_time']:.2f}s"
            })
        
        return pd.DataFrame(summary)
