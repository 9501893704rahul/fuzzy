"""
Run Experiments Script
Main script for running all experiments on medical datasets.
Focuses on datasets with typically low accuracy.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.preprocessing import StandardScaler
import warnings
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.experiment_framework import ExperimentFramework
from src.fuzzy_classifier import FuzzyRuleClassifier, EnsembleFuzzyClassifier

warnings.filterwarnings('ignore')


def load_pima_diabetes():
    """Load Pima Indians Diabetes dataset - known for low accuracy."""
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    try:
        columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                   'Insulin', 'BMI', 'DiabetesPedigree', 'Age', 'Outcome']
        df = pd.read_csv(url, names=columns)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        return X, y, columns[:-1], ['No Diabetes', 'Diabetes']
    except:
        print("Could not load Pima dataset from URL, generating synthetic data...")
        return generate_synthetic_low_accuracy_data()


def load_heart_disease():
    """Load Cleveland Heart Disease dataset."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    try:
        columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                   'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
        df = pd.read_csv(url, names=columns, na_values='?')
        df = df.dropna()
        X = df.iloc[:, :-1].values
        y = (df.iloc[:, -1].values > 0).astype(int)  # Binary: disease or not
        return X, y, columns[:-1], ['No Disease', 'Disease']
    except:
        print("Could not load Heart Disease dataset, using breast cancer instead...")
        data = load_breast_cancer()
        return data.data, data.target, list(data.feature_names), list(data.target_names)


def load_hepatitis():
    """Load Hepatitis dataset - small dataset with low accuracy."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/hepatitis/hepatitis.data"
    try:
        columns = ['Class', 'AGE', 'SEX', 'STEROID', 'ANTIVIRALS', 'FATIGUE', 
                   'MALAISE', 'ANOREXIA', 'LIVER_BIG', 'LIVER_FIRM', 'SPLEEN_PALPABLE',
                   'SPIDERS', 'ASCITES', 'VARICES', 'BILIRUBIN', 'ALK_PHOSPHATE',
                   'SGOT', 'ALBUMIN', 'PROTIME', 'HISTOLOGY']
        df = pd.read_csv(url, names=columns, na_values='?')
        df = df.dropna()
        X = df.iloc[:, 1:].values
        y = df.iloc[:, 0].values - 1  # Convert to 0/1
        return X, y, columns[1:], ['Die', 'Live']
    except:
        print("Could not load Hepatitis dataset, generating synthetic data...")
        return generate_synthetic_low_accuracy_data(n_samples=150, n_features=19)


def generate_synthetic_low_accuracy_data(n_samples=500, n_features=8, noise=0.3):
    """
    Generate synthetic dataset that is difficult to classify.
    Simulates characteristics of low-accuracy medical datasets.
    """
    np.random.seed(42)
    
    # Create overlapping class distributions
    n_per_class = n_samples // 2
    
    # Class 0: centered around origin with spread
    X0 = np.random.randn(n_per_class, n_features) * 1.5
    
    # Class 1: slightly shifted but overlapping
    X1 = np.random.randn(n_per_class, n_features) * 1.5 + 0.5
    
    X = np.vstack([X0, X1])
    y = np.array([0] * n_per_class + [1] * n_per_class)
    
    # Add noise
    X += np.random.randn(*X.shape) * noise
    
    # Add some irrelevant features
    X[:, -2:] = np.random.randn(n_samples, 2)
    
    # Shuffle
    idx = np.random.permutation(n_samples)
    X, y = X[idx], y[idx]
    
    feature_names = [f'Feature_{i}' for i in range(n_features)]
    class_names = ['Class_0', 'Class_1']
    
    return X, y, feature_names, class_names


def run_all_experiments():
    """Run comprehensive experiments on all datasets."""
    
    print("="*80)
    print("FUZZY RULE-BASED CLASSIFICATION SYSTEM - EXPERIMENTS")
    print("Focus: Datasets with Low Baseline Accuracy")
    print("="*80)
    
    framework = ExperimentFramework(random_state=42, n_folds=5)
    
    # Load datasets
    print("\n" + "="*60)
    print("LOADING DATASETS")
    print("="*60)
    
    # Pima Diabetes - known for ~75% accuracy ceiling
    X_pima, y_pima, feat_pima, class_pima = load_pima_diabetes()
    framework.load_dataset('pima_diabetes', X_pima, y_pima, feat_pima)
    
    # Heart Disease
    X_heart, y_heart, feat_heart, class_heart = load_heart_disease()
    framework.load_dataset('heart_disease', X_heart, y_heart, feat_heart)
    
    # Synthetic low-accuracy dataset
    X_synth, y_synth, feat_synth, class_synth = generate_synthetic_low_accuracy_data()
    framework.load_dataset('synthetic_difficult', X_synth, y_synth, feat_synth)
    
    # =========================================================================
    # EXPERIMENT 1: Rule Generation Method Comparison
    # =========================================================================
    print("\n" + "="*80)
    print("EXPERIMENT 1: RULE GENERATION METHOD COMPARISON")
    print("="*80)
    
    rule_methods = {
        'Wang-Mendel': {'rule_method': 'wang_mendel', 'optimize': False},
        'Clustering': {'rule_method': 'clustering', 'optimize': False},
        'Decision Tree': {'rule_method': 'decision_tree', 'optimize': False},
        'Hybrid': {'rule_method': 'hybrid', 'optimize': False},
        'Hybrid + GA': {'rule_method': 'hybrid', 'optimize': True, 'optimization_method': 'standard'},
        'Hybrid + Adaptive GA': {'rule_method': 'hybrid', 'optimize': True, 'optimization_method': 'adaptive'}
    }
    
    for dataset in ['pima_diabetes', 'synthetic_difficult']:
        print(f"\n--- Dataset: {dataset} ---")
        framework.run_comparison_experiment(dataset, rule_methods)
    
    # =========================================================================
    # EXPERIMENT 2: Partition Number Sensitivity
    # =========================================================================
    print("\n" + "="*80)
    print("EXPERIMENT 2: PARTITION NUMBER SENSITIVITY")
    print("="*80)
    
    framework.run_sensitivity_analysis('pima_diabetes', 'n_partitions', [3, 5, 7, 9])
    
    # =========================================================================
    # EXPERIMENT 3: Membership Function Type Comparison
    # =========================================================================
    print("\n" + "="*80)
    print("EXPERIMENT 3: MEMBERSHIP FUNCTION TYPE COMPARISON")
    print("="*80)
    
    mf_configs = {
        'Triangular': {'mf_type': 'triangular', 'optimize': True},
        'Gaussian': {'mf_type': 'gaussian', 'optimize': True},
        'Trapezoidal': {'mf_type': 'trapezoidal', 'optimize': True}
    }
    
    framework.run_comparison_experiment('pima_diabetes', mf_configs)
    
    # =========================================================================
    # EXPERIMENT 4: Partitioning Method Comparison
    # =========================================================================
    print("\n" + "="*80)
    print("EXPERIMENT 4: PARTITIONING METHOD COMPARISON")
    print("="*80)
    
    partition_configs = {
        'Uniform': {'partition_method': 'uniform', 'optimize': True},
        'Quantile': {'partition_method': 'quantile', 'optimize': True},
        'K-Means': {'partition_method': 'kmeans', 'optimize': True},
        'Adaptive': {'partition_method': 'adaptive', 'optimize': True},
        'Class-Aware': {'partition_method': 'class_aware', 'optimize': True}
    }
    
    framework.run_comparison_experiment('pima_diabetes', partition_configs)
    
    # =========================================================================
    # EXPERIMENT 5: Baseline Comparison
    # =========================================================================
    print("\n" + "="*80)
    print("EXPERIMENT 5: COMPARISON WITH ML BASELINES")
    print("="*80)
    
    best_fuzzy_config = {
        'n_partitions': 5,
        'mf_type': 'gaussian',
        'partition_method': 'adaptive',
        'rule_method': 'hybrid',
        'optimize': True,
        'optimization_method': 'adaptive',
        'n_generations': 50
    }
    
    for dataset in ['pima_diabetes', 'heart_disease', 'synthetic_difficult']:
        framework.run_baseline_comparison(dataset, best_fuzzy_config)
    
    # =========================================================================
    # EXPERIMENT 6: Robustness Testing
    # =========================================================================
    print("\n" + "="*80)
    print("EXPERIMENT 6: ROBUSTNESS TESTING")
    print("="*80)
    
    framework.run_robustness_test('pima_diabetes')
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    
    summary = framework.get_summary()
    print(summary.to_string(index=False))
    
    return framework


def demo_single_dataset():
    """Quick demo on a single dataset."""
    print("="*60)
    print("FUZZY CLASSIFIER DEMO - LOW ACCURACY DATASET")
    print("="*60)
    
    # Load Pima Diabetes dataset
    X, y, feature_names, class_names = load_pima_diabetes()
    
    print(f"\nDataset: Pima Indians Diabetes")
    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train optimized fuzzy classifier
    print("\n--- Training Optimized Fuzzy Classifier ---")
    clf = FuzzyRuleClassifier(
        n_partitions=5,
        mf_type='gaussian',
        partition_method='adaptive',
        rule_method='hybrid',
        optimize=True,
        optimization_method='adaptive',
        n_generations=50,
        handle_imbalance=True,
        verbose=True
    )
    
    clf.fit(X_train, y_train, feature_names)
    
    # Evaluate
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    
    print(f"\nResults:")
    print(f"  Training Accuracy: {train_acc:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Number of Rules: {len(clf.rules)}")
    
    # Print top rules
    print("\n--- Top 10 Fuzzy Rules ---")
    clf.print_rules(n=10)
    
    # Feature importance
    print("\n--- Feature Importance ---")
    importance = clf.get_feature_importance()
    for feat, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feat}: {imp:.4f}")
    
    # Compare with ensemble
    print("\n--- Ensemble Fuzzy Classifier ---")
    ensemble = EnsembleFuzzyClassifier(n_estimators=5, verbose=True)
    ensemble.fit(X_train, y_train, feature_names)
    
    ensemble_train = ensemble.score(X_train, y_train)
    ensemble_test = ensemble.score(X_test, y_test)
    
    print(f"\nEnsemble Results:")
    print(f"  Training Accuracy: {ensemble_train:.4f}")
    print(f"  Test Accuracy: {ensemble_test:.4f}")
    
    return clf, ensemble


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Fuzzy Classifier Experiments')
    parser.add_argument('--demo', action='store_true', help='Run quick demo')
    parser.add_argument('--full', action='store_true', help='Run full experiments')
    
    args = parser.parse_args()
    
    if args.demo:
        demo_single_dataset()
    elif args.full:
        run_all_experiments()
    else:
        # Default: run demo
        demo_single_dataset()
