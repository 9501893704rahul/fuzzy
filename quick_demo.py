"""
Quick Demo - Fuzzy Rule-Based Classification System
Demonstrates the system on a low-accuracy dataset with faster settings.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import warnings
import time

warnings.filterwarnings('ignore')

from src.fuzzy_classifier import FuzzyRuleClassifier, EnsembleFuzzyClassifier


def load_pima_diabetes():
    """Load Pima Indians Diabetes dataset - known for low accuracy (~75%)."""
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    try:
        columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                   'Insulin', 'BMI', 'DiabetesPedigree', 'Age', 'Outcome']
        df = pd.read_csv(url, names=columns)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        return X, y, columns[:-1]
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Generating synthetic low-accuracy data instead...")
        return generate_synthetic_data()


def generate_synthetic_data(n_samples=500, n_features=8):
    """Generate synthetic dataset with overlapping classes (difficult to classify)."""
    np.random.seed(42)
    
    n_per_class = n_samples // 2
    
    # Create overlapping distributions
    X0 = np.random.randn(n_per_class, n_features) * 1.2
    X1 = np.random.randn(n_per_class, n_features) * 1.2 + 0.6
    
    X = np.vstack([X0, X1])
    y = np.array([0] * n_per_class + [1] * n_per_class)
    
    # Shuffle
    idx = np.random.permutation(n_samples)
    X, y = X[idx], y[idx]
    
    feature_names = [f'Feature_{i}' for i in range(n_features)]
    return X, y, feature_names


def main():
    print("=" * 70)
    print("FUZZY RULE-BASED CLASSIFICATION SYSTEM")
    print("Demo on Low-Accuracy Dataset (Pima Indians Diabetes)")
    print("=" * 70)
    
    # Load data
    print("\n[1] Loading Dataset...")
    X, y, feature_names = load_pima_diabetes()
    
    print(f"    Dataset: Pima Indians Diabetes")
    print(f"    Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"    Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"    (This dataset typically achieves ~75% accuracy with ML methods)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # =========================================================================
    # Test 1: Basic Fuzzy Classifier (No Optimization)
    # =========================================================================
    print("\n" + "=" * 70)
    print("[2] BASIC FUZZY CLASSIFIER (No GA Optimization)")
    print("=" * 70)
    
    start = time.time()
    clf_basic = FuzzyRuleClassifier(
        n_partitions=5,
        mf_type='triangular',
        partition_method='uniform',
        rule_method='wang_mendel',
        optimize=False,
        verbose=False
    )
    clf_basic.fit(X_train, y_train, feature_names)
    basic_time = time.time() - start
    
    train_acc = clf_basic.score(X_train, y_train)
    test_acc = clf_basic.score(X_test, y_test)
    
    print(f"    Training Accuracy: {train_acc:.4f}")
    print(f"    Test Accuracy: {test_acc:.4f}")
    print(f"    Number of Rules: {len(clf_basic.rules)}")
    print(f"    Time: {basic_time:.2f}s")
    
    # =========================================================================
    # Test 2: Optimized Fuzzy Classifier
    # =========================================================================
    print("\n" + "=" * 70)
    print("[3] OPTIMIZED FUZZY CLASSIFIER (With GA)")
    print("=" * 70)
    
    start = time.time()
    clf_opt = FuzzyRuleClassifier(
        n_partitions=5,
        mf_type='gaussian',
        partition_method='adaptive',
        rule_method='hybrid',
        optimize=True,
        optimization_method='standard',
        n_generations=30,  # Reduced for speed
        population_size=50,
        max_rules=100,  # Limit rules for speed
        handle_imbalance=True,
        verbose=False
    )
    clf_opt.fit(X_train, y_train, feature_names)
    opt_time = time.time() - start
    
    train_acc = clf_opt.score(X_train, y_train)
    test_acc = clf_opt.score(X_test, y_test)
    
    print(f"    Training Accuracy: {train_acc:.4f}")
    print(f"    Test Accuracy: {test_acc:.4f}")
    print(f"    Number of Rules: {len(clf_opt.rules)}")
    print(f"    Time: {opt_time:.2f}s")
    
    # =========================================================================
    # Test 3: Show Interpretable Rules
    # =========================================================================
    print("\n" + "=" * 70)
    print("[4] INTERPRETABLE FUZZY RULES (Top 10)")
    print("=" * 70)
    clf_opt.print_rules(n=10)
    
    # =========================================================================
    # Test 4: Feature Importance
    # =========================================================================
    print("\n" + "=" * 70)
    print("[5] FEATURE IMPORTANCE")
    print("=" * 70)
    importance = clf_opt.get_feature_importance()
    for feat, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(imp * 50)
        print(f"    {feat:20s}: {imp:.4f} {bar}")
    
    # =========================================================================
    # Test 5: Comparison with ML Baselines
    # =========================================================================
    print("\n" + "=" * 70)
    print("[6] COMPARISON WITH ML BASELINES")
    print("=" * 70)
    
    results = []
    
    # Fuzzy Classifier
    results.append({
        'Classifier': 'Fuzzy RBCS (Basic)',
        'Test Accuracy': clf_basic.score(X_test, y_test),
        'Interpretable': 'Yes'
    })
    
    results.append({
        'Classifier': 'Fuzzy RBCS (Optimized)',
        'Test Accuracy': clf_opt.score(X_test, y_test),
        'Interpretable': 'Yes'
    })
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    results.append({
        'Classifier': 'Random Forest',
        'Test Accuracy': rf.score(X_test, y_test),
        'Interpretable': 'No'
    })
    
    # SVM
    svm = SVC(kernel='rbf', random_state=42)
    svm.fit(X_train, y_train)
    results.append({
        'Classifier': 'SVM (RBF)',
        'Test Accuracy': svm.score(X_test, y_test),
        'Interpretable': 'No'
    })
    
    # Decision Tree
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    results.append({
        'Classifier': 'Decision Tree',
        'Test Accuracy': dt.score(X_test, y_test),
        'Interpretable': 'Yes'
    })
    
    df = pd.DataFrame(results)
    df['Test Accuracy'] = df['Test Accuracy'].apply(lambda x: f"{x:.4f}")
    print(df.to_string(index=False))
    
    # =========================================================================
    # Test 6: Different Configurations Comparison
    # =========================================================================
    print("\n" + "=" * 70)
    print("[7] CONFIGURATION COMPARISON")
    print("=" * 70)
    
    configs = {
        'Wang-Mendel': {'rule_method': 'wang_mendel', 'optimize': False},
        'Clustering': {'rule_method': 'clustering', 'optimize': False},
        'Hybrid': {'rule_method': 'hybrid', 'optimize': False},
        'Hybrid+GA': {'rule_method': 'hybrid', 'optimize': True, 'n_generations': 20, 'max_rules': 80},
    }
    
    config_results = []
    for name, config in configs.items():
        clf = FuzzyRuleClassifier(n_partitions=5, verbose=False, **config)
        clf.fit(X_train, y_train, feature_names)
        
        config_results.append({
            'Configuration': name,
            'Train Acc': f"{clf.score(X_train, y_train):.4f}",
            'Test Acc': f"{clf.score(X_test, y_test):.4f}",
            'Rules': len(clf.rules)
        })
    
    df_config = pd.DataFrame(config_results)
    print(df_config.to_string(index=False))
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
    The Fuzzy Rule-Based Classification System provides:
    
    ✓ Competitive accuracy on low-accuracy datasets
    ✓ Interpretable IF-THEN rules for decision explanation
    ✓ Multiple rule generation methods (Wang-Mendel, Clustering, Hybrid)
    ✓ Genetic Algorithm optimization for improved performance
    ✓ Adaptive membership function partitioning
    ✓ Class imbalance handling
    
    Key advantage: Unlike black-box models, fuzzy rules can be understood
    and validated by domain experts (e.g., medical professionals).
    """)
    
    return clf_opt


if __name__ == '__main__':
    clf = main()
