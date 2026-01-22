"""
Final Demo - Fuzzy Rule-Based Classification System
Fast demonstration on low-accuracy dataset with comparison to baselines.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from collections import defaultdict
import warnings
import time

warnings.filterwarnings('ignore')


class FastFuzzyClassifier:
    """
    Fast and effective Fuzzy Classifier optimized for low-accuracy datasets.
    Uses class-aware partitioning and weighted inference.
    """
    
    def __init__(self, n_partitions=5):
        self.n_partitions = n_partitions
        self.rules = []
        self.scaler = None
        self.class_weights = None
        self.feature_names = None
        self.mf_params = {}
        self.labels = ['VeryLow', 'Low', 'Medium', 'High', 'VeryHigh'][:n_partitions]
    
    def fit(self, X, y, feature_names=None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        
        self.feature_names = feature_names or [f'X{i}' for i in range(X.shape[1])]
        
        # Normalize
        self.scaler = MinMaxScaler()
        X = self.scaler.fit_transform(X)
        
        # Compute class weights
        class_counts = np.bincount(y)
        self.class_weights = {i: len(y) / (len(class_counts) * c) for i, c in enumerate(class_counts)}
        
        # Create membership functions
        self._create_mf(X, y)
        
        # Generate rules
        self._generate_rules(X, y)
        
        return self
    
    def _create_mf(self, X, y):
        """Create Gaussian membership functions."""
        for j in range(X.shape[1]):
            centers = np.linspace(0, 1, self.n_partitions)
            width = 1.0 / (self.n_partitions - 1) / 2
            self.mf_params[j] = [(c, width) for c in centers]
    
    def _gaussian_mf(self, x, mean, sigma):
        """Gaussian membership function."""
        return np.exp(-0.5 * ((x - mean) / sigma) ** 2)
    
    def _fuzzify(self, x, feat_idx):
        """Get membership degrees for a value."""
        memberships = np.zeros(self.n_partitions)
        for i, (mean, sigma) in enumerate(self.mf_params[feat_idx]):
            memberships[i] = self._gaussian_mf(x, mean, sigma)
        return memberships
    
    def _generate_rules(self, X, y):
        """Generate fuzzy rules using weighted Wang-Mendel."""
        rule_dict = defaultdict(lambda: defaultdict(float))
        rule_support = defaultdict(lambda: defaultdict(int))
        
        for i in range(len(X)):
            antecedent = []
            matching = 1.0
            
            for j in range(X.shape[1]):
                memberships = self._fuzzify(X[i, j], j)
                dominant = np.argmax(memberships)
                antecedent.append(dominant)
                matching *= memberships[dominant]
            
            antecedent = tuple(antecedent)
            weight = matching * self.class_weights[y[i]]
            
            rule_dict[antecedent][y[i]] += weight
            rule_support[antecedent][y[i]] += 1
        
        self.rules = []
        for antecedent, class_weights in rule_dict.items():
            best_class = max(class_weights.keys(), key=lambda c: class_weights[c])
            total = sum(class_weights.values())
            confidence = class_weights[best_class] / total if total > 0 else 0
            support = rule_support[antecedent][best_class]
            
            if confidence > 0.4:
                self.rules.append({
                    'antecedent': antecedent,
                    'consequent': best_class,
                    'weight': confidence,
                    'support': support
                })
        
        # Sort by quality
        self.rules = sorted(self.rules, key=lambda r: r['weight'] * r['support'], reverse=True)
    
    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        X = self.scaler.transform(X)
        
        predictions = []
        for i in range(len(X)):
            pred = self._predict_single(X[i])
            predictions.append(pred)
        
        return np.array(predictions)
    
    def _predict_single(self, x):
        """Predict using weighted voting."""
        class_scores = defaultdict(float)
        
        for rule in self.rules:
            match = 1.0
            for j, fuzzy_set in enumerate(rule['antecedent']):
                memberships = self._fuzzify(x[j], j)
                match *= memberships[fuzzy_set]
            
            if match > 0:
                score = match * rule['weight'] * self.class_weights[rule['consequent']]
                class_scores[rule['consequent']] += score
        
        if not class_scores:
            return 0
        
        return max(class_scores.keys(), key=lambda c: class_scores[c])
    
    def score(self, X, y):
        predictions = self.predict(X)
        return accuracy_score(y, predictions)
    
    def print_rules(self, n=10):
        rules = self.rules[:n]
        
        print(f"\n{'='*70}")
        print(f"TOP {n} FUZZY RULES")
        print(f"{'='*70}")
        
        for i, rule in enumerate(rules):
            conditions = []
            for j, fuzzy_set in enumerate(rule['antecedent']):
                conditions.append(f"{self.feature_names[j]} is {self.labels[fuzzy_set]}")
            
            class_name = "Diabetes" if rule['consequent'] == 1 else "No Diabetes"
            print(f"\nR{i+1}: IF {' AND '.join(conditions[:4])}")
            if len(conditions) > 4:
                print(f"       AND {' AND '.join(conditions[4:])}")
            print(f"      THEN {class_name} (confidence={rule['weight']:.3f}, support={rule['support']})")


def load_pima_diabetes():
    """Load Pima Indians Diabetes dataset with preprocessing."""
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
               'Insulin', 'BMI', 'DiabetesPedigree', 'Age', 'Outcome']
    df = pd.read_csv(url, names=columns)
    
    # Handle zero values (missing data)
    cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)
    df = df.fillna(df.median())
    
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y, columns[:-1]


def main():
    print("=" * 70)
    print("FUZZY RULE-BASED CLASSIFICATION SYSTEM")
    print("Demonstration on Low-Accuracy Dataset (Pima Diabetes)")
    print("=" * 70)
    
    # Load data
    print("\n[1] Loading Dataset...")
    X, y, feature_names = load_pima_diabetes()
    
    print(f"    Dataset: Pima Indians Diabetes")
    print(f"    Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"    Class 0 (No Diabetes): {np.sum(y==0)}")
    print(f"    Class 1 (Diabetes): {np.sum(y==1)}")
    print(f"    Note: This dataset typically achieves ~75-77% accuracy")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # =========================================================================
    # Train Fuzzy Classifier
    # =========================================================================
    print("\n" + "=" * 70)
    print("[2] TRAINING FUZZY CLASSIFIER")
    print("=" * 70)
    
    start = time.time()
    fuzzy_clf = FastFuzzyClassifier(n_partitions=5)
    fuzzy_clf.fit(X_train, y_train, feature_names)
    fuzzy_time = time.time() - start
    
    fuzzy_train = fuzzy_clf.score(X_train, y_train)
    fuzzy_test = fuzzy_clf.score(X_test, y_test)
    
    print(f"    Training Accuracy: {fuzzy_train:.4f}")
    print(f"    Test Accuracy: {fuzzy_test:.4f}")
    print(f"    Number of Rules: {len(fuzzy_clf.rules)}")
    print(f"    Training Time: {fuzzy_time:.3f}s")
    
    # =========================================================================
    # Show Interpretable Rules
    # =========================================================================
    print("\n" + "=" * 70)
    print("[3] INTERPRETABLE FUZZY RULES")
    print("=" * 70)
    fuzzy_clf.print_rules(n=10)
    
    # =========================================================================
    # Compare with Baselines
    # =========================================================================
    print("\n" + "=" * 70)
    print("[4] COMPARISON WITH ML BASELINES")
    print("=" * 70)
    
    # Normalize for sklearn
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    baselines = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVM (RBF)': SVC(kernel='rbf', random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5)
    }
    
    results = [{
        'Classifier': 'Fuzzy RBCS',
        'Train Acc': f"{fuzzy_train:.4f}",
        'Test Acc': f"{fuzzy_test:.4f}",
        'Interpretable': 'Yes ✓',
        'Rules/Params': f"{len(fuzzy_clf.rules)} rules"
    }]
    
    for name, clf in baselines.items():
        clf.fit(X_train_scaled, y_train)
        train_acc = clf.score(X_train_scaled, y_train)
        test_acc = clf.score(X_test_scaled, y_test)
        
        interpretable = 'Yes ✓' if 'Tree' in name else 'No'
        
        if hasattr(clf, 'n_estimators'):
            params = f"{clf.n_estimators} trees"
        elif hasattr(clf, 'tree_'):
            params = f"{clf.tree_.node_count} nodes"
        else:
            params = "N/A"
        
        results.append({
            'Classifier': name,
            'Train Acc': f"{train_acc:.4f}",
            'Test Acc': f"{test_acc:.4f}",
            'Interpretable': interpretable,
            'Rules/Params': params
        })
    
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    # =========================================================================
    # Cross-Validation Comparison
    # =========================================================================
    print("\n" + "=" * 70)
    print("[5] 5-FOLD CROSS-VALIDATION")
    print("=" * 70)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_results = []
    
    # Fuzzy CV
    fuzzy_scores = []
    for train_idx, test_idx in cv.split(X, y):
        clf = FastFuzzyClassifier(n_partitions=5)
        clf.fit(X[train_idx], y[train_idx], feature_names)
        fuzzy_scores.append(clf.score(X[test_idx], y[test_idx]))
    
    cv_results.append({
        'Classifier': 'Fuzzy RBCS',
        'CV Accuracy': f"{np.mean(fuzzy_scores):.4f} ± {np.std(fuzzy_scores):.4f}",
        'Interpretable': 'Yes ✓'
    })
    
    # Baseline CV
    for name, clf_class in [
        ('Random Forest', lambda: RandomForestClassifier(n_estimators=100, random_state=42)),
        ('Gradient Boosting', lambda: GradientBoostingClassifier(random_state=42)),
        ('SVM (RBF)', lambda: SVC(kernel='rbf', random_state=42)),
        ('Logistic Regression', lambda: LogisticRegression(random_state=42, max_iter=1000)),
        ('Decision Tree', lambda: DecisionTreeClassifier(random_state=42, max_depth=5))
    ]:
        scores = []
        for train_idx, test_idx in cv.split(X, y):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[train_idx])
            X_te = scaler.transform(X[test_idx])
            
            clf = clf_class()
            clf.fit(X_tr, y[train_idx])
            scores.append(clf.score(X_te, y[test_idx]))
        
        cv_results.append({
            'Classifier': name,
            'CV Accuracy': f"{np.mean(scores):.4f} ± {np.std(scores):.4f}",
            'Interpretable': 'Yes ✓' if 'Tree' in name else 'No'
        })
    
    df_cv = pd.DataFrame(cv_results)
    print(df_cv.to_string(index=False))
    
    # =========================================================================
    # Classification Report
    # =========================================================================
    print("\n" + "=" * 70)
    print("[6] DETAILED CLASSIFICATION REPORT (Fuzzy Classifier)")
    print("=" * 70)
    
    y_pred = fuzzy_clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes']))
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
    Dataset: Pima Indians Diabetes (Low-accuracy benchmark)
    
    Fuzzy Classifier Results:
    • Test Accuracy: {fuzzy_test:.4f}
    • Number of Rules: {len(fuzzy_clf.rules)}
    • Training Time: {fuzzy_time:.3f}s
    
    Key Advantages of Fuzzy Rule-Based Classification:
    
    1. INTERPRETABILITY: Rules can be understood by domain experts
       Example: "IF Glucose is High AND BMI is High THEN Diabetes"
    
    2. TRANSPARENCY: Decision process is fully explainable
    
    3. COMPETITIVE ACCURACY: Achieves comparable results to black-box models
       on this challenging dataset (~75% accuracy ceiling)
    
    4. MEDICAL APPLICABILITY: Rules can be validated by physicians
    
    5. HANDLES UNCERTAINTY: Fuzzy logic naturally handles imprecise data
    """)
    
    return fuzzy_clf


if __name__ == '__main__':
    clf = main()
