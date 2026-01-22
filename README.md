# Fuzzy Rule-Based Classification System

A research project implementing **Fuzzy Rule-Based Classification** with automatic rule generation and genetic algorithm optimization for interpretable medical diagnosis.

## 🎯 Project Overview

This project explores the intersection of fuzzy logic and machine learning, focusing on:

- **Automatic Fuzzy Rule Generation** from data
- **Genetic Algorithms** for optimizing fuzzy rule bases
- **Interpretable Fuzzy Classifiers** for medical diagnosis

## 🔬 Research Objectives

1. Develop automatic fuzzy rule generation methods (Wang-Mendel, Clustering-based)
2. Implement genetic algorithm optimization for rule bases and membership functions
3. Create interpretable diagnostic systems for medical datasets
4. Compare fuzzy classifiers with traditional ML methods

## 📁 Project Structure

```
fuzzy/
├── README.md
├── requirements.txt
├── src/
│   ├── fuzzy_classifier.py      # Main FRBCS implementation
│   ├── genetic_optimizer.py     # GA optimization module
│   ├── rule_generation.py       # Rule generation methods
│   └── membership_functions.py  # MF utilities
├── experiments/
│   ├── experiment_framework.py  # Experimental setup
│   └── run_experiments.py       # Main experiment runner
├── notebooks/
│   └── analysis.ipynb           # Result analysis
└── data/
    └── datasets/                # Medical datasets
```

## 🛠️ Installation

```bash
git clone https://github.com/9501893704rahul/fuzzy.git
cd fuzzy
pip install -r requirements.txt
```

## 📊 Datasets

- Pima Indians Diabetes
- Cleveland Heart Disease
- Wisconsin Breast Cancer
- Hepatitis
- Thyroid Disease

## 🚀 Usage

```python
from src.fuzzy_classifier import FuzzyRuleClassifier

# Initialize classifier
clf = FuzzyRuleClassifier(n_partitions=5, optimize=True)

# Train
clf.fit(X_train, y_train, feature_names)

# Predict
predictions = clf.predict(X_test)

# View interpretable rules
clf.print_rules(n=10)
```

## 📈 Experiments

1. **Rule Generation Comparison** - Compare Wang-Mendel, Clustering, DT-Fuzzy methods
2. **GA Optimization Impact** - Measure improvement from genetic optimization
3. **Multi-Objective Optimization** - Accuracy vs interpretability trade-off
4. **Baseline Comparison** - Compare with Random Forest, SVM, Neural Networks
5. **Sensitivity Analysis** - Parameter impact study
6. **Robustness Testing** - Noise and missing data handling

## 📚 Key Libraries

- `scikit-fuzzy` - Fuzzy logic toolkit
- `DEAP` - Evolutionary algorithms
- `scikit-learn` - ML utilities
- `numpy`, `pandas` - Data manipulation
- `matplotlib`, `seaborn` - Visualization

## 📝 References

- Ishibuchi, H. - "Fuzzy rule-based classification systems"
- Cordon, O. - "Genetic fuzzy systems: evolutionary tuning"
- Alcala-Fdez, J. - "KEEL: A software tool for data mining"

## 📄 License

MIT License

## 👤 Author

Rahul

---

*This project is part of research on interpretable machine learning for medical diagnosis.*
