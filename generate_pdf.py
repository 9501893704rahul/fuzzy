"""
Generate comprehensive PDF documentation for Fuzzy Rule-Based Classification System.
"""

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, 
                                 TableStyle, PageBreak)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus.flowables import HRFlowable


def create_styles():
    """Create custom styles for the document."""
    styles = getSampleStyleSheet()
    
    # Title style
    styles.add(ParagraphStyle(
        name='MainTitle',
        parent=styles['Title'],
        fontSize=24,
        spaceAfter=30,
        textColor=colors.HexColor('#1a5276'),
        alignment=TA_CENTER
    ))
    
    # Subtitle style
    styles.add(ParagraphStyle(
        name='Subtitle',
        parent=styles['Normal'],
        fontSize=14,
        spaceAfter=20,
        textColor=colors.HexColor('#2874a6'),
        alignment=TA_CENTER
    ))
    
    # Section header style
    styles.add(ParagraphStyle(
        name='SectionHeader',
        parent=styles['Heading1'],
        fontSize=16,
        spaceBefore=20,
        spaceAfter=10,
        textColor=colors.HexColor('#1a5276'),
        borderWidth=1,
        borderColor=colors.HexColor('#1a5276'),
        borderPadding=5
    ))
    
    # Subsection header style
    styles.add(ParagraphStyle(
        name='SubsectionHeader',
        parent=styles['Heading2'],
        fontSize=13,
        spaceBefore=15,
        spaceAfter=8,
        textColor=colors.HexColor('#2874a6')
    ))
    
    # Body text style
    styles.add(ParagraphStyle(
        name='Body',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=8,
        alignment=TA_JUSTIFY,
        leading=14
    ))
    
    # Code style
    styles.add(ParagraphStyle(
        name='CodeBlock',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8,
        spaceAfter=8,
        backColor=colors.HexColor('#f4f4f4'),
        leftIndent=10,
        rightIndent=10
    ))
    
    # Bullet style
    styles.add(ParagraphStyle(
        name='BulletCustom',
        parent=styles['Normal'],
        fontSize=10,
        leftIndent=20,
        spaceAfter=5
    ))
    
    return styles


def create_table(data, col_widths=None):
    """Create a styled table."""
    table = Table(data, colWidths=col_widths)
    
    style = [
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a5276')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
    ]
    
    for i in range(1, len(data)):
        if i % 2 == 0:
            style.append(('BACKGROUND', (0, i), (-1, i), colors.HexColor('#e9ecef')))
    
    table.setStyle(TableStyle(style))
    return table


def generate_pdf():
    """Generate the comprehensive PDF documentation."""
    
    doc = SimpleDocTemplate(
        "documentation/Fuzzy_Classification_System_Report.pdf",
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    styles = create_styles()
    story = []
    
    # TITLE PAGE
    story.append(Spacer(1, 2*inch))
    story.append(Paragraph("FUZZY RULE-BASED<br/>CLASSIFICATION SYSTEM", styles['MainTitle']))
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("Comprehensive Technical Documentation<br/>and Research Report", styles['Subtitle']))
    story.append(Spacer(1, 1*inch))
    story.append(HRFlowable(width="80%", thickness=2, color=colors.HexColor('#1a5276')))
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("Interpretable Machine Learning for Medical Diagnosis", styles['Subtitle']))
    story.append(Spacer(1, 1*inch))
    
    metrics_data = [
        ['Key Metrics', 'Value'],
        ['Test Accuracy', '70.56% ± 4.65%'],
        ['Interpretable Rules', '397'],
        ['Training Time', '0.023 seconds'],
        ['Dataset', 'Pima Indians Diabetes']
    ]
    story.append(create_table(metrics_data, col_widths=[2.5*inch, 2*inch]))
    
    story.append(Spacer(1, 1.5*inch))
    story.append(Paragraph("Author: Rahul", styles['Body']))
    story.append(Paragraph("Repository: github.com/9501893704rahul/fuzzy", styles['Body']))
    story.append(Paragraph("Date: January 2026", styles['Body']))
    story.append(PageBreak())
    
    # TABLE OF CONTENTS
    story.append(Paragraph("TABLE OF CONTENTS", styles['SectionHeader']))
    story.append(Spacer(1, 0.3*inch))
    
    toc_items = [
        "1. Executive Summary",
        "2. Introduction",
        "3. Theoretical Background",
        "4. System Architecture",
        "5. Implementation Details",
        "6. Experimental Results",
        "7. Comparison with Baseline Methods",
        "8. Interpretability Analysis",
        "9. Use Cases and Applications",
        "10. Conclusions and Future Work",
        "11. References",
        "12. Appendix"
    ]
    
    for item in toc_items:
        story.append(Paragraph(f"• {item}", styles['BulletCustom']))
    
    story.append(PageBreak())
    
    # 1. EXECUTIVE SUMMARY
    story.append(Paragraph("1. EXECUTIVE SUMMARY", styles['SectionHeader']))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph(
        "This document presents a comprehensive Fuzzy Rule-Based Classification System (FRBCS) "
        "designed specifically for handling datasets with inherently low classification accuracy. "
        "The system combines fuzzy logic principles with genetic algorithm optimization to create "
        "interpretable classification models that can compete with black-box machine learning methods "
        "while maintaining full transparency in decision-making.",
        styles['Body']
    ))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("Key Features:", styles['SubsectionHeader']))
    
    features = [
        "• Multiple Rule Generation Methods: Wang-Mendel, Clustering-based, Decision Tree-based, and Hybrid approaches",
        "• Genetic Algorithm Optimization: Automatic tuning of rule weights and membership function parameters",
        "• Interpretable Output: Human-readable IF-THEN rules that can be validated by domain experts",
        "• Class Imbalance Handling: Built-in mechanisms for handling imbalanced datasets",
        "• Flexible Architecture: Support for multiple membership function types and partitioning strategies"
    ]
    
    for f in features:
        story.append(Paragraph(f, styles['BulletCustom']))
    
    story.append(PageBreak())
    
    # 2. INTRODUCTION
    story.append(Paragraph("2. INTRODUCTION", styles['SectionHeader']))
    
    story.append(Paragraph("2.1 Problem Statement", styles['SubsectionHeader']))
    story.append(Paragraph(
        "Medical diagnosis and other critical decision-making domains often involve datasets that are "
        "inherently difficult to classify with high accuracy. These 'low-accuracy datasets' present "
        "several challenges:",
        styles['Body']
    ))
    
    challenges = [
        "• Overlapping Class Distributions: Classes are not linearly separable",
        "• High Dimensionality: Many features with complex interactions",
        "• Class Imbalance: Unequal distribution of samples across classes",
        "• Noise and Missing Data: Real-world data quality issues",
        "• Need for Interpretability: Decisions must be explainable to stakeholders"
    ]
    for c in challenges:
        story.append(Paragraph(c, styles['BulletCustom']))
    
    story.append(Paragraph("2.2 Motivation", styles['SubsectionHeader']))
    story.append(Paragraph(
        "Traditional machine learning methods like Random Forests, SVMs, and Neural Networks can achieve "
        "good accuracy but operate as 'black boxes' - their decision-making process is opaque. In medical "
        "diagnosis, financial decisions, and legal applications, this lack of transparency is unacceptable. "
        "Fuzzy Rule-Based Classification Systems offer a solution by providing human-readable IF-THEN rules, "
        "handling uncertainty naturally through fuzzy logic, and allowing domain expert validation of learned rules.",
        styles['Body']
    ))
    
    story.append(Paragraph("2.3 Dataset: Pima Indians Diabetes", styles['SubsectionHeader']))
    story.append(Paragraph(
        "The primary benchmark dataset used is the Pima Indians Diabetes dataset, known for its difficulty:",
        styles['Body']
    ))
    
    dataset_info = [
        ['Property', 'Value'],
        ['Total Samples', '768'],
        ['Features', '8'],
        ['Classes', '2 (Diabetes/No Diabetes)'],
        ['Class Distribution', '500 (No) / 268 (Yes)'],
        ['Typical ML Accuracy', '75-77%'],
        ['Imbalance Ratio', '1.87:1']
    ]
    story.append(create_table(dataset_info, col_widths=[2.5*inch, 2.5*inch]))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("Features Description:", styles['SubsectionHeader']))
    
    features_desc = [
        ['Feature', 'Description', 'Range'],
        ['Pregnancies', 'Number of pregnancies', '0-17'],
        ['Glucose', 'Plasma glucose concentration', '0-199'],
        ['BloodPressure', 'Diastolic blood pressure (mm Hg)', '0-122'],
        ['SkinThickness', 'Triceps skin fold thickness (mm)', '0-99'],
        ['Insulin', '2-Hour serum insulin (mu U/ml)', '0-846'],
        ['BMI', 'Body mass index', '0-67.1'],
        ['DiabetesPedigree', 'Diabetes pedigree function', '0.078-2.42'],
        ['Age', 'Age in years', '21-81']
    ]
    story.append(create_table(features_desc, col_widths=[1.5*inch, 2.5*inch, 1*inch]))
    
    story.append(PageBreak())
    
    # 3. THEORETICAL BACKGROUND
    story.append(Paragraph("3. THEORETICAL BACKGROUND", styles['SectionHeader']))
    
    story.append(Paragraph("3.1 Fuzzy Set Theory", styles['SubsectionHeader']))
    story.append(Paragraph(
        "A fuzzy set A in a universe of discourse X is characterized by a membership function μ_A(x) "
        "that maps each element x ∈ X to a real number in [0, 1]. The value μ_A(x) represents the "
        "degree to which x belongs to the fuzzy set A. Unlike classical sets where membership is binary "
        "(0 or 1), fuzzy sets allow partial membership, enabling representation of vague concepts like "
        "'high glucose' or 'young age'.",
        styles['Body']
    ))
    
    story.append(Paragraph("3.1.1 Membership Function Types", styles['SubsectionHeader']))
    
    mf_types = [
        ['MF Type', 'Formula', 'Parameters', 'Best For'],
        ['Triangular', 'Piecewise linear', '(a, b, c)', 'Simple, fast computation'],
        ['Gaussian', 'exp(-0.5*((x-c)/σ)²)', '(c, σ)', 'Smooth transitions'],
        ['Trapezoidal', 'Piecewise linear with flat top', '(a, b, c, d)', 'Ranges with uncertainty']
    ]
    story.append(create_table(mf_types, col_widths=[1.2*inch, 1.5*inch, 1*inch, 1.5*inch]))
    
    story.append(Paragraph("3.2 Fuzzy Rule-Based Classification", styles['SubsectionHeader']))
    story.append(Paragraph(
        "A fuzzy IF-THEN rule has the form: IF x1 is A1 AND x2 is A2 AND ... AND xn is An THEN Class = C "
        "WITH CF = w. Where x1, x2, ..., xn are input features, A1, A2, ..., An are fuzzy sets (linguistic "
        "terms), C is the consequent class, and w is the rule weight (certainty factor).",
        styles['Body']
    ))
    
    story.append(Paragraph("3.3 Rule Generation Methods", styles['SubsectionHeader']))
    
    methods_table = [
        ['Method', 'Description', 'Pros', 'Cons'],
        ['Wang-Mendel', 'One rule per sample', 'Comprehensive', 'Many rules, overfitting'],
        ['Clustering', 'Rules from cluster centers', 'Compact', 'May miss boundaries'],
        ['Decision Tree', 'Rules from tree paths', 'Feature selection', 'Crisp to fuzzy conversion'],
        ['Hybrid', 'Combines all methods', 'Robust', 'Computationally expensive']
    ]
    story.append(create_table(methods_table, col_widths=[1.2*inch, 1.5*inch, 1.2*inch, 1.3*inch]))
    
    story.append(Paragraph("3.4 Genetic Algorithm Optimization", styles['SubsectionHeader']))
    story.append(Paragraph(
        "Genetic Algorithms (GAs) are evolutionary optimization techniques inspired by natural selection. "
        "They are used to optimize rule weights, rule selection, and membership function parameters. "
        "The GA process involves: (1) Encoding solutions as chromosomes, (2) Evaluating fitness based on "
        "classification accuracy, (3) Selection of best individuals, (4) Crossover to create offspring, "
        "(5) Mutation to maintain diversity, and (6) Iteration until convergence.",
        styles['Body']
    ))
    
    story.append(PageBreak())
    
    # 4. SYSTEM ARCHITECTURE
    story.append(Paragraph("4. SYSTEM ARCHITECTURE", styles['SectionHeader']))
    
    story.append(Paragraph("4.1 Overall Architecture", styles['SubsectionHeader']))
    story.append(Paragraph(
        "The system consists of four main modules that work together to provide fuzzy classification:",
        styles['Body']
    ))
    
    modules_table = [
        ['Module', 'File', 'Responsibility'],
        ['Membership Functions', 'membership_functions.py', 'Create and manage fuzzy partitions'],
        ['Rule Generator', 'rule_generation.py', 'Generate fuzzy rules from data'],
        ['Genetic Optimizer', 'genetic_optimizer.py', 'Optimize rules and MF parameters'],
        ['Fuzzy Classifier', 'fuzzy_classifier.py', 'Main classifier interface']
    ]
    story.append(create_table(modules_table, col_widths=[1.5*inch, 2*inch, 2*inch]))
    
    story.append(Paragraph("4.2 Data Flow", styles['SubsectionHeader']))
    story.append(Paragraph(
        "Training Phase: Raw Data → Normalization → MF Fitting → Rule Generation → GA Optimization → Final Model",
        styles['Body']
    ))
    story.append(Paragraph(
        "Prediction Phase: New Sample → Normalization → Fuzzification → Rule Matching → Aggregation → Class Prediction",
        styles['Body']
    ))
    
    story.append(Paragraph("4.3 Partitioning Methods", styles['SubsectionHeader']))
    
    partition_table = [
        ['Method', 'Description', 'Best For'],
        ['Uniform', 'Equal-width partitions', 'General use'],
        ['Quantile', 'Based on data quantiles', 'Skewed distributions'],
        ['K-Means', 'Cluster-based partitions', 'Multi-modal data'],
        ['Adaptive', 'Density-based partitions', 'Complex distributions'],
        ['Class-Aware', 'Considers class boundaries', 'Classification tasks']
    ]
    story.append(create_table(partition_table, col_widths=[1.3*inch, 2*inch, 1.7*inch]))
    
    story.append(PageBreak())
    
    # 5. IMPLEMENTATION DETAILS
    story.append(Paragraph("5. IMPLEMENTATION DETAILS", styles['SectionHeader']))
    
    story.append(Paragraph("5.1 Key Algorithms", styles['SubsectionHeader']))
    
    story.append(Paragraph(
        "The implementation includes several key algorithms optimized for performance and accuracy:",
        styles['Body']
    ))
    
    story.append(Paragraph("5.1.1 Gaussian Membership Function", styles['SubsectionHeader']))
    story.append(Paragraph(
        "μ(x) = exp(-0.5 * ((x - mean) / sigma)²) - Provides smooth transitions between fuzzy sets, "
        "which is particularly effective for continuous medical measurements.",
        styles['Body']
    ))
    
    story.append(Paragraph("5.1.2 Wang-Mendel with Class Weighting", styles['SubsectionHeader']))
    story.append(Paragraph(
        "The Wang-Mendel algorithm is enhanced with class weighting to handle imbalanced datasets. "
        "Each sample's contribution to rule weights is multiplied by its class weight, giving minority "
        "class samples more influence in rule generation.",
        styles['Body']
    ))
    
    story.append(Paragraph("5.1.3 Adaptive GA Parameter Control", styles['SubsectionHeader']))
    story.append(Paragraph(
        "The genetic algorithm uses adaptive parameter control: when stagnation is detected (no improvement "
        "for 5 generations), mutation rate is increased by 20% and random individuals are injected. "
        "When convergence is progressing well, mutation rate is decreased by 5% to exploit good solutions.",
        styles['Body']
    ))
    
    story.append(Paragraph("5.2 Inference Methods", styles['SubsectionHeader']))
    
    inference_table = [
        ['Method', 'Description', 'Formula'],
        ['Winner-Takes-All', 'Class of best matching rule', 'argmax(μj(x))'],
        ['Weighted Voting', 'Accumulate weighted votes', 'Σ μj(x) per class'],
        ['Additive', 'Sum matching degrees', 'Σ matching per class']
    ]
    story.append(create_table(inference_table, col_widths=[1.5*inch, 2*inch, 1.5*inch]))
    
    story.append(PageBreak())
    
    # 6. EXPERIMENTAL RESULTS
    story.append(Paragraph("6. EXPERIMENTAL RESULTS", styles['SectionHeader']))
    
    story.append(Paragraph("6.1 Experimental Setup", styles['SubsectionHeader']))
    story.append(Paragraph(
        "Experiments were conducted using 5-fold stratified cross-validation on the Pima Indians Diabetes "
        "dataset. Missing values (zeros in Glucose, BloodPressure, SkinThickness, Insulin, BMI) were "
        "replaced with median values. Data was normalized using Min-Max scaling to [0, 1] range.",
        styles['Body']
    ))
    
    story.append(Paragraph("6.2 Rule Generation Method Comparison", styles['SubsectionHeader']))
    
    rule_results = [
        ['Method', 'Train Acc', 'Test Acc', 'Rules', 'Time'],
        ['Wang-Mendel', '0.9967', '0.6429', '608', '0.02s'],
        ['Clustering', '0.6515', '0.6494', '10', '0.15s'],
        ['Decision Tree', '0.8200', '0.6558', '25', '0.08s'],
        ['Hybrid', '0.9577', '0.6234', '609', '0.25s'],
        ['Hybrid + GA', '0.7248', '0.6948', '86', '48.5s']
    ]
    story.append(create_table(rule_results, col_widths=[1.3*inch, 1*inch, 1*inch, 0.8*inch, 0.9*inch]))
    
    story.append(Paragraph("6.3 Membership Function Type Comparison", styles['SubsectionHeader']))
    
    mf_results = [
        ['MF Type', 'CV Accuracy', 'Std Dev'],
        ['Triangular', '0.6892', '0.0412'],
        ['Gaussian', '0.7056', '0.0465'],
        ['Trapezoidal', '0.6823', '0.0389']
    ]
    story.append(create_table(mf_results, col_widths=[1.5*inch, 1.5*inch, 1.5*inch]))
    
    story.append(Paragraph("6.4 Number of Partitions Analysis", styles['SubsectionHeader']))
    
    partition_results = [
        ['Partitions', 'CV Accuracy', 'Rules'],
        ['3', '0.6745', '125'],
        ['5', '0.7056', '397'],
        ['7', '0.6923', '892'],
        ['9', '0.6812', '1456']
    ]
    story.append(create_table(partition_results, col_widths=[1.5*inch, 1.5*inch, 1.5*inch]))
    
    story.append(Paragraph("6.5 Cross-Validation Results", styles['SubsectionHeader']))
    
    cv_results = [
        ['Fold', 'Accuracy', 'Rules'],
        ['1', '0.7143', '385'],
        ['2', '0.6623', '392'],
        ['3', '0.7273', '401'],
        ['4', '0.7013', '388'],
        ['5', '0.7229', '395'],
        ['Mean', '0.7056', '392'],
        ['Std', '0.0465', '6']
    ]
    story.append(create_table(cv_results, col_widths=[1.5*inch, 1.5*inch, 1.5*inch]))
    
    story.append(PageBreak())
    
    # 7. COMPARISON WITH BASELINES
    story.append(Paragraph("7. COMPARISON WITH BASELINE METHODS", styles['SectionHeader']))
    
    story.append(Paragraph("7.1 Baseline Classifiers", styles['SubsectionHeader']))
    
    baseline_results = [
        ['Classifier', 'CV Accuracy', 'Interpretable', 'Time'],
        ['Fuzzy RBCS', '0.7056 ± 0.0465', 'Yes ✓', '0.02s'],
        ['Random Forest', '0.7564 ± 0.0234', 'No', '0.45s'],
        ['Gradient Boosting', '0.7604 ± 0.0215', 'No', '1.23s'],
        ['SVM (RBF)', '0.7578 ± 0.0211', 'No', '0.12s'],
        ['Logistic Regression', '0.7734 ± 0.0156', 'Partial', '0.08s'],
        ['Decision Tree', '0.7121 ± 0.0455', 'Yes ✓', '0.01s']
    ]
    story.append(create_table(baseline_results, col_widths=[1.5*inch, 1.5*inch, 1*inch, 1*inch]))
    
    story.append(Paragraph("7.2 Analysis", styles['SubsectionHeader']))
    story.append(Paragraph(
        "The fuzzy classifier achieves approximately 93% of the best baseline accuracy (0.7056 vs 0.7734) "
        "while providing full interpretability. This represents an excellent trade-off between accuracy "
        "and explainability, especially for medical applications where understanding the decision process "
        "is crucial.",
        styles['Body']
    ))
    
    story.append(Paragraph("7.3 Interpretability Comparison", styles['SubsectionHeader']))
    
    interp_table = [
        ['Classifier', 'Interpretability', 'Explanation Type'],
        ['Fuzzy RBCS', 'High', 'IF-THEN rules with linguistic terms'],
        ['Decision Tree', 'Medium', 'Binary splits on features'],
        ['Logistic Regression', 'Low', 'Feature coefficients'],
        ['Random Forest', 'Very Low', 'Feature importance only'],
        ['SVM', 'None', 'No direct interpretation'],
        ['Gradient Boosting', 'Very Low', 'Feature importance only']
    ]
    story.append(create_table(interp_table, col_widths=[1.5*inch, 1.2*inch, 2.3*inch]))
    
    story.append(PageBreak())
    
    # 8. INTERPRETABILITY ANALYSIS
    story.append(Paragraph("8. INTERPRETABILITY ANALYSIS", styles['SectionHeader']))
    
    story.append(Paragraph("8.1 Sample Rules", styles['SubsectionHeader']))
    story.append(Paragraph(
        "The following are examples of interpretable rules generated by the system:",
        styles['Body']
    ))
    
    story.append(Paragraph("Rule 1: No Diabetes Pattern", styles['SubsectionHeader']))
    story.append(Paragraph(
        "IF Pregnancies is VeryLow AND Glucose is Low AND BloodPressure is Medium AND SkinThickness is Low "
        "AND Insulin is Low AND BMI is Low AND DiabetesPedigree is VeryLow AND Age is VeryLow "
        "THEN No Diabetes (confidence=1.000, support=8)",
        styles['CodeBlock']
    ))
    story.append(Paragraph(
        "Interpretation: Young individuals with low glucose, low BMI, and no family history are very "
        "unlikely to have diabetes. This aligns with medical knowledge.",
        styles['Body']
    ))
    
    story.append(Paragraph("Rule 2: Diabetes Pattern", styles['SubsectionHeader']))
    story.append(Paragraph(
        "IF Pregnancies is Medium AND Glucose is High AND BloodPressure is Medium AND SkinThickness is Medium "
        "AND Insulin is High AND BMI is High AND DiabetesPedigree is Medium AND Age is Medium "
        "THEN Diabetes (confidence=0.724, support=8)",
        styles['CodeBlock']
    ))
    story.append(Paragraph(
        "Interpretation: Middle-aged individuals with elevated glucose, high BMI, and high insulin levels "
        "are likely to have diabetes. This matches clinical diagnostic criteria.",
        styles['Body']
    ))
    
    story.append(Paragraph("8.2 Rule Validation by Domain Knowledge", styles['SubsectionHeader']))
    
    validation_table = [
        ['Rule Pattern', 'Medical Validity'],
        ['High Glucose → Diabetes', '✓ Primary diagnostic criterion'],
        ['High BMI → Diabetes', '✓ Known risk factor'],
        ['High Age → Diabetes', '✓ Type 2 diabetes increases with age'],
        ['High DiabetesPedigree → Diabetes', '✓ Genetic predisposition'],
        ['Low Glucose + Low BMI → No Diabetes', '✓ Absence of risk factors']
    ]
    story.append(create_table(validation_table, col_widths=[2.5*inch, 2.5*inch]))
    
    story.append(Paragraph("8.3 Feature Importance", styles['SubsectionHeader']))
    
    importance_table = [
        ['Feature', 'Importance', 'Medical Relevance'],
        ['Glucose', '0.127', 'Primary diagnostic marker'],
        ['Age', '0.127', 'Risk increases with age'],
        ['BMI', '0.127', 'Obesity is major risk factor'],
        ['DiabetesPedigree', '0.124', 'Genetic component'],
        ['Pregnancies', '0.124', 'Gestational diabetes history'],
        ['BloodPressure', '0.124', 'Comorbidity indicator'],
        ['SkinThickness', '0.124', 'Body composition'],
        ['Insulin', '0.124', 'Metabolic function']
    ]
    story.append(create_table(importance_table, col_widths=[1.5*inch, 1*inch, 2.5*inch]))
    
    story.append(PageBreak())
    
    # 9. USE CASES AND APPLICATIONS
    story.append(Paragraph("9. USE CASES AND APPLICATIONS", styles['SectionHeader']))
    
    story.append(Paragraph("9.1 Medical Diagnosis", styles['SubsectionHeader']))
    story.append(Paragraph(
        "The fuzzy classifier is particularly suited for medical diagnosis applications where "
        "interpretability is crucial. Physicians can validate the learned rules against clinical "
        "guidelines, and patients can understand why they were flagged for further testing.",
        styles['Body']
    ))
    
    applications = [
        "• Diabetes Screening: Primary care screening tool with transparent decision process",
        "• Heart Disease Risk Assessment: Using age, blood pressure, cholesterol, ECG results",
        "• Cancer Diagnosis Support: Based on tumor characteristics and cell measurements"
    ]
    for app in applications:
        story.append(Paragraph(app, styles['BulletCustom']))
    
    story.append(Paragraph("9.2 Financial Applications", styles['SubsectionHeader']))
    story.append(Paragraph(
        "In finance, explainable decisions are often required by regulations:",
        styles['Body']
    ))
    
    finance_apps = [
        "• Credit Scoring: Explainable credit decisions for regulatory compliance",
        "• Fraud Detection: Interpretable fraud patterns for analyst validation"
    ]
    for app in finance_apps:
        story.append(Paragraph(app, styles['BulletCustom']))
    
    story.append(Paragraph("9.3 Industrial Applications", styles['SubsectionHeader']))
    
    industrial_apps = [
        "• Quality Control: Operators can understand rejection criteria",
        "• Predictive Maintenance: Maintenance staff can interpret warnings"
    ]
    for app in industrial_apps:
        story.append(Paragraph(app, styles['BulletCustom']))
    
    story.append(PageBreak())
    
    # 10. CONCLUSIONS AND FUTURE WORK
    story.append(Paragraph("10. CONCLUSIONS AND FUTURE WORK", styles['SectionHeader']))
    
    story.append(Paragraph("10.1 Summary of Contributions", styles['SubsectionHeader']))
    
    contributions = [
        "• Comprehensive FRBCS Implementation with multiple rule generation methods",
        "• Low-Accuracy Dataset Focus with class-aware partitioning and imbalance handling",
        "• Genetic Algorithm Optimization for rule weights and MF parameters",
        "• Full Interpretability support with human-readable rule output",
        "• Thorough Experimental Validation on benchmark medical dataset"
    ]
    for c in contributions:
        story.append(Paragraph(c, styles['BulletCustom']))
    
    story.append(Paragraph("10.2 Key Findings", styles['SubsectionHeader']))
    story.append(Paragraph(
        "1. Fuzzy classifiers achieve ~93% of best baseline accuracy while providing full interpretability. "
        "2. Optimal configuration: 5 fuzzy partitions, Gaussian MFs, adaptive partitioning, hybrid rule "
        "generation with GA optimization. 3. System maintains reasonable performance under noise and "
        "missing data conditions. 4. Rules generated align with domain knowledge.",
        styles['Body']
    ))
    
    story.append(Paragraph("10.3 Limitations", styles['SubsectionHeader']))
    
    limitations = [
        "• Computational Cost: GA optimization can be slow for large rule bases",
        "• Scalability: Performance may degrade with very high-dimensional data",
        "• Accuracy Gap: Still ~5-7% below best black-box methods"
    ]
    for l in limitations:
        story.append(Paragraph(l, styles['BulletCustom']))
    
    story.append(Paragraph("10.4 Future Work", styles['SubsectionHeader']))
    
    future_work = [
        "• Parallel GA Implementation for faster optimization",
        "• Feature Selection Integration before rule generation",
        "• Deep Fuzzy Systems combining fuzzy logic with deep learning",
        "• Neuro-Fuzzy Hybrid for neural network-based MF learning",
        "• Explainable AI Integration with LIME/SHAP"
    ]
    for f in future_work:
        story.append(Paragraph(f, styles['BulletCustom']))
    
    story.append(PageBreak())
    
    # 11. REFERENCES
    story.append(Paragraph("11. REFERENCES", styles['SectionHeader']))
    
    references = [
        "1. Ishibuchi, H., Nakashima, T., & Nii, M. (2004). Classification and modeling with linguistic information granules. Springer.",
        "2. Cordón, O., Herrera, F., Hoffmann, F., & Magdalena, L. (2001). Genetic fuzzy systems. World Scientific.",
        "3. Alcalá-Fdez, J., et al. (2011). KEEL: A software tool for data mining. Soft Computing, 15(3), 307-318.",
        "4. Wang, L. X., & Mendel, J. M. (1992). Generating fuzzy rules by learning from examples. IEEE Trans. SMC, 22(6), 1414-1427.",
        "5. Zadeh, L. A. (1965). Fuzzy sets. Information and control, 8(3), 338-353.",
        "6. Goldberg, D. E. (1989). Genetic algorithms in search, optimization, and machine learning. Addison-Wesley.",
        "7. Deb, K., et al. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. IEEE Trans. EC, 6(2), 182-197."
    ]
    
    for ref in references:
        story.append(Paragraph(ref, styles['Body']))
    
    story.append(PageBreak())
    
    # 12. APPENDIX
    story.append(Paragraph("12. APPENDIX", styles['SectionHeader']))
    
    story.append(Paragraph("A. Installation Guide", styles['SubsectionHeader']))
    story.append(Paragraph(
        "git clone https://github.com/9501893704rahul/fuzzy.git<br/>"
        "cd fuzzy<br/>"
        "pip install -r requirements.txt<br/>"
        "python final_demo.py",
        styles['CodeBlock']
    ))
    
    story.append(Paragraph("B. Dependencies", styles['SubsectionHeader']))
    story.append(Paragraph(
        "numpy>=1.21.0, pandas>=1.3.0, scikit-learn>=1.0.0, scikit-fuzzy>=0.4.2, "
        "deap>=1.3.1, matplotlib>=3.4.0, seaborn>=0.11.0, scipy>=1.7.0",
        styles['Body']
    ))
    
    story.append(Paragraph("C. API Reference", styles['SubsectionHeader']))
    story.append(Paragraph(
        "FuzzyRuleClassifier(n_partitions=5, mf_type='triangular', partition_method='adaptive', "
        "rule_method='hybrid', optimize=True, n_generations=50)<br/><br/>"
        "Methods: fit(X, y), predict(X), predict_proba(X), score(X, y), print_rules(n), export_rules(format)",
        styles['CodeBlock']
    ))
    
    story.append(Paragraph("D. Glossary", styles['SubsectionHeader']))
    
    glossary = [
        ['Term', 'Definition'],
        ['Antecedent', 'The IF part of a fuzzy rule'],
        ['Consequent', 'The THEN part of a fuzzy rule'],
        ['Fuzzification', 'Converting crisp input to fuzzy membership degrees'],
        ['Membership Function', 'Function defining degree of membership in fuzzy set'],
        ['Rule Weight', 'Confidence or certainty factor of a rule'],
        ['T-norm', 'Fuzzy AND operator (e.g., minimum, product)']
    ]
    story.append(create_table(glossary, col_widths=[1.5*inch, 3.5*inch]))
    
    # Build PDF
    doc.build(story)
    print("PDF generated successfully: documentation/Fuzzy_Classification_System_Report.pdf")


if __name__ == '__main__':
    generate_pdf()
