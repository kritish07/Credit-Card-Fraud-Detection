# Credit-Card-Fraud-Detection

## Abstract 
This study compared machine learning algorithms for credit card fraud detection on a reduced-dimensionality dataset. We evaluated Random Forest, XGBoost, and others using accuracy, AUC-ROC, and F1-score. Random Forest achieved exceptional True Positive Rates (TPR) exceeding 0.93 and AUC-ROC scores exceeding 0.94, potentially reducing False Positives compared to XGBoost. Feature importance analysis varied: Logistic Regression/XGBoost highlighted "V23," "V13," and "V16," while Random Forest emphasized "V2," "V19," and "V12." Our findings suggest Random Forest is a powerful tool for credit card fraud detection, with feature interpretation guiding model selection for specific needs.

## Introduction
Credit card fraud is a pervasive problem with significant financial repercussions for both issuers and cardholders. Fraudulent transactions can result in substantial financial losses, damage creditworthiness, and erode consumer trust in the financial system. Consequently, developing robust and effective methods for credit card fraud detection is paramount.

Machine learning (ML) techniques have emerged as powerful tools for combating credit card fraud. These algorithms can analyze vast amounts of transaction data to identify patterns and anomalies indicative of fraudulent activity. Prior research has explored various ML approaches for this purpose, achieving promising results. However, a systematic evaluation comparing different algorithms and identifying the most effective models for credit card fraud detection remains necessary.

This study addresses this gap by comprehensively investigating a diverse set of ML algorithms for credit card fraud detection. We employ a well-established credit card fraud dataset and evaluate the performance of various models using a combination of metrics, including accuracy, AUC-ROC, F1-score, precision, recall, and confusion matrices. Additionally, we explore the relationship between the most influential features and the target variable (fraudulent transaction) to gain insights into the factors driving model performance.

Our findings contribute to the ongoing effort to combat credit card fraud by:

**Performance Comparison:** Which ML algorithms outperform others in accurately classifying fraudulent transactions?

**Feature Importance:** What are the most influential features used by these models to identify fraud?

**Feature Relationship:** How do these critical features correlate with the likelihood of fraud (positive or negative relationship)?

By advancing our understanding of effective ML approaches for credit card fraud detection, this research can empower financial institutions to develop more robust fraud prevention mechanisms, ultimately safeguarding consumers and promoting a secure financial environment.

## Literature Review
Recent research in credit card fraud detection (CCFD) emphasizes the effectiveness of machine learning (ML) and deep learning techniques at identifying fraudulent transactions. For instance, works by Fu et al. (2016) and Zorion, et al. (2023) achieve high accuracy in credit card fraud detection using deep learning architectures, while acknowledging the challenges of imbalanced datasets common in fraud scenarios. Additionally, research by Bodepudi (2021) explores unsupervised anomaly detection methods to identify outliers which might represent fraudulent activity. These studies highlight the opportunities for researchers to continuously develop and refine ML algorithms to combat increasingly sophisticated fraud attempts.

## Methodology
**Dataset:** The dataset contains a subset of transactions made by European cardholders during September 2013. The original, full-sized dataset was collected during a research collaboration of the Worldline and the Machine Learning Group (http://mlg.ulb.ac.be) and ULB (Université Libre de Bruxelles) on big data mining and fraud detection. The full-sized dataset (N = 284,807) and list of researchers involved is available at https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud. A smaller dataset more amenable to working in Google Colab (N = 10,000) was compiled by Sean Humpherys and includes all the cases of fraud (n = 492) present in the original dataset and a random sampling of non-fraud cases (n = 9508). The dataset is imbalanced with only 4.92% of transactions being fraudlent. The smaller dataset is available at https://raw.githubusercontent.com/sean-humpherys/randomfilestorage/main/cc_transactions_10000.csv <br><br>
**Dependent Variable:**<br>
Class: A binary variable denotes the outcome of fraud classification. A value of '0' corresponds to a non-fraudulent transaction (negative class), and '1' represents a suspected fraudulent transaction (positive class).<br><br>
**Independent Variables:**<br>
Time: This variable represents the transaction age, measured in seconds. It indicates the time elapsed between the current transaction and the very first transaction recorded in the dataset.

V1 through V28: These variables represent a set of reduced-dimensionality features obtained through Principal Component Analysis (PCA). PCA is a technique commonly used in machine learning to reduce the number of features while preserving the most significant information. Due to privacy concerns, the original features (V1 to V28) have been transformed into these uninterpretable components. However, the relationship between these features and the target variable (Class) is preserved.

Amount: This variable represents the transaction amount in Euros. It reflects the monetary value associated with the credit card transaction.<br><br>

**Algorithms Used for Credit Card Fraud Detection:**

-   **Logistic Regression:** A linear classifier that models the probability of a transaction being fraudulent based on its features.
-   **Decision Trees:** Classify transactions by following a tree-like structure based on decision rules learned from the data. Two decision tree variants were used:
    -   **Limited Depth:** Tree stops growing after a certain depth, preventing overfitting.
    -   **Unlimited Depth:** Tree grows until all data points are separated (potentially leading to overfitting).
-   **Random Forest:** Ensemble method combining multiple decision trees, achieving higher accuracy and robustness than individual trees.
-   **XGBoost:** Another ensemble method using gradient boosting trees, known for efficiency and handling complex relationships.
-   **Support Vector Machine (SVM):** Creates a hyperplane to separate fraudulent and legitimate transactions in a high-dimensional space.
-   **Naive Bayes:** Classifies transactions based on the assumption that features are independent, efficient for large datasets.
-   **Neural Network:** A complex model inspired by the human brain, capable of learning intricate patterns but prone to overfitting if not carefully regularized.<br>

**Approach:**

Motivated by the goal of building the best possible model for credit card fraud detection, we employed two distinct approaches to evaluate various machine learning algorithms. This commitment to a rigorous evaluation process reflects our dedication to finding the most effective solution to this critical problem.

Hyperparameter-Tuned Models: The first approach focused on optimizing the internal parameters of four specific models: Random Forest, XGBoost, Isolation Forest, and SVC. By carefully tuning these hyperparameters, we aimed to extract the maximum performance potential from each model for the given credit card fraud dataset.

Pre-defined Classifiers: The second approach involved a broader exploration, evaluating a pre-defined set of classifier models. This set encompassed Logistic Regression, Decision Trees with varying depths, Random Forest, XGBoost, Support Vector Machines (SVM), Naive Bayes, and a basic Neural Network. Each model was trained and evaluated using the same training and testing data split, allowing for a direct comparison of their performance without the added complexity of individual hyperparameter tuning.

This two-pronged approach not only facilitates a comprehensive understanding of various models' strengths and weaknesses in the credit card fraud domain but also underscores our commitment to a data-driven and meticulous model selection process.<br><br>

**Performance Metrics:**

-   **Accuracy:** Measures the proportion of correctly classified transactions (both fraudulent and legitimate).
-   **AUC-ROC:** Measures the model's ability to distinguish between classes (fraudulent vs. legitimate) across all classification thresholds. A higher AUC-ROC indicates better performance.
-   **F1-score:** Combines precision (correctly predicted positives) and recall (correctly identified actual positives) into a single metric, balancing both aspects. A higher F1-score is generally better.
-   **Precision:** Measures the proportion of predicted positives that are truly positive (avoiding false positives).
-   **Recall:** Measures the proportion of actual positives that are correctly identified by the model (avoiding missing fraudulent transactions).
