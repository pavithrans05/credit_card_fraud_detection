# Credit Card Fraud Detection (Multi-Phase Analysis)

### Project Overview

This project presents an end-to-end machine learning pipeline for credit card fraud detection, showcasing a systematic approach to solving a real-world, high-stakes problem. It addresses key challenges inherent in fraud detection, including extreme class imbalance, concept drift over time, and data heterogeneity by leveraging and comparing two distinct datasets: a historical, highly imbalanced dataset from 2013, and a modern, synthetically balanced dataset from 2023.

The methodology spans five distinct stages, from exploratory data analysis (EDA) and robust preprocessing to advanced dual-dataset modeling strategies. The project culminates in a comprehensive comparative analysis of multiple models to identify the most effective and robust solution.

### Key Features & Techniques

* **Exploratory Data Analysis (EDA):** Comprehensive analysis of feature distributions, correlations, and class imbalances across heterogeneous datasets.
* **Data Preprocessing:** Feature engineering to extract temporal insights, and robust scaling techniques tailored for multi-dataset consistency.
* **Imbalanced Learning:** Implemented and evaluated various strategies including class weighting, oversampling (SMOTE), and undersampling to mitigate extreme class imbalance.
* **Concept Drift Analysis:** Quantified the impact of evolving fraud patterns by training on 2013 data and testing on 2023 data, demonstrating model degradation.
* **Advanced Dual-Dataset Modeling:** Explored and compared the effectiveness of combined training and meta-learning ensemble approaches.
* **Model Selection & Evaluation:** Rigorous comparison of multiple models (Logistic Regression, Random Forest, XGBoost) using appropriate metrics for imbalanced data (AUPRC, Precision, Recall, F1-Score).

### Project Structure

credit_card_fraud_detection/
├── data/
│   ├── ... (This folder is .gitignored to exclude large files)
├── notebooks/
│   ├── 01_eda_2013.ipynb
│   ├── 02_modeling_2013.ipynb
│   ├── 03_eda_2023.ipynb
│   ├── 04_modeling_2023.ipynb
│   ├── 05_dual_concept_drift.ipynb
│   ├── 06_dual_combined_training.ipynb
│   ├── 07_dual_ensemble_stacking.ipynb
├── scripts/
│   ├── ... (Reusable code)
└── README.md
└── .gitignore


### Methodology & Key Findings

#### **Stage 1 (2013 Dataset - Imbalanced)**

* **Objective:** To build a foundational model on a highly imbalanced dataset (0.17% fraud).
* **Key Findings:** EDA confirmed extreme class imbalance, but identified key predictive signals in PCA-transformed `V` features and temporal patterns. After preprocessing and applying class weighting, an **XGBoost Classifier** achieved an impressive AUPRC of **0.8883** (Precision: 0.87, Recall: 0.85), effectively balancing fraud detection with a low false-positive rate.

#### **Stage 2 (2023 Dataset - Balanced/Synthetic)**

* **Objective:** To analyze and model a modern, but perfectly balanced dataset (50% fraud).
* **Key Findings:** EDA revealed the dataset's synthetic nature with highly discriminative `V` features and a non-predictive `Amount` feature. Models achieved near-perfect scores, with an AUPRC of **1.0000** for Random Forest and XGBoost, demonstrating the dataset's high separability. This highlighted a key difference from real-world data challenges.

#### **Stage 3 (Concept Drift Analysis)**

* **Objective:** To quantify the impact of evolving fraud patterns.
* **Key Findings:** A model trained on the 2013 dataset experienced a catastrophic **94% drop in fraud recall** when tested on the 2023 dataset. This provided a powerful, quantifiable demonstration of **concept drift**, underscoring the necessity for continuous model monitoring and adaptation in dynamic environments.

#### **Stage 4 (Combined Training)**

* **Objective:** To create a single, robust model by training on a consolidated dataset from both 2013 and 2023.
* **Key Findings:** By combining the datasets (resulting in a 33.37% fraud ratio), a single XGBoost model achieved a perfect AUPRC of **1.0000**. This showcased the power of leveraging a larger, heterogeneous dataset where strong signals from one data source can significantly boost overall performance.

#### **Stage 5 (Ensemble Learning)**

* **Objective:** To build a robust ensemble model using a meta-learning approach.
* **Key Findings:** A meta-learning ensemble combining a 2013-trained XGBoost model with a 2023-trained Random Forest model achieved a near-perfect AUPRC of **0.9990** and an ROC AUC of **0.9998**. The meta-model's coefficients showed a higher reliance on the 2023-trained model, reflecting the greater discriminative power of the newer dataset.

### Final Recommendation

Based on the project's findings, the **XGBoost Classifier trained on the Combined (2013 + 2023) Dataset** is the recommended model. It achieved perfect discrimination (AUPRC 1.0000) with a simpler architecture than the ensemble, making it both highly effective and practical for deployment. This approach demonstrates a successful strategy for building a robust model by leveraging heterogeneous data sources.

### Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn, Imbalanced-learn
* XGBoost
* Matplotlib, Seaborn
* Joblib
* Git

### How to Run the Project

1.  **Clone the repository:** `git clone https://github.com/your-username/credit_card_fraud_detection.git`
2.  **Navigate to the project directory:** `cd credit_card_fraud_detection`
3.  **Install dependencies:** A `requirements.txt` file is recommended for a comprehensive list of dependencies. For a minimal setup, you can install the core libraries: `pip install pandas numpy scikit-learn imbalanced-learn xgboost matplotlib seaborn jupyterlab`
4.  **Download the datasets:** Download `creditcard.csv` and `creditcard_2023.csv` from Kaggle and place them in a `data/` subdirectory.
5.  **Run the notebooks:** Open the notebooks in a Jupyter environment and run them sequentially (from `01_eda_2013.ipynb` to `07_dual_ensemble_stacking.ipynb`).

---