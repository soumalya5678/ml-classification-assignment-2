# Machine Learning Assignment - 2

## a. Problem statement

The aim of this project is to implement classification models on a public dataset, compare their performance using different evaluation metrics, and deploy the models through a Streamlit web application.

## b. Dataset description

**Dataset:** Breast Cancer Wisconsin (Diagnostic)

**Source:** UCI Machine Learning Repository.

The dataset contains 569 instances and 30 numerical input features. The target is a binary classification variable. The dataset is suitable for this assignment because it has more than 500 instances and more than 12 features.

For this project, the data was divided into 80% training data and 20% test data using `random_state=42` and stratification.

## c. Github Repository Link

Paste your GitHub repository link here after creating the repository:

`<YOUR_GITHUB_REPOSITORY_LINK>`

## d. Models used

The assignment PDF lists five named models even though one line says six models. The five specified model names are implemented here:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbour (kNN)
4. Gaussian Naive Bayes
5. Random Forest (Ensemble)

### Comparison table

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |
| kNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Random Forest | 0.9561 | 0.9931 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |

### Observations

| ML Model | Observation about model performance |
|---|---|
| Logistic Regression | It gave the best overall performance. Accuracy, AUC, F1 and MCC were all high. |
| Decision Tree | It was the weakest model among the five. It had lower accuracy and AUC compared with the other models. |
| kNN | It performed well and gave good recall and F1 score. |
| Naive Bayes | It gave good AUC, but its accuracy was slightly lower than kNN and Random Forest. |
| Random Forest | It performed well and had a very high AUC. Its accuracy was slightly below Logistic Regression on this split. |
| Overall Winner | Logistic Regression |

## Streamlit application

The Streamlit app provides:

- CSV test-data upload
- Model selection dropdown
- Accuracy, AUC, Precision, Recall, F1 and MCC
- Confusion matrix
- Prediction output

## Live Streamlit App Link

Paste the deployed Streamlit Community Cloud link here:

`<YOUR_STREAMLIT_APP_LINK>`

## Files in the repository

```text
project-folder/
|-- app.py
|-- train_models.py
|-- requirements.txt
|-- README.md
|-- test_data.csv
|-- model/
    |-- logistic_regression.pkl
    |-- decision_tree.pkl
    |-- knn.pkl
    |-- naive_bayes.pkl
    |-- random_forest.pkl
```

## How to run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then upload `test_data.csv` and select a model.

## Note

The BITS Virtual Lab execution screenshot required by the assignment must be taken by the student while running the project in BITS Virtual Lab and added to the final submission PDF.
