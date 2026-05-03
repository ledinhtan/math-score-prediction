# 🎓 Student Math Score Prediction — End-to-End Machine Learning Project

End-to-end machine learning pipeline for predicting student math scores with MLOps practices.

![Python](https://img.shields.io/badge/python-3.10-3670A0?logo=python&logoColor=ffdd54)
![Kaggle](https://img.shields.io/badge/Data-Kaggle-20BEFF?logo=kaggle&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Overview

This project is an end-to-end machine learning pipeline for predicting student math performance. It is designed with a modular structure to simulate a production-ready workflow, including data ingestion, preprocessing, model training, and evaluation.

**Tech stack:** Python, scikit-learn, flask

> 🚧 This project is currently under development. Additional features such as a web application and CI/CD pipeline will be added soon.

---

## 📊 Dataset

**Source:** [Kaggle - Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)

This dataset contains 1000 student records with 5 categorical features and 3 numerical scores. It is used to analyse how factors like gender, race/ethnicity, parental education, lunch type, and test preparation course influence student performance in math, reading, and writing.

**Key features:**
- `gender`, `race_ethnicity`, `parental_level_of_education`, `lunch`, `test_preparation_course`
- `math_score`, `reading_score`, `writing_score` (range: 0–100)

**Goal:** Predict `math_score` based on the other attributes.

---

## 🗂️ Repository Structure

```bash
math-score-prediction/
├─ data/
│  └─ StudentsPerformance.csv
├─ notebooks/
│  └─ student_performance_eda.ipynb
├─ src/
│  ├─ components/
│  │  ├─ __init__.py
│  │  ├─ data_ingestion.py
│  │  ├─ data_transformation.py
│  │  └─ model_trainer.py
│  ├─ pipeline/
│  │  ├─ __init__.py
│  │  ├─ prediction_pipeline.py
│  │  └─ training_pipeline.py
│  ├─ __init__.py
│  ├─ exception.py
│  ├─ logger.py
│  └─ utils.py
├─ templates/
│  ├─ home.html
│  └─ index.html
├─ .gitignore
├─ app.py
├─ README.md
├─ requirements.txt
└─ setup.py
```

> **Note:** `artifacts/` directory is generated at runtime to store `train_dataset.csv`, `test_dataset.csv`, `preprocessor.pkl`, and `model.pkl`, and is excluded from version control.

---
