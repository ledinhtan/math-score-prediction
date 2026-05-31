# 🎓 Student Math Score Prediction — End-to-End Machine Learning Project

End-to-end machine learning pipeline for predicting student math scores with MLOps practices.

![Python](https://img.shields.io/badge/python-3.10-3670A0?logo=python&logoColor=ffdd54)
![Flask](https://img.shields.io/badge/flask-%23000.svg?logo=flask&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?logo=docker&logoColor=white)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?logo=githubactions&logoColor=white)
![Infrastructure](https://img.shields.io/badge/Infrastructure-Render-46E3B7?logo=render&logoColor=white)

---

## 📌 Overview

This project is an end-to-end machine learning pipeline for predicting student math performance. It is designed with a modular structure to simulate a production-ready workflow, including data ingestion, preprocessing, model training, and evaluation.

**Tech stack:** Python, Scikit-learn, Optuna, MLflow, Flask

---

## 📊 Dataset

**Source:** [Kaggle - Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)

This dataset contains 1000 student records with 5 categorical features and 3 numerical scores. It is used to analyse how factors like gender, race/ethnicity, parental education, lunch type, and test preparation course influence student performance in math, reading, and writing.

**Key features:**
- `gender`, `race_ethnicity`, `parental_level_of_education`, `lunch`, `test_preparation_course`
- `math_score`, `reading_score`, `writing_score` (range: 0–100)

**Goal:** Predict `math_score` based on the other attributes.

---

## ✨ Features

- End-to-end ML pipeline (ingestion → preprocessing → training → evaluation)
- Optuna hyperparameter tuning with MLflow experiment tracking
- YAML-driven pipeline configuration for production-ready flexibility
- Modular and production-ready project structure
- Web application for real-time score prediction
- CI/CD pipeline using GitHub Actions + Render Deploy Hook
- Dockerised for consistent deployment

---

##   Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the training pipeline:
   ```bash
   python -m src.pipeline.training_pipeline
   ```
3. Start the Flask app:
   ```bash
   python app.py
   ```

---

##   ️ Repository Structure

```bash
math-score-prediction/
├─ .github/
│  └─ workflows/
│     └─ main.yaml
├─ config/
│  └─ config.yaml
├─ data/
│  └─ StudentsPerformance.csv
├─ models/
│  ├─ model.pkl
│  └─ preprocessor.pkl
├─ notebooks/
│  └─ student_performance_eda.ipynb
├─ src/
│  ├─ components/
│  │  ├─ __init__.py
│  │  ├─ data_ingestion.py
│  │  ├─ data_transformation.py
│  │  └─ model_trainer.py
│  ├─ config/
│  │  ├─ __init__.py
│  │  └─ configuration.py
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
├─ .dockerignore
├─ .gitignore
├─ app.py
├─ Dockerfile
├─ README.md
├─ requirements.txt
└─ setup.py
```

> **Note:** `artifacts/` directory is generated at runtime to store `raw_dataset.csv`, `train_dataset.csv`, and `test_dataset.csv`, and is excluded from version control.

---

## 🔄 CI/CD Pipeline

This project includes a lightweight CI/CD workflow using GitHub Actions and Render Deploy Hook:

- **CI:** Runs on every push to `main`, installs dependencies, and builds the Docker image.
- **CD:** After CI passes, GitHub Actions triggers Render Deploy Hook to automatically deploy the app.
- Ensures consistent, automated deployment with no manual steps required.