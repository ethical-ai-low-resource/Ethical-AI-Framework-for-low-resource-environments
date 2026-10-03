# Urdu Sentiment Analysis & Model Interpretability

An end-to-end Machine Learning pipeline designed for Urdu text classification. This repository contains the complete workflow—from dataset preprocessing and Logistic Regression model training to an interactive Streamlit application and feature importance analysis.

---

## 📌 Project Overview
Urdu is a low-resource language in computational linguistics. This project aims to build an efficient, baseline sentiment classification pipeline while providing full transparency into model decisions through feature weight analysis.

### Key Highlights
* **Preprocessing Pipeline:** Custom cleaning tailored for Urdu text features.
* **Feature Extraction:** TF-IDF Vectorization for high-dimensional text representation.
* **Classification Model:** Trainable Logistic Regression model producing probabilistic outputs.
* **Interactive Web UI:** Built with Streamlit, supporting native Right-to-Left (RTL) Urdu rendering.
* **Explainability (XAI):** Extraction of model coefficients to visualize positive and negative sentiment driver words.

---

## 🛠️ Project Structure

```text
├── app.py                      # Streamlit web application
├── train_model.py              # Model training and vectorizer pipeline
├── explain_shap.py             # Feature importance extraction script
├── paper.tex                   # Research paper LaTeX source file
├── urdu_5k_dataset.csv         # Dataset used for training and evaluation
├── urdu_model.pkl              # Serialized trained model
├── vectorizer.pkl              # Serialized TF-IDF vectorizer
└── shap_feature_importance.png # Generated feature weight visualization
