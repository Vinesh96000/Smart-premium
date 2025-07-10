# 💡 SmartPremium – Predicting Insurance Costs with Machine Learning

Welcome to **SmartPremium** – a machine learning project built using Python and Streamlit to predict insurance premium costs based on user-provided information. This project is part of the GUVI AI/ML portfolio.

## 🚀 Overview

This project demonstrates the complete ML workflow:
- Data Cleaning and Preprocessing
- Feature Engineering
- Model Training
- Saving the pipeline using `joblib`
- Streamlit Web App for real-time prediction

---

## 📦 Dataset

- **Source**: Custom dataset with insurance features.
- **File**: `data/train.csv`
- **Target Variable**: `Premium Amount`

### 🔢 Sample Features
- Age, Gender, Marital Status, Education Level
- Annual Income, Credit Score, Vehicle Age
- Smoking Status, Health Score, Exercise Frequency

---

## ⚙️ Tech Stack

| Tool | Usage |
|------|-------|
| Python | Core language |
| pandas, numpy | Data manipulation |
| scikit-learn | Machine Learning |
| Streamlit | Web Application |
| joblib | Model Serialization |

---

## 🌐 Web Application

The app allows users to enter their insurance-related info and instantly get premium predictions.

### ▶️ How to Run the App Locally

```bash
# Clone the repository
git clone https://github.com/Vinesh96000/Smart-premium.git
cd Smart-premium

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
