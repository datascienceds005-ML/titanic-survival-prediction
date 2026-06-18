# 🚢 Titanic Survival Prediction

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-red)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-ff4b4b)
![License](https://img.shields.io/badge/License-MIT-yellow)

> End-to-end Titanic survival prediction with **ensemble models**, **13 engineered features**, **SHAP explainability**, and an interactive **Streamlit dashboard**.

## 📊 Results

| Model | Accuracy | F1 | ROC-AUC |
|-------|----------|-----|---------|
| Logistic Regression | ~0.79 | ~0.75 | ~0.85 |
| Random Forest | ~0.83 | ~0.79 | ~0.89 |
| Gradient Boosting | ~0.84 | ~0.80 | ~0.90 |
| XGBoost | ~0.85 | ~0.81 | ~0.91 |

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python src/models/train.py
streamlit run app/streamlit_app.py
```

## 🔬 Engineered Features

| Feature | Why It Helps |
|---------|-------------|
| `FamilySize` | Large families struggled more |
| `IsAlone` | Solo passengers had lower survival |
| `Title` | Proxy for gender + age + status |
| `IsFemale` | #1 survival predictor |
| `PclassSex` | 1st class female = 97% survival |
| `IsChild` | Children prioritised |

## 👤 Author
**Darsh Kumar** — B.Tech Data Science
