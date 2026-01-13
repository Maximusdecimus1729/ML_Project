---
title: Diamond Price Predictor
emoji: 💎
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
---

<div align="center">

# 💎 Diamond Price Predictor

### An End-to-End Machine Learning Solution for Accurate Diamond Valuation

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Latest-orange.svg)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-yellow.svg)](https://huggingface.co/spaces)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Live Demo](#-live-demo) • [Features](#-features) • [Installation](#-installation) • [API Reference](#-api-reference) • [Model Details](#-model-architecture)

</div>

---

## 📖 Overview

**Diamond Price Predictor** is a production-ready machine learning application that predicts diamond prices with high accuracy based on the famous 4Cs (Carat, Cut, Color, Clarity) and physical dimensions. Built with a modular architecture following industry best practices, this project demonstrates end-to-end ML pipeline development—from data ingestion to deployment.

Whether you're a jeweler looking for quick valuations, a data scientist exploring regression techniques, or a developer learning ML deployment, this project provides a comprehensive, well-documented solution.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎯 **High Accuracy Predictions** | Ensemble learning with multiple algorithms including CatBoost, XGBoost, and Random Forest |
| 🌐 **Web Interface** | Modern, responsive UI for easy diamond price predictions |
| 🔌 **REST API** | JSON-based API endpoint for seamless integration with other applications |
| 🐳 **Docker Support** | Containerized deployment for consistent environments |
| 🤗 **Hugging Face Ready** | Pre-configured for one-click deployment to Hugging Face Spaces |
| 📊 **Modular Pipeline** | Separate components for data ingestion, transformation, and model training |
| 📝 **Comprehensive Logging** | Built-in logging and exception handling for production reliability |

---

## 🏗️ Project Architecture

```
diamond-price-predictor/
├── 📁 app.py                    # Flask application entry point
├── 📁 src/
│   ├── components/
│   │   ├── data_ingestion.py    # Data loading and train-test splitting
│   │   ├── data_transformation.py # Feature engineering & preprocessing
│   │   └── model_trainer.py     # Model training and evaluation
│   ├── pipeline/
│   │   ├── train_pipeline.py    # Training orchestration
│   │   └── predict_pipeline.py  # Inference pipeline
│   ├── exception.py             # Custom exception handling
│   ├── logger.py                # Logging configuration
│   └── utils.py                 # Utility functions
├── 📁 artifacts/                # Trained models & preprocessors
├── 📁 notebook/                 # EDA & experimentation notebooks
├── 📁 templates/                # HTML templates
├── 📁 static/                   # CSS and images
├── 📄 Dockerfile                # Container configuration
├── 📄 requirements.txt          # Python dependencies
└── 📄 setup.py                  # Package configuration
```

---

## 🔧 Installation

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Git
- Docker (optional, for containerized deployment)

### Option 1: Local Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/diamond-price-predictor.git
   cd diamond-price-predictor
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Linux/macOS
   source venv/bin/activate
   
   # On Windows
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the application**
   
   Open your browser and navigate to: `http://localhost:7860`

### Option 2: Docker Installation

1. **Build the Docker image**
   ```bash
   docker build -t diamond-price-predictor .
   ```

2. **Run the container**
   ```bash
   docker run -p 7860:7860 diamond-price-predictor
   ```

3. **Access the application**
   
   Open your browser and navigate to: `http://localhost:7860`


## 📊 Input Features

The model accepts the following diamond characteristics:

| Feature | Type | Description | Valid Values |
|---------|------|-------------|--------------|
| **Carat** | Float | Weight of the diamond | 0.1 - 10.0 |
| **Cut** | Categorical | Quality of the cut | Fair, Good, Very Good, Premium, Ideal |
| **Color** | Categorical | Diamond color grade | D, E, F, G, H, I, J (D is best) |
| **Clarity** | Categorical | Clarity grade | I1, SI2, SI1, VS2, VS1, VVS2, VVS1, IF |
| **Depth** | Float | Height percentage | 40 - 80 (%) |
| **Table** | Float | Width of top facet | 40 - 80 (%) |
| **X** | Float | Length in mm | 0 - 15 |
| **Y** | Float | Width in mm | 0 - 15 |
| **Z** | Float | Depth in mm | 0 - 15 |

---

## 🔌 API Reference

### Web Interface

Navigate to the home page and fill in the form with diamond characteristics to get instant price predictions.

### REST API Endpoint

**POST** `/predictAPI`

#### Request Body

```json
{
    "carat": 1.52,
    "cut": "Premium",
    "color": "G",
    "clarity": "VS2",
    "depth": 62.2,
    "table": 58.0,
    "x": 7.27,
    "y": 7.33,
    "z": 4.55
}
```

#### Response

```json
{
    "price": 12543.67
}
```

#### Example cURL Request

```bash
curl -X POST http://localhost:7860/predictAPI \
  -H "Content-Type: application/json" \
  -d '{
    "carat": 1.52,
    "cut": "Premium",
    "color": "G",
    "clarity": "VS2",
    "depth": 62.2,
    "table": 58.0,
    "x": 7.27,
    "y": 7.33,
    "z": 4.55
  }'
```

#### Python Integration Example

```python
import requests

url = "http://localhost:7860/predictAPI"
payload = {
    "carat": 1.52,
    "cut": "Premium",
    "color": "G",
    "clarity": "VS2",
    "depth": 62.2,
    "table": 58.0,
    "x": 7.27,
    "y": 7.33,
    "z": 4.55
}

response = requests.post(url, json=payload)
print(f"Predicted Price: ${response.json()['price']}")
```

---

## 🧠 Model Architecture

### Algorithms Evaluated

The training pipeline evaluates multiple regression algorithms:

| Algorithm | Description |
|-----------|-------------|
| Linear Regression | Baseline linear model |
| Ridge & Lasso | Regularized linear models |
| K-Neighbors Regressor | Instance-based learning |
| Decision Tree | Single tree-based model |
| Random Forest | Ensemble of decision trees |
| **CatBoost Regressor** | Gradient boosting with categorical support |
| **XGBoost Regressor** | Extreme gradient boosting |
| Gradient Boosting | Sequential ensemble method |
| AdaBoost | Adaptive boosting |

### Data Preprocessing

- **Numerical Features**: Median imputation + Standard scaling
- **Categorical Features**: Most frequent imputation + Ordinal encoding + Standard scaling

### Feature Engineering

The preprocessing pipeline handles:
- Missing value imputation
- Ordinal encoding for categorical variables with proper ordering
- Feature scaling for optimal model performance

---

## 🔄 Training Your Own Model

To retrain the model with new data:

1. **Prepare your data**
   - Place your dataset in `notebook/data/` as `gemstone.csv`
   - Ensure it has the required columns: carat, cut, color, clarity, depth, table, x, y, z, price

2. **Run the training pipeline**
   ```python
   from src.pipeline.train_pipeline import TrainPipeline
   
   pipeline = TrainPipeline()
   pipeline.run()
   ```

3. **New artifacts will be saved**
   - `artifacts/model.pkl` - Trained model
   - `artifacts/preprocessor.pkl` - Data preprocessor

---

## 📓 Notebooks

Explore the Jupyter notebooks for detailed analysis:

| Notebook | Description |
|----------|-------------|
| [1_EDA_Gemstone_price.ipynb](notebook/1_EDA_Gemstone_price.ipynb) | Exploratory Data Analysis |
| [2_Model_Training_Gemstone.ipynb](notebook/2_Model_Training_Gemstone.ipynb) | Model training experiments |
| [3_Explainability_with_LIME.ipynb](notebook/3_Explainability_with_LIME.ipynb) | Model interpretability |

---

## 🛠️ Tech Stack

- **Backend**: Flask, Gunicorn
- **ML/Data**: scikit-learn, CatBoost, XGBoost, Pandas, NumPy
- **Frontend**: HTML5, CSS3, Jinja2
- **Containerization**: Docker
- **Deployment**: Hugging Face Spaces

---

## 📁 Data

The model was trained on a comprehensive diamond dataset containing approximately 50,000+ diamonds with their prices and characteristics. The dataset includes:

- Diverse carat weights from small to large diamonds
- All quality grades for cut, color, and clarity
- Accurate dimensional measurements

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Diamond dataset sourced from public repositories
- Built with modern ML best practices
- Inspired by production-grade ML system design

---

<div align="center">

**[⬆ Back to Top](#-diamond-price-predictor)**

Made with ❤️ for the ML Community

</div>
