# Ensemble Learning for Stock Price Prediction using Financial Statements and News Sentiment

This repository contains the source code and resources for my undergraduate final year thesis titled: **"Ensemble Learning for Stock Price Prediction using Financial Statements and News Sentiment"**. The project was submitted as a partial fulfillment for the Bachelor's degree in Informatics at the Institut Teknologi Bandung.

Author: **Hobert Anthony Jonatan**

[![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-blue.svg)](https://www.kaggle.com/datasets/hobertj/dataset-tugas-akhir)

---

## Overview

The volatility of the stock market makes accurate price prediction a significant challenge. While many studies focus on a single factor (fundamental, technical, or sentiment), this research explores the synergy of combining **fundamental analysis** from quarterly financial reports with **market sentiment analysis** from daily news articles.

This project develops and evaluates a novel stacking ensemble model, leveraging deep learning architectures (LSTM, GRU, and Transformer) to predict the stock prices of major Indonesian banks: BBRI, BBCA, and BMRI.

## Key Features

-   **Multi-Source Data Integration:** Combines historical stock prices, quarterly financial reports, and daily news articles to create a rich dataset for prediction.
-   **Advanced Sentiment Analysis:** Utilizes transformer-based language models to convert Indonesian financial news into meaningful embedding vectors, capturing nuanced market sentiment.
-   **Ensemble Deep Learning:** Implements a stacking ensemble method that combines the predictive power of LSTM, GRU, and Transformer base models with a Multilayer Perceptron (MLP) meta-learner.
-   **Feature Importance:** Employs permutation importance to select the most impactful financial metrics, reducing noise and improving model performance.
-   **Proven Effectiveness:** The proposed model demonstrates significantly lower prediction errors compared to baseline models that use only financial data or sentiment data alone.

## Results

The primary experiments were conducted on the BBRI stock dataset, with BBCA and BMRI used for validation. The model's performance was evaluated using Mean Absolute Percentage Error (MAPE), where lower is better.

The ensemble model combining financial reports and news sentiment achieved a **MAPE of 2.33%** on BBRI test data.

### Performance Comparison on BBRI Stock [cite: 1031]

| Model Configuration | MSE | MAE | MAPE |
| :--- | :--- | :--- | :--- |
| **Ensemble (Financials + Sentiment)** | **24,328.28** | **119.53** | **2.33%** |
| Ensemble (Financials Only - Baseline) | 278,901.98 | 263.23 | 5.03% |
| Ensemble (Sentiment Only - Baseline) | 272,718.40 | 254.59 | 4.86% |

The model showed consistent and robust performance when validated on other banking stocks, confirming the effectiveness of the approach.

-   **BBCA Validation:** Achieved a MAPE of **2.89%**.
-   **BMRI Validation:** Achieved a MAPE of **2.33%**.

## Methodology Workflow

The project follows a structured methodology from data collection to final evaluation:

1.  **Data Collection**:
    -   **Financial Reports**: Quarterly reports for BBRI, BBCA, and BMRI were downloaded from official company websites.
    -   **News Articles**: Scraped from major Indonesian news portals (EmitenNews, Bisnis.com, Detikfinance).
    -   **Historical Prices**: Daily stock data (OHLC, Volume) was sourced from Investing.com.

2.  **Data Preprocessing**:
    -   News articles were converted into sentiment embeddings using the `paraphrase-multilingual-mpnet-base-v2` model, with PCA for dimensionality reduction.
    -   The 11 most important features from financial reports were selected using permutation importance.
    -   All data sources were merged based on date to create the final time-series dataset.

3.  **Model Training**:
    -   Three base models (LSTM, GRU, Transformer) were trained on the dataset
    -   A Multilayer Perceptron (MLP) was trained as a meta-learner on the predictions from the base models to produce the final stock price prediction.

4.  **Evaluation**:
    -   The final model was evaluated against baseline models on the test set using MSE, MAE, and MAPE metrics.

## Dataset

The complete dataset generated and used in this research is publicly available on Kaggle. This includes the raw data, processed features, and final merged datasets.

[**Link to Kaggle Dataset**](https://www.kaggle.com/datasets/hobertj/dataset-tugas-akhir)

## How to Use

To replicate this research or run the models, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```
2.  **Set up a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Download the dataset:**
    Download the dataset from the [Kaggle link](https://www.kaggle.com/datasets/hobertj/dataset-tugas-akhir) and place it in the `/data` directory.

5.  **Run the experiments:**
    Open the Jupyter notebooks in the `/notebooks` directory to see the step-by-step process of data preprocessing, model training, and evaluation.

## Technologies Used

-   **Python 3.x**
-   **TensorFlow & Keras** for building deep learning models.
-   **Scikit-learn** for feature selection and evaluation metrics.
-   **Pandas** for data manipulation and analysis.
-   **Transformers (Hugging Face)** for sentiment analysis.
-   **Jupyter Notebook** for experimentation and visualization.

---