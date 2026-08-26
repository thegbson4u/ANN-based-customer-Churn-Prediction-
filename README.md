# Customer Churn Prediction using ANN

A machine learning web application that predicts whether a customer is likely to **churn** using an Artificial Neural Network (ANN).

## 🚀 Project Overview

The model is trained on customer information such as credit score, geography, gender, age, tenure, balance, number of products, credit card ownership, active membership, and estimated salary.

The preprocessing pipeline includes:

* **Label Encoding** for Gender
* **One-Hot Encoding** for Geography
* **Standard Scaling** for numerical features

The ANN consists of:

* Dense layer with 64 neurons + ReLU
* Dense layer with 32 neurons + ReLU
* Output layer with 1 neuron + Sigmoid

The model uses **Binary Cross-Entropy loss**, the **Adam optimizer**, and **Accuracy** as a metric.

## 🛠️ Technologies

* Python
* TensorFlow/Keras
* Scikit-learn
* Pandas & NumPy
* Streamlit
* Pickle

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The application accepts customer details and returns a **churn probability** along with a prediction indicating whether the customer is likely to churn.

## 📁 Main Files

`app.py` — Streamlit application
`model.h5` — Trained ANN model
`.pkl` files — Saved preprocessing encoders and scaler
`Churn_Modelling.csv` — Dataset

## 📌 Purpose

This project demonstrates an end-to-end ANN classification workflow, from preprocessing and model training to deployment and real-time prediction.
