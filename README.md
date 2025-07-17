# 🛡️ Credit Card Fraud Detection App

A machine learning-based web application that predicts whether a credit card transaction is **fraudulent** or **legitimate** based on transaction features. Built with a clean UI using **Streamlit**, and optimized for deployment.

---

## 🚀 Features

- Predicts fraud based on transaction patterns  
- Built-in UI for entering transaction values  
- Option to view sample input data  
- Uses undersampling to address class imbalance  
- Trained on cleaned dataset (reduced to 25,000 entries for size efficiency)

---

## 📊 Input Features

- V1 to V28 (Anonymized PCA-transformed features)  
- `Amount` (Transaction amount)  
- `Time` (Seconds elapsed from the first transaction)

---

## 🧠 Model Used

- **Random Forest Classifier**  
- Trained using an **undersampled** version of the credit card dataset

---

## 🧪 Technologies Used

- Python  
- Pandas, NumPy  
- scikit-learn  
- Streamlit  
- Pickle (`model.pkl` for saving the trained model)

---

## 🖥️ How to Run Locally

1. **Clone the repository**:

   ```bash
   git clone https://github.com/YOUR_USERNAME/creditcard-fraud-streamlit.git
   cd creditcard-fraud-streamlit
