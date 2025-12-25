# Customer Review Sentiment Analysis System 🚀

This project is an **AI/ML-based Sentiment Analysis System** that predicts whether a customer review is **Positive or Negative** using **Word2Vec embeddings** and an **Artificial Neural Network (ANN)**.  
The trained model is deployed using **FastAPI**.

---

## 📌 Project Overview

- **Domain:** Machine Learning / NLP  
- **Model Used:** ANN (Artificial Neural Network)  
- **Text Representation:** Word2Vec  
- **Backend API:** FastAPI  
- **Deployment Ready:** Yes (FastAPI + Uvicorn)

---

## 🧠 Model Architecture

1. **Text Preprocessing**
   - Lowercasing
   - Tokenization (split by space)

2. **Feature Extraction**
   - Word2Vec converts words into vectors
   - Sentence vector = mean of all word vectors

3. **Classification**
   - ANN model predicts sentiment probability
   - Threshold-based classification

---

## 📂 Project Structure

Customer Review Sentiment Analysis System/
│
├── App/
│ ├── main.py # FastAPI application
│ ├── ann_model.h5 # Trained ANN model
│ ├── word2vec.model # Word2Vec base model
│ ├── word2vec.model.wv.vectors.npy
│ ├── word2vec.model.syn1neg.npy
│ └── requirements.txt
│
├── Notbooks/
│
├── README.md
└── .gitignore

## Install Dependencies

pip install -r requirements.txt

## ▶️ Run the FastAPI Server

cd App
python -m uvicorn main:app --reload


## Predict Sentiment

POST /predict

Request Body (JSON)
{
  "text": "This product is amazing and very useful"
}

Response
{
  "text": "This product is amazing and very useful",
  "sentiment": "Positive",
  "confidence": 0.91
}

## Model Performance

ANN Accuracy: ~88%

Logistic Regression Accuracy: ~87%

ANN selected for deployment

## 📦 Model Files Explanation

.h5 → Keras ANN model

.model → Gensim Word2Vec model

.npy → Word2Vec internal vector storage

## 🚀 Future Improvements

Add LSTM / Transformer-based model

Improve preprocessing (stopwords, lemmatization)

Deploy on Render / AWS

Add frontend UI

 👤 Author

Name: Jayswal Kunj
Field: Electronics & Communication Engineering
Interest: AI / ML / NLP
Location: India 🇮🇳

## ⭐ Acknowledgment

Thanks to open-source libraries like TensorFlow, FastAPI, and Gensim.