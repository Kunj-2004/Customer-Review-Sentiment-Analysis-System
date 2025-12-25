from fastapi import FastAPI
import numpy as np
from gensim.models import Word2Vec
from tensorflow.keras.models import load_model
from pydantic import BaseModel

class TextInput(BaseModel):
    text: str

app = FastAPI(title="Sentiment Analysis API (ANN + Word2Vec)")

# ---------------- LOAD MODELS ----------------
w2v_model = Word2Vec.load("word2vec.model")
ann_model = load_model("ann_model.h5")

VECTOR_SIZE = w2v_model.vector_size

# ---------------- PREPROCESSING ----------------
def preprocess(text: str):
    return text.lower().split()

def sentence_vector(words):
    vectors = []
    for word in words:
        if word in w2v_model.wv:
            vectors.append(w2v_model.wv[word])

    if len(vectors) == 0:
        return np.zeros(VECTOR_SIZE)

    return np.mean(vectors, axis=0)

# ---------------- ROUTES ----------------
@app.get("/")
def home():
    return {"message": "ANN + Word2Vec Sentiment API is running"}

@app.post("/predict")
def predict(data: TextInput):
    words = preprocess(data.text)
    vector = sentence_vector(words).reshape(1, -1)

    prob = ann_model.predict(vector)[0][0]
    sentiment = "Positive" if prob > 0.8 else "Negative"

    return {
        "text": data.text,
        "sentiment": sentiment,
        "confidence": round(float(prob), 3)
    }
