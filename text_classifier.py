from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model_mini = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

sample_texts = [
    "I am a self-taught data scientist with hands-on experience in building real-world ML systems.",
    "I developed a fraud detection pipeline on the IEEE-CIS dataset using XGBoost and SHAP.",
    "I created a retail price forecasting model using LightGBM and Prophet with 1.7M+ rows.",
    "I built a customer churn prediction web app using Flask, Random Forest, and SQLite.",
    "I specialize in EDA, feature engineering, statistical analysis, and model tuning.",
    "I have experience with anomaly detection using Isolation Forest, LOF, DBSCAN, and clustering.",
    "I am learning about LLMs, embeddings, vector search, and RAG pipelines.",
    "I use Python, SQL, and Scikit-learn extensively in my ML workflows.",
    "I integrate SHAP for explainability in my ML models.",
    "I enjoy learning deep learning, semantic embeddings, and vector similarity search."
]


categories = [
    "General Machine Learning",
    "Fraud Detection",
    "Time Series Forecasting",
    "Churn Prediction",
    "EDA & Feature Engineering",
    "Anomaly Detection",
    "LLMs & RAG",
    "ML Development Tools",
    "Explainable AI",
    "Deep Learning & Embeddings"
]
def classify_input(user_input, model):
    emb_input = model.encode(user_input)
    db_emb = model.encode(sample_texts)

    sims = cosine_similarity([emb_input], db_emb)[0]
    top_idx = np.argmax(sims)
    return categories[top_idx]

def main():
    print("Text Classification Based on embeddings CLI")

    while True:
        user_text = input("\nEnter text to classify (or 'exit' to quit): ")
        if user_text.lower() == 'exit':
            break

        category = classify_input(user_text, model_mini)
        print(f"Predicted Category: {category}")

if __name__ == "__main__":

    main()
